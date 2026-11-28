# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Checkpoint-engine integration for SGLang.
This module provides weight update functionality via IPC for checkpoint-engine compatibility,
including P1 fast IPC (one-shot marshal manifest handshake, pre-built _foreach_copy_ MoE plan,
and dedicated CUDA stream synchronization).
"""

import gc
import logging
import marshal
import os
import pickle
import struct
import threading
import time
import traceback
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import zmq

try:
    from checkpoint_engine.worker import (
        _extract_weights,
        _rebuild_ipc,
        update_weights_from_ipc,
    )
except ImportError:
    raise ImportError(
        "checkpoint-engine is not installed. "
        "Please install it with: pip install sglang[checkpoint-engine]"
    )

logger = logging.getLogger(__name__)

# Bounded LINGER for the IPC context: if the loader peer disappears while our
# final ack is still undelivered, context teardown waits at most this long
# instead of blocking the scheduler thread forever in zmq ctx.term().
CE_IPC_LINGER_MS = max(0, int(os.getenv("SGLANG_CE_IPC_LINGER_MS", "5000")))

FAST_IPC_MAGIC = b"SGBK"
FAST_IPC_MANIFEST_MAGIC = b"SGMF"

_MOE_PROJ_TO_PARAM = {
    "gate_proj": ("w13_", "w1"),
    "up_proj": ("w13_", "w3"),
    "down_proj": ("w2_", "w2"),
}
_SHARD_ID_TO_DIM = {"w1": 0, "w2": 1, "w3": 0}
_WNA16_MOE_METHODS = (
    "CompressedTensorsWNA16MarlinMoE",
    "CompressedTensorsWNA16MoE",
    "CompressedTensorsWNA16TritonMoE",
)

_DTYPE_CACHE: Dict[str, torch.dtype] = {}
for _dt_name in (
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
    "float8_e4m3fn",
    "float8_e5m2",
):
    _dt_val = getattr(torch, _dt_name, None)
    if isinstance(_dt_val, torch.dtype):
        _DTYPE_CACHE[_dt_name] = _dt_val
        _DTYPE_CACHE[f"torch.{_dt_name}"] = _dt_val


def _str_to_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    dt = _DTYPE_CACHE.get(value)
    if dt is not None:
        return dt
    s = str(value)
    if s.startswith("torch."):
        s = s.split(".", 1)[1]
    dtype = getattr(torch, s, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype: {value!r}")
    _DTYPE_CACHE[value] = dtype
    return dtype


def _resolve_fast_moe_copies(
    name: str,
    dtype: torch.dtype,
    shape_tuple: Tuple[int, ...],
    buf_typed: torch.Tensor,
    buf_device: torch.device,
    abs_offset: int,
    moe_info_map: Dict[str, Tuple[Any, ...]],
    params_dict: Dict[str, torch.nn.Parameter],
    cpu_map_py: Optional[List[List[List[int]]]],
    num_logical_experts: int,
    fallback_reasons: Counter,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Pre-resolve a FusedMoE compressed-tensors WNA16 weight copy using single-op as_strided.

    Returns:
      - list of (dst_slice, src_slice) pairs (empty list if remote EP expert -> skipped)
      - None if the tensor is outside the strictly verified fast path (caller falls back)
    """
    exp_idx = name.find(".experts.")
    if exp_idx == -1:
        fallback_reasons["non_moe"] += 1
        return None
    mod_prefix = name[: exp_idx + 8]  # len(".experts") == 8
    info = moe_info_map.get(mod_prefix)
    if info is None:
        fallback_reasons["mod_not_matched"] += 1
        return None

    (
        moe,
        layer_id,
        moe_tp_size,
        moe_tp_rank,
        is_gated,
        switch_w13,
        num_local_experts,
        has_fused_shared,
        param_cache,
    ) = info

    # Parse "<expert_id>.<proj>.<leaf>"
    rest = name[exp_idx + 9 :]  # len(".experts.") == 9
    parts = rest.split(".", 2)
    if len(parts) != 3 or not parts[0].isdigit():
        fallback_reasons["bad_name_format"] += 1
        return None
    expert_id = int(parts[0])
    proj_info = _MOE_PROJ_TO_PARAM.get(parts[1])
    if proj_info is None:
        fallback_reasons["unknown_proj"] += 1
        return None
    param_prefix, shard_id = proj_info
    leaf = parts[2]
    if "zero" in leaf or "input_scale" in leaf or "g_idx" in leaf or "bias" in leaf:
        fallback_reasons[f"leaf_{leaf}"] += 1
        return None

    param_key = param_prefix + leaf
    p_entry = param_cache.get(param_key)
    if p_entry is None:
        target_name = f"{mod_prefix}.{param_key}"
        param = params_dict.get(target_name)
        if param is None:
            fallback_reasons["param_missing"] += 1
            return None
        pdata = param.data
        p_entry = (
            pdata,
            tuple(pdata.shape),
            tuple(pdata.stride()),
            int(pdata.storage_offset()),
            pdata.dtype,
            pdata.device,
            getattr(param, "_sglang_require_global_experts", False),
            getattr(param, "is_transposed", False),
            getattr(param, "quant_method", None),
        )
        param_cache[param_key] = p_entry

    (
        pdata,
        p_shape,
        p_stride,
        p_offset,
        p_dtype,
        p_device,
        require_global_experts,
        is_transposed,
        p_qmethod,
    ) = p_entry

    if p_device != buf_device:
        fallback_reasons["device_mismatch"] += 1
        return None

    itemsize = dtype.itemsize
    if abs_offset % itemsize != 0:
        fallback_reasons["unaligned_offset"] += 1
        return None
    elem_off = abs_offset // itemsize

    # Resolve logical expert_id -> local physical expert IDs (exact match to FusedMoE.weight_loader)
    if cpu_map_py is None:
        if not require_global_experts:
            local_eid = moe._map_global_expert_id_to_local_expert_id(expert_id)
            if local_eid == -1:
                return []
            local_eids = (local_eid,)
        else:
            local_eids = (expert_id,)
    else:
        if has_fused_shared and expert_id >= num_logical_experts:
            fallback_reasons["fused_shared"] += 1
            return None
        if layer_id >= len(cpu_map_py):
            if require_global_experts:
                num_phys = len(cpu_map_py[0][0]) if cpu_map_py and cpu_map_py[0] else num_logical_experts
                phys_ids = range(expert_id, num_phys, num_logical_experts)
            else:
                phys_ids = (expert_id,)
        elif require_global_experts:
            num_phys = len(cpu_map_py[layer_id][0])
            phys_ids = range(expert_id, num_phys, num_logical_experts)
        else:
            row = cpu_map_py[layer_id][expert_id]
            phys_ids = [pe for pe in row if pe != -1]

        local_list: List[int] = []
        for pe in phys_ids:
            if not require_global_experts:
                le = moe._map_global_expert_id_to_local_expert_id(pe)
                if le < 0 or le >= num_local_experts:
                    continue
                local_list.append(le)
            else:
                local_list.append(pe)
        if not local_list:
            return []
        local_eids = local_list

    if leaf == "weight_shape":
        # w13_weight_shape / w2_weight_shape are allocated in model dtype (e.g. bf16)
        # while checkpoint weight_shape is int32; _foreach_copy_ supports cross-dtype natively.
        if len(shape_tuple) != 1 or p_shape[1:] != shape_tuple:
            fallback_reasons["shape_mismatch_wshape"] += 1
            return None
        raw_view = buf_typed.as_strided(shape_tuple, (1,), elem_off)
        s_exp = p_stride[0]
        dst_stride = p_stride[1:]
        return [
            (pdata.as_strided(shape_tuple, dst_stride, p_offset + le * s_exp), raw_view)
            for le in local_eids
        ]

    # Only accelerate 2D weight_packed / group-or-block weight_scale tensors
    if len(shape_tuple) != 2 or len(p_shape) != 3:
        fallback_reasons["ndim_not_2"] += 1
        return None
    if p_dtype != dtype:
        fallback_reasons[f"dtype_{leaf}"] += 1
        return None
    if "scale" in leaf:
        if p_qmethod not in ("group", "block"):
            fallback_reasons[f"scale_qmethod_{p_qmethod}"] += 1
            return None
    elif "weight" not in leaf:
        fallback_reasons["not_weight"] += 1
        return None

    # Fused single-op as_strided for: buffer.narrow().view(dtype).view(R, C).t().narrow(shard_dim, ...)
    R, C = shape_tuple
    shard_dim = _SHARD_ID_TO_DIM[shard_id]
    if is_transposed:
        shard_dim = 1 - shard_dim

    if shard_dim == 0:
        loaded_shard_size = C // moe_tp_size
        if loaded_shard_size <= 0 or loaded_shard_size * moe_tp_size != C:
            fallback_reasons["bad_tp_shard"] += 1
            return None
        src_shape = (loaded_shard_size, R)
        src_slice = buf_typed.as_strided(
            src_shape, (1, C), elem_off + loaded_shard_size * moe_tp_rank
        )
    else:
        loaded_shard_size = R // moe_tp_size
        if loaded_shard_size <= 0 or loaded_shard_size * moe_tp_size != R:
            fallback_reasons["bad_tp_shard"] += 1
            return None
        src_shape = (C, loaded_shard_size)
        src_slice = buf_typed.as_strided(
            src_shape, (1, C), elem_off + loaded_shard_size * moe_tp_rank * C
        )

    d0, d1 = p_shape[1], p_shape[2]
    s_exp, s0, s1 = p_stride[0], p_stride[1], p_stride[2]
    is_w13 = shard_id in ("w1", "w3")
    if is_w13:
        use_second = (
            (switch_w13 and shard_id == "w1") or (not switch_w13 and shard_id == "w3")
        ) and is_gated
        if shard_dim == 0:
            shard_size = d0 // 2 if is_gated else d0
            start0 = shard_size if use_second else 0
            len0 = src_shape[0] if src_shape[0] < shard_size else shard_size
            start1, len1 = 0, d1
        else:
            shard_size = d1 // 2 if is_gated else d1
            start1 = shard_size if use_second else 0
            len1 = src_shape[1] if src_shape[1] < shard_size else shard_size
            start0, len0 = 0, d0
    else:
        if shard_dim == 0:
            start0, len0 = 0, (src_shape[0] if src_shape[0] < d0 else d0)
            start1, len1 = 0, d1
        else:
            start0, len0 = 0, d0
            start1, len1 = 0, (src_shape[1] if src_shape[1] < d1 else d1)

    if (len0, len1) != src_shape:
        fallback_reasons[f"mismatch_{leaf}"] += 1
        return None

    base_off = p_offset + start0 * s0 + start1 * s1
    dst_stride = (s0, s1)
    return [
        (pdata.as_strided(src_shape, dst_stride, base_off + le * s_exp), src_slice)
        for le in local_eids
    ]


class _AsyncFastIPCPlan:
    """Paced background compiler using single-op as_strided (~0.7s total, ~3ms/bucket).

    Uses a sliding lookahead window (6 buckets ahead of applied_idx) so CPU work
    runs during the idle socket.recv() gap between buckets with zero GIL contention
    and zero Phase 4 loader interference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        buffer: torch.Tensor,
        max_bucket_size: int,
        num_slots: int,
        buckets: List[List[Tuple[str, str, Tuple[int, ...], int]]],
        device_id: int,
    ):
        t0 = time.perf_counter()
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

        self.max_bucket_size = max_bucket_size
        self.num_slots = num_slots
        self._buffer = buffer
        self._buf_device = buffer.device
        self._buf_views: Dict[torch.dtype, torch.Tensor] = {}
        self._buckets = buckets
        self._device_id = device_id
        self._num_buckets = len(buckets)
        self._plans: List[Optional[Dict[str, Any]]] = [None] * self._num_buckets
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._applied_idx = -1
        self._lookahead = 6
        self._error: Optional[BaseException] = None

        global_expert_meta = get_global_expert_location_metadata()
        self._cpu_map_py: Optional[List[List[List[int]]]] = None
        self._num_logical_experts = 0
        if global_expert_meta is not None:
            self._cpu_map_py = global_expert_meta.logical_to_all_physical_map_cpu.tolist()
            self._num_logical_experts = int(global_expert_meta.num_logical_experts)

        self._moe_info_map: Dict[str, Tuple[Any, ...]] = {}
        for mod_name, mod in model.named_modules():
            if not isinstance(mod, FusedMoE):
                continue
            if (
                getattr(mod, "use_padded_loading", False)
                or getattr(mod, "use_presharded_weights", False)
                or getattr(mod, "use_triton_kernels", False)
                or getattr(mod, "use_flashinfer_trtllm_moe", False)
            ):
                continue
            method = getattr(mod, "scheme", None)
            if method is None:
                method = getattr(mod, "quant_method", None)
            if method is None or method.__class__.__name__ == "KTEPWrapperMethod":
                continue
            if method.__class__.__name__ not in _WNA16_MOE_METHODS:
                continue
            self._moe_info_map[mod_name] = (
                mod,
                getattr(mod, "layer_id", 0),
                mod.moe_tp_size,
                mod.moe_tp_rank,
                mod.moe_runner_config.is_gated,
                getattr(mod.quant_method, "load_up_proj_weight_first", False),
                mod.num_local_experts,
                getattr(mod, "_has_fused_shared", False),
                {},  # per-module param_cache
            )

        self._params_dict = dict(model.named_parameters(remove_duplicate=False))
        self._init_dt = time.perf_counter() - t0
        self._thread = threading.Thread(
            target=self._compile_loop, name="fast-ipc-compiler", daemon=True
        )
        self._thread.start()

    def _get_buf_view(self, dtype: torch.dtype) -> torch.Tensor:
        v = self._buf_views.get(dtype)
        if v is None:
            v = self._buffer.view(dtype)
            self._buf_views[dtype] = v
        return v

    def _compile_loop(self) -> None:
        t0 = time.perf_counter()
        cpu_work_s = 0.0
        fallback_reasons: Counter = Counter()
        n_total = 0
        n_fast = 0
        n_skipped = 0
        try:
            for b_idx, items in enumerate(self._buckets):
                # Paced lookahead: stay at most `_lookahead` buckets ahead of worker apply
                # so CPU view creation happens during idle recv gaps with zero GIL contention.
                if b_idx > self._applied_idx + self._lookahead:
                    with self._cond:
                        while (
                            b_idx > self._applied_idx + self._lookahead
                            and not self._stop.is_set()
                        ):
                            self._cond.wait(timeout=0.05)
                if self._stop.is_set():
                    return

                tb0 = time.perf_counter()
                prebuilt_slot = b_idx % self.num_slots
                slot_offset = prebuilt_slot * self.max_bucket_size
                dst_list: List[torch.Tensor] = []
                src_list: List[torch.Tensor] = []
                fallback_specs: List[Tuple[str, torch.dtype, Tuple[int, ...], int, int]] = []

                for name, dtype_str, shape_tuple, item_offset in items:
                    n_total += 1
                    dtype = _str_to_dtype(dtype_str)
                    buf_typed = self._get_buf_view(dtype)
                    pairs = _resolve_fast_moe_copies(
                        name,
                        dtype,
                        shape_tuple,
                        buf_typed,
                        self._buf_device,
                        slot_offset + item_offset,
                        self._moe_info_map,
                        self._params_dict,
                        self._cpu_map_py,
                        self._num_logical_experts,
                        fallback_reasons,
                    )
                    if pairs is not None:
                        if not pairs:
                            n_skipped += 1
                        else:
                            for dst_t, src_t in pairs:
                                dst_list.append(dst_t)
                                src_list.append(src_t)
                            n_fast += 1
                    else:
                        numel = 1
                        for d in shape_tuple:
                            numel *= d
                        fallback_specs.append(
                            (name, dtype, shape_tuple, item_offset, dtype.itemsize * numel)
                        )

                cpu_work_s += time.perf_counter() - tb0
                plan = {
                    "prebuilt_slot": prebuilt_slot,
                    "dst_list": dst_list,
                    "src_list": src_list,
                    "fallback_specs": fallback_specs,
                    "raw_items": items,
                }
                with self._cond:
                    self._plans[b_idx] = plan
                    self._cond.notify_all()

            dt = time.perf_counter() - t0
            if self._device_id == 0:
                pct = (n_fast + n_skipped) * 100.0 / max(n_total, 1)
                logger.info(
                    "Fast IPC paced as_strided plan compiled in %.2fs CPU (%.2fs wall, init %.2fs): "
                    "%d/%d tensors (%.1f%%) accelerated via _foreach_copy_ "
                    "(%d moe_modules, %d skipped, %d fallback, reasons=%s)",
                    cpu_work_s,
                    dt,
                    self._init_dt,
                    n_fast,
                    n_total,
                    pct,
                    len(self._moe_info_map),
                    n_skipped,
                    n_total - n_fast - n_skipped,
                    dict(fallback_reasons.most_common(6)),
                )
        except BaseException as exc:  # noqa: BLE001
            with self._cond:
                self._error = exc
                self._cond.notify_all()

    def get_bucket_plan(self, b_idx: int) -> Dict[str, Any]:
        plan = self._plans[b_idx]
        if plan is not None:
            return plan
        with self._cond:
            while self._plans[b_idx] is None and self._error is None:
                self._cond.wait()
            if self._error is not None:
                raise self._error
            return self._plans[b_idx]  # type: ignore[return-value]

    def release_bucket(self, b_idx: int) -> None:
        # Drop tensor views for finished buckets promptly and advance lookahead window.
        self._plans[b_idx] = None
        with self._cond:
            if b_idx > self._applied_idx:
                self._applied_idx = b_idx
                self._cond.notify_all()

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=30.0)
        self._plans.clear()
        self._buf_views.clear()
        self._buffer = None  # type: ignore[assignment]
        self._buckets = []
        self._moe_info_map.clear()
        self._params_dict.clear()


def _update_weights_from_ipc_fast(
    zmq_ctx: zmq.Context,
    zmq_handle: str,
    device_id: int,
    *,
    model: Optional[torch.nn.Module],
    run: Callable[[List[Tuple[str, torch.Tensor]]], None],
    post_hook: Optional[Callable[[], None]] = None,
) -> None:
    """Enhanced checkpoint-engine IPC loop with P1 fast manifest + _foreach_copy_ + apply_stream."""
    socket = zmq_ctx.socket(zmq.REP)
    socket.connect(zmq_handle)
    buffer: Optional[torch.Tensor] = None
    ipc_handle = None
    fast_plan: Optional[_AsyncFastIPCPlan] = None
    apply_stream: Optional[torch.cuda.Stream] = None
    slot_event: Optional[torch.cuda.Event] = None

    try:
        ipc_handle = socket.recv_pyobj()
        assert isinstance(ipc_handle, tuple)
        buffer = _rebuild_ipc(ipc_handle, device_id)
        assert buffer.dtype == torch.uint8
        socket.send(b"")
    except Exception as e:
        msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        socket.send_string(msg)
        socket.recv()  # wait for ack
        raise

    try:
        released = False
        while True:
            raw = socket.recv()
            # Fast-path binary bucket trigger: 12 bytes (b"SGBK" + uint32 idx + uint32 slot_id)
            if (
                not released
                and fast_plan is not None
                and len(raw) == 12
                and raw[:4] == FAST_IPC_MAGIC
            ):
                try:
                    b_idx, slot_id = struct.unpack("<II", raw[4:12])
                    plan = fast_plan.get_bucket_plan(b_idx)
                    slot_offset = slot_id * fast_plan.max_bucket_size

                    if apply_stream is None:
                        apply_stream = torch.cuda.Stream(device=device_id)
                        slot_event = torch.cuda.Event()

                    with torch.cuda.stream(apply_stream):
                        if slot_id == plan["prebuilt_slot"]:
                            dst_list = plan["dst_list"]
                            if dst_list:
                                torch._foreach_copy_(dst_list, plan["src_list"])
                            fb_specs = plan["fallback_specs"]
                            if fb_specs:
                                fb_weights = [
                                    (
                                        nm,
                                        buffer.narrow(0, slot_offset + off, nb)
                                        .view(dt)
                                        .view(sh),
                                    )
                                    for nm, dt, sh, off, nb in fb_specs
                                ]
                                run(fb_weights)
                        else:
                            dyn_weights = []
                            for nm, dt_s, sh_t, off in plan["raw_items"]:
                                dt = _str_to_dtype(dt_s)
                                numel = 1
                                for d in sh_t:
                                    numel *= d
                                nb = dt.itemsize * numel
                                dyn_weights.append(
                                    (
                                        nm,
                                        buffer.narrow(0, slot_offset + off, nb)
                                        .view(dt)
                                        .view(sh_t),
                                    )
                                )
                            run(dyn_weights)

                    # Sync only apply_stream via lightweight CUDA Event so the ring-buffer
                    # slot is released immediately (~3.6ms) without starving the 4-stage pipeline.
                    slot_event.record(apply_stream)
                    slot_event.synchronize()
                    fast_plan.release_bucket(b_idx)
                    socket.send(b"")
                except Exception as e:  # noqa: BLE001
                    msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    socket.send_string(msg)
                continue

            # Fast-path marshal manifest handshake: b"SGMF" + marshal.dumps((max_bucket_size, num_slots, buckets))
            if not released and len(raw) > 4 and raw[:4] == FAST_IPC_MANIFEST_MAGIC:
                try:
                    assert buffer is not None
                    max_bsize, n_slots, buckets_meta = marshal.loads(raw[4:])
                    if model is not None:
                        fast_plan = _AsyncFastIPCPlan(
                            model, buffer, max_bsize, n_slots, buckets_meta, device_id
                        )
                    socket.send(b"")
                except Exception as e:  # noqa: BLE001
                    msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    socket.send_string(msg)
                continue

            # Standard pickled control or legacy payload messages
            payload = pickle.loads(raw)

            if released:
                assert payload is None, "Should not receive any payload after released"
                if post_hook is not None:
                    post_hook()
                torch.cuda.synchronize(device_id)
                socket.send(b"")
                break

            if payload is None:  # None #1: synchronize & release IPC buffer
                if apply_stream is not None:
                    apply_stream.synchronize()
                torch.cuda.synchronize(device_id)
                released = True
                # Join background compiler and drop ALL views referencing `buffer` before ipc_collect()!
                if fast_plan is not None:
                    fast_plan.close()
                    fast_plan = None
                slot_event = None
                apply_stream = None
                buffer = None
                del ipc_handle
                ipc_handle = None

                gc.collect()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device_id)
                socket.send(b"")
                continue

            if isinstance(payload, dict) and payload.get("__sglang_fast_ipc__") == "v1":
                try:
                    assert buffer is not None
                    if model is not None:
                        fast_plan = _AsyncFastIPCPlan(
                            model,
                            buffer,
                            payload["max_bucket_size"],
                            payload["num_slots"],
                            payload["buckets"],
                            device_id,
                        )
                    socket.send(b"")
                except Exception as e:  # noqa: BLE001
                    msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    socket.send_string(msg)
                continue

            if isinstance(payload, list):  # Legacy per-bucket metadata path
                try:
                    run(_extract_weights(payload, buffer))
                    torch.cuda.synchronize(device_id)
                    socket.send(b"")
                except Exception as e:  # noqa: BLE001
                    msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    socket.send_string(msg)
            elif isinstance(payload, Exception):
                raise payload
            else:
                raise RuntimeError(f"Unexpected IPC payload type: {type(payload)}")
    finally:
        if fast_plan is not None:
            fast_plan.close()
            fast_plan = None
        slot_event = None
        apply_stream = None
        buffer = None
        socket.close()


class SGLangCheckpointEngineWorkerExtension:
    """Worker extension for SGLang to support checkpoint-engine IPC weight updates."""

    def __init__(self):
        self._zmq_ctx: Optional[zmq.Context] = None

    def get_device_uuid(self) -> str:
        raise NotImplementedError(
            "This method should be overridden by SGLang integration"
        )

    def get_device_id(self) -> int:
        raise NotImplementedError(
            "This method should be overridden by SGLang integration"
        )

    def get_model_loader(self) -> Callable:
        raise NotImplementedError(
            "This method should be overridden by SGLang integration"
        )

    def get_post_hook(self) -> Optional[Callable]:
        return None

    def update_weights_from_ipc(self, zmq_handles: Dict[str, str]):
        """Update weights from IPC communication."""
        device_uuid = self.get_device_uuid()
        device_id = self.get_device_id()
        if device_uuid not in zmq_handles:
            raise ValueError(
                f"Device UUID {device_uuid} not found in zmq_handles: {list(zmq_handles.keys())}"
            )
        ctx = zmq.Context()
        ctx.setsockopt(zmq.LINGER, CE_IPC_LINGER_MS)
        try:
            model = getattr(getattr(self, "model_runner", None), "model", None)
            if model is not None:
                from sglang.srt.model_loader.utils import restore_weights_before_loading

                restore_weights_before_loading(model)
                model._sglang_weight_load_session = {}
            try:
                _update_weights_from_ipc_fast(
                    ctx,
                    zmq_handles[device_uuid],
                    device_id=device_id,
                    model=model,
                    run=self.get_model_loader(),
                    post_hook=self.get_post_hook(),
                )
            finally:
                if model is not None and hasattr(model, "_sglang_weight_load_session"):
                    delattr(model, "_sglang_weight_load_session")
        finally:
            start = time.monotonic()
            ctx.destroy(linger=CE_IPC_LINGER_MS)
            waited = time.monotonic() - start
            if waited > 1.0:
                logger.warning(
                    "checkpoint-engine IPC context teardown waited %.1fs; "
                    "the loader peer likely closed before reading our final "
                    "ack (undelivered message dropped after bounded linger)",
                    waited,
                )


class SGLangCheckpointEngineWorkerExtensionImpl(SGLangCheckpointEngineWorkerExtension):
    """Concrete implementation of SGLangCheckpointEngineWorkerExtension for ModelRunner."""

    def __init__(self, model_runner):
        super().__init__()
        self.model_runner = model_runner

    def get_device_uuid(self) -> str:
        device_id = torch.cuda.current_device()
        try:
            return f"GPU-{torch.cuda.get_device_properties(device_id).uuid!s}"
        except AssertionError as e:
            raise ValueError(f"Failed to get GPU UUID for device {device_id}") from e

    def get_device_id(self) -> int:
        return torch.cuda.current_device()

    def get_model_loader(self) -> Callable:
        def model_loader(weights):
            if not hasattr(self.model_runner.model, "_sglang_weight_load_session"):
                from sglang.srt.model_loader.utils import restore_weights_before_loading

                restore_weights_before_loading(self.model_runner.model)
            return self.model_runner.model.load_weights(weights)

        return model_loader

    def get_post_hook(self) -> Optional[Callable]:
        def post_hook():
            try:
                from sglang.srt.model_loader.loader import device_loading_context

                for _, module in self.model_runner.model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        target_device = torch.device(
                            "cuda", torch.cuda.current_device()
                        )
                        with device_loading_context(module, target_device):
                            quant_method.process_weights_after_loading(module)
                if hasattr(self.model_runner.model, "post_load_weights"):
                    self.model_runner.model.post_load_weights()
            except Exception:
                logger.exception("Post-hook processing failed; failing IPC update")
                raise

        return post_hook
