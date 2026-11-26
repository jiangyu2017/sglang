"""Offline CPU-only unit tests for group-aware atomic bucketing in dump_to_mooncake."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from typing import Dict, List, Tuple

import torch

from sglang.srt.checkpoint_engine.dump_to_mooncake import (
    ALIGN_SIZE,
    TensorRecord,
    _align_size,
    _atomic_group_key,
    _collect_tensor_records,
    _fill_bucket_from_records,
    _list_safetensors_files,
    _partition_records,
    _verify_atomic_partitions,
)
from sglang.srt.checkpoint_engine.mooncake_loader import _validate_manifest_layout


def _make_record(name: str, shape: List[int], dtype: torch.dtype = torch.bfloat16) -> TensorRecord:
    nbytes = dtype.itemsize * int(torch.Size(shape).numel())
    return TensorRecord(
        name=name,
        dtype=dtype,
        shape=shape,
        size_aligned=_align_size(dtype, torch.Size(shape)),
        path="/dummy.safetensors",
        file_offset=0,
        nbytes=nbytes,
    )


def _bucket_index_map(partitions: List[Tuple[List[TensorRecord], int]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for b_idx, (recs, _) in enumerate(partitions):
        for r in recs:
            mapping[r.name] = b_idx
    return mapping


def _write_safetensors_file(path: str, tensors: Dict[str, torch.Tensor]) -> None:
    dtype_to_st = {
        torch.float32: "F32",
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.int32: "I32",
        torch.uint8: "U8",
        torch.float8_e4m3fn: "F8_E4M3",
    }
    header: Dict[str, object] = {}
    raw_chunks: List[bytes] = []
    offset = 0
    for name, t in tensors.items():
        t_contig = t.contiguous().cpu()
        raw = t_contig.view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": dtype_to_st[t_contig.dtype],
            "shape": list(t_contig.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        raw_chunks.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk in raw_chunks:
            f.write(chunk)


class TestAtomicGroupBucketing(unittest.TestCase):
    def test_mla_qkv_a_proj_kept_in_same_bucket_across_boundary(self):
        # Alphabetically, kv_b_proj and o_proj sit between kv_a_proj_with_mqa and q_a_proj.
        # With bucket_size_bytes = 2048, naive lexicographical packing would split them.
        recs = [
            _make_record("language_model.model.layers.17.self_attn.kv_a_proj_with_mqa.weight", [16, 16]),  # 512B
            _make_record("language_model.model.layers.17.self_attn.kv_a_proj_with_mqa.weight_scale_inv", [1, 1], torch.float32),  # 256B aligned
            _make_record("language_model.model.layers.17.self_attn.kv_b_proj.weight", [32, 32]),  # 2048B
            _make_record("language_model.model.layers.17.self_attn.o_proj.weight", [32, 32]),  # 2048B
            _make_record("language_model.model.layers.17.self_attn.q_a_proj.weight", [16, 16]),  # 512B
            _make_record("language_model.model.layers.17.self_attn.q_a_proj.weight_scale_inv", [1, 1], torch.float32),  # 256B aligned
        ]
        partitions = _partition_records(recs, bucket_size_bytes=2048)
        bmap = _bucket_index_map(partitions)
        self.assertEqual(
            bmap["language_model.model.layers.17.self_attn.kv_a_proj_with_mqa.weight"],
            bmap["language_model.model.layers.17.self_attn.q_a_proj.weight"],
        )
        self.assertEqual(
            bmap["language_model.model.layers.17.self_attn.kv_a_proj_with_mqa.weight"],
            bmap["language_model.model.layers.17.self_attn.kv_a_proj_with_mqa.weight_scale_inv"],
        )
        self.assertEqual(
            bmap["language_model.model.layers.17.self_attn.q_a_proj.weight"],
            bmap["language_model.model.layers.17.self_attn.q_a_proj.weight_scale_inv"],
        )

    def test_dots3_three_way_mla_gate_fusion(self):
        recs = [
            _make_record("model.layers.3.self_attn.g_proj.weight", [16, 16]),
            _make_record("model.layers.3.self_attn.k_filler.weight", [32, 32]),
            _make_record("model.layers.3.self_attn.kv_a_proj_with_mqa.weight", [16, 16]),
            _make_record("model.layers.3.self_attn.o_proj.weight", [32, 32]),
            _make_record("model.layers.3.self_attn.q_a_proj.weight", [16, 16]),
        ]
        partitions = _partition_records(recs, bucket_size_bytes=2048)
        bmap = _bucket_index_map(partitions)
        self.assertEqual(
            {
                bmap["model.layers.3.self_attn.g_proj.weight"],
                bmap["model.layers.3.self_attn.kv_a_proj_with_mqa.weight"],
                bmap["model.layers.3.self_attn.q_a_proj.weight"],
            },
            {bmap["model.layers.3.self_attn.g_proj.weight"]},
        )

    def test_dsa_indexer_wk_and_weights_proj_atomic(self):
        recs = [
            _make_record("model.layers.5.self_attn.indexer.k_norm.weight", [32, 32]),
            _make_record("model.layers.5.self_attn.indexer.weights_proj.weight", [16, 16]),
            _make_record("model.layers.5.self_attn.indexer.weights_proj.scales", [1, 1], torch.float32),
            _make_record("model.layers.5.self_attn.indexer.wq_b.weight", [32, 32]),
            _make_record("model.layers.5.self_attn.indexer.wk.weight", [16, 16]),
            _make_record("model.layers.5.self_attn.indexer.wk.weight_scale_inv", [1, 1], torch.float32),
        ]
        partitions = _partition_records(recs, bucket_size_bytes=2048)
        bmap = _bucket_index_map(partitions)
        b_wk = bmap["model.layers.5.self_attn.indexer.wk.weight"]
        self.assertEqual(b_wk, bmap["model.layers.5.self_attn.indexer.wk.weight_scale_inv"])
        self.assertEqual(b_wk, bmap["model.layers.5.self_attn.indexer.weights_proj.weight"])
        self.assertEqual(b_wk, bmap["model.layers.5.self_attn.indexer.weights_proj.scales"])

    def test_deepseek_v4_wqkv_compressor_and_wo_a_atomic(self):
        recs = [
            _make_record("layers.0.attn.compressor.wgate.weight", [16, 16]),
            _make_record("layers.0.attn.compressor.wkv.weight", [16, 16]),
            _make_record("layers.0.attn.wkv.weight", [16, 16]),
            _make_record("layers.0.attn.wkv.scale", [1, 1], torch.float32),
            _make_record("layers.0.attn.wo_a.weight", [32, 32]),
            _make_record("layers.0.attn.wo_a.scale", [1, 1], torch.float32),
            _make_record("layers.0.attn.wo_b.weight", [32, 32]),
            _make_record("layers.0.attn.wq_a.weight", [16, 16]),
            _make_record("layers.0.attn.wq_a.scale", [1, 1], torch.float32),
        ]
        partitions = _partition_records(recs, bucket_size_bytes=2560)
        bmap = _bucket_index_map(partitions)
        # Compressor wkv & wgate stay together
        self.assertEqual(
            bmap["layers.0.attn.compressor.wgate.weight"],
            bmap["layers.0.attn.compressor.wkv.weight"],
        )
        # Attn wq_a & wkv stay together (distinct from compressor)
        self.assertEqual(
            bmap["layers.0.attn.wq_a.weight"],
            bmap["layers.0.attn.wkv.weight"],
        )
        self.assertEqual(
            bmap["layers.0.attn.wq_a.weight"],
            bmap["layers.0.attn.wq_a.scale"],
        )
        self.assertEqual(
            bmap["layers.0.attn.wkv.weight"],
            bmap["layers.0.attn.wkv.scale"],
        )
        # wo_a weight & scale stay together for immediate dequant
        self.assertEqual(
            bmap["layers.0.attn.wo_a.weight"],
            bmap["layers.0.attn.wo_a.scale"],
        )
        # Ensure compressor.wkv did NOT collide with attn.wq_a
        self.assertNotEqual(
            _atomic_group_key("layers.0.attn.compressor.wkv.weight"),
            _atomic_group_key("layers.0.attn.wkv.weight"),
        )

    def test_compressed_tensors_int4_triplets_atomic(self):
        recs = [
            _make_record("language_model.model.layers.1.mlp.experts.147.down_proj.weight_packed", [32, 16], torch.int32),
            _make_record("language_model.model.layers.1.mlp.experts.147.gate_proj.weight_packed", [32, 16], torch.int32),
            _make_record("language_model.model.layers.1.mlp.experts.147.gate_proj.weight_scale", [32, 1], torch.bfloat16),
            _make_record("language_model.model.layers.1.mlp.experts.147.gate_proj.weight_shape", [2], torch.int32),
            _make_record("language_model.model.layers.1.mlp.experts.147.gate_proj.weight_zero_point", [32, 1], torch.int32),
        ]
        partitions = _partition_records(recs, bucket_size_bytes=2560)
        bmap = _bucket_index_map(partitions)
        b = bmap["language_model.model.layers.1.mlp.experts.147.gate_proj.weight_packed"]
        self.assertEqual(b, bmap["language_model.model.layers.1.mlp.experts.147.gate_proj.weight_scale"])
        self.assertEqual(b, bmap["language_model.model.layers.1.mlp.experts.147.gate_proj.weight_shape"])
        self.assertEqual(b, bmap["language_model.model.layers.1.mlp.experts.147.gate_proj.weight_zero_point"])

    def test_prefused_3d_moe_tensors_not_merged_or_bloated(self):
        # Qwen3.5 / Qwen3-VL store 3D MoE weights directly without .weight leaf.
        # They must NOT be merged into one oversized atomic group.
        recs = [
            _make_record("model.language_model.layers.0.mlp.experts.down_proj", [64, 64]),     # 8192B
            _make_record("model.language_model.layers.0.mlp.experts.gate_up_proj", [64, 64]),  # 8192B
        ]
        self.assertNotEqual(
            _atomic_group_key(recs[0].name),
            _atomic_group_key(recs[1].name),
        )
        partitions = _partition_records(recs, bucket_size_bytes=8192)
        self.assertEqual(len(partitions), 2)
        self.assertEqual(max(sz for _, sz in partitions), 8192)

    def test_end_to_end_safetensors_manifest_validation_and_ipc_clobber_simulation(self):
        # Build a realistic multi-shard safetensors checkpoint on disk, pack with
        # dump_to_mooncake, validate with mooncake_loader._validate_manifest_layout,
        # and simulate bucket-by-bucket IPC loading with destructive buffer clobber.
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.manual_seed(42)
            t_kv_a = torch.randn(8, 16, dtype=torch.bfloat16)
            t_q_a = torch.randn(12, 16, dtype=torch.bfloat16)
            t_kv_b = torch.randn(32, 32, dtype=torch.bfloat16)
            t_kv_b_scale = torch.randn(2, 2, dtype=torch.float32)
            t_o = torch.randn(32, 32, dtype=torch.bfloat16)
            t_wk = torch.randn(8, 16, dtype=torch.bfloat16)
            t_wproj = torch.randn(4, 16, dtype=torch.bfloat16)

            shard1 = {
                "language_model.model.layers.17.self_attn.kv_a_proj_with_mqa.weight": t_kv_a,
                "language_model.model.layers.17.self_attn.kv_b_proj.weight": t_kv_b,
                "language_model.model.layers.17.self_attn.kv_b_proj.weight_scale_inv": t_kv_b_scale,
                "language_model.model.layers.17.self_attn.indexer.wk.weight": t_wk,
            }
            shard2 = {
                "language_model.model.layers.17.self_attn.o_proj.weight": t_o,
                "language_model.model.layers.17.self_attn.q_a_proj.weight": t_q_a,
                "language_model.model.layers.17.self_attn.indexer.weights_proj.weight": t_wproj,
            }
            p1 = os.path.join(tmpdir, "model-00001-of-00002.safetensors")
            p2 = os.path.join(tmpdir, "model-00002-of-00002.safetensors")
            _write_safetensors_file(p1, shard1)
            _write_safetensors_file(p2, shard2)
            wmap = {k: os.path.basename(p1) for k in shard1}
            wmap.update({k: os.path.basename(p2) for k in shard2})
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": wmap}, f)

            files = _list_safetensors_files(tmpdir)
            records = _collect_tensor_records(files)
            partitions = _partition_records(records, bucket_size_bytes=2560)
            self.assertGreaterEqual(len(partitions), 3)

            # Build real bucket buffers and manifest
            manifest_buckets = []
            bucket_raw_bytes = []
            for idx, (recs, sz) in enumerate(partitions):
                buf = torch.empty(sz, dtype=torch.uint8)
                items = _fill_bucket_from_records(buf, recs)
                manifest_buckets.append({
                    "bucket_key": f"ckpt:test:bucket:{idx}",
                    "size": sz,
                    "items": items,
                })
                bucket_raw_bytes.append(buf.clone())

            manifest = {
                "checkpoint": "test",
                "align_size": ALIGN_SIZE,
                "buckets": manifest_buckets,
            }
            # Must pass mooncake_loader's strict layout validator unchanged!
            _validate_manifest_layout(manifest)

            # Simulate pristine per-bucket stateless/local-dict model.load_weights()
            # WITH destructive IPC slot buffer reuse (clobbering buffer after each bucket).
            max_sz = max(b["size"] for b in manifest_buckets)
            shared_ipc_slot = torch.empty(max_sz, dtype=torch.uint8)
            loaded_model_state: Dict[str, torch.Tensor] = {}

            def mock_pristine_load_weights(bucket_weights):
                # Local dicts destroyed at the end of EVERY bucket call (exact pristine behavior)
                cached_a_proj: Dict[str, torch.Tensor] = {}
                pending_indexer: Dict[str, torch.Tensor] = {}
                for name, view in bucket_weights:
                    lname = name.replace("language_model.", "")
                    if "q_a_proj" in lname or "kv_a_proj_with_mqa" in lname:
                        # No clone! Mirrors pristine handover/17b2d66 loader.
                        cached_a_proj[lname] = view
                        qn = lname if "q_a_proj" in lname else lname.replace("kv_a_proj_with_mqa", "q_a_proj")
                        kvn = lname if "kv_a_proj_with_mqa" in lname else lname.replace("q_a_proj", "kv_a_proj_with_mqa")
                        if qn in cached_a_proj and kvn in cached_a_proj:
                            fused = torch.cat([cached_a_proj[qn], cached_a_proj[kvn]], dim=0).clone()
                            loaded_model_state["fused_qkv_a_proj_with_mqa.weight"] = fused
                            cached_a_proj.pop(qn)
                            cached_a_proj.pop(kvn)
                    elif ".indexer.wk." in lname or ".indexer.weights_proj." in lname:
                        pending_indexer[lname] = view
                        wkn = lname if ".wk." in lname else lname.replace(".weights_proj.", ".wk.")
                        wpn = lname if ".weights_proj." in lname else lname.replace(".wk.", ".weights_proj.")
                        if wkn in pending_indexer and wpn in pending_indexer:
                            fused = torch.cat([pending_indexer[wkn], pending_indexer[wpn]], dim=0).clone()
                            loaded_model_state["indexer.wk_weights_proj.weight"] = fused
                            pending_indexer.pop(wkn)
                            pending_indexer.pop(wpn)
                    else:
                        loaded_model_state[lname] = view.clone()
                # Strict assertion: ZERO leftover half-paired tensors at end of any bucket!
                self.assertEqual(len(cached_a_proj), 0, f"Unpaired a_proj left over: {list(cached_a_proj)}")
                self.assertEqual(len(pending_indexer), 0, f"Unpaired indexer left over: {list(pending_indexer)}")

            dtype_lookup = {
                "torch.bfloat16": torch.bfloat16,
                "torch.float32": torch.float32,
            }
            for b_meta, raw_buf in zip(manifest_buckets, bucket_raw_bytes):
                shared_ipc_slot[: b_meta["size"]].copy_(raw_buf)
                views = []
                for item in b_meta["items"]:
                    dt = dtype_lookup[item["dtype"]]
                    sh = torch.Size(item["shape"])
                    nbytes = dt.itemsize * sh.numel()
                    v = shared_ipc_slot[item["offset"] : item["offset"] + nbytes].view(dt).view(sh)
                    views.append((item["name"], v))
                mock_pristine_load_weights(views)
                # Destructively overwrite shared IPC slot immediately after worker ACK!
                shared_ipc_slot.fill_(0xFF)

            # Verify bit-exact fused and standalone weights
            expected_qkv_a = torch.cat([t_q_a, t_kv_a], dim=0)
            expected_idx = torch.cat([t_wk, t_wproj], dim=0)
            self.assertTrue(torch.equal(loaded_model_state["fused_qkv_a_proj_with_mqa.weight"], expected_qkv_a))
            self.assertTrue(torch.equal(loaded_model_state["indexer.wk_weights_proj.weight"], expected_idx))
            self.assertTrue(torch.equal(loaded_model_state["model.layers.17.self_attn.kv_b_proj.weight"], t_kv_b))
            self.assertTrue(torch.equal(loaded_model_state["model.layers.17.self_attn.kv_b_proj.weight_scale_inv"], t_kv_b_scale))
            self.assertTrue(torch.equal(loaded_model_state["model.layers.17.self_attn.o_proj.weight"], t_o))

    def test_verify_atomic_partitions_fails_closed_on_split(self):
        r1 = _make_record("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", [16, 16])
        r2 = _make_record("model.layers.0.self_attn.q_a_proj.weight", [16, 16])
        bad_partitions = [([r1], r1.size_aligned), ([r2], r2.size_aligned)]
        with self.assertRaises(RuntimeError):
            _verify_atomic_partitions(bad_partitions, expected_count=2)


if __name__ == "__main__":
    unittest.main()
