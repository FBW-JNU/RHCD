#!/usr/bin/env python
"""Export portable main/RD weights and their recovery metric records."""

import argparse
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.config import CfgNode
from inference import RHCD, RHCDReceiver, RHCDRelay
from utils.checkpoint import FORMAT, load_model, sha256
from utils.hce_checkpoint import HCE_FORMAT
from utils.rd_checkpoint import RD_FORMAT, load_rd


def portable_config(config):
    result = copy.deepcopy(dict(config))
    replacements = {"data_root": "data/custom", "train_list": None, "val_list": None,
                    "weight": None, "resume": None, "save_path": "runs/rhcd"}
    for key, value in replacements.items():
        if key in result:
            result[key] = value
    result.pop("initialization_record", None)
    return result


def cpu_state(module):
    state = {key: value.detach().cpu().contiguous() for key, value in module.state_dict().items()}
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Model state contains nonfinite values")
    return state


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


@torch.inference_mode()
def export(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    source, config, payload = load_model(args.checkpoint)
    source.eval()
    source_hash = sha256(args.checkpoint)
    config = CfgNode(portable_config(config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + args.output.name + "_", dir=args.output.parent))
    try:
        weights = {"format": FORMAT, "architecture_version": source.architecture_version,
                   "config": dict(config), "epoch": int(payload["epoch"]),
                   "model_state": cpu_state(source),
                   "runtime_state": {"texture_bias": source.current_texture_bias, "secret_lambda": source.secret_lambda},
                   "optimizer": None, "scheduler": None, "rng_state": None, "rank_rng_states": None,
                   "best_metric": payload.get("best_metric"), "deployment_weights": True,
                   "provenance": {"source_checkpoint_sha256": source_hash}}
        main_path = temporary / "rhcd.pth"
        torch.save(weights, main_path)
        main_hash = sha256(main_path)
        hce_path = temporary / "hce.pth"
        torch.save({"format": HCE_FORMAT, "architecture_version": source.architecture_version,
                    "config": dict(config), "epoch": int(payload["epoch"]),
                    "hce_state": cpu_state(source.hce), "runtime_state": dict(weights["runtime_state"]),
                    "base_checkpoint_sha256": main_hash, "deployment_weights": True,
                    "provenance": {"source_checkpoint_sha256": source_hash}}, hce_path)
        metadata = {"format": "rhcd_bundle_v1", "architecture_version": source.architecture_version,
                    "main": {"file": main_path.name, "sha256": main_hash, "epoch": payload["epoch"],
                             "source_checkpoint_sha256": source_hash},
                    "geometry": source.sizes, "scale": source.scale, "rounds": source.rounds,
                    "runtime_state": dict(weights["runtime_state"]),
                    "parameters": sum(p.numel() for p in source.parameters()),
                    "representation": "RGB HWC float32 NPY", "config": dict(config), "metrics": []}
        metadata["hce"] = {"file": hce_path.name, "sha256": sha256(hce_path),
                           "base_checkpoint_sha256": main_hash,
                           "parameters": sum(p.numel() for p in source.hce.parameters())}
        source_rd, rd_payload, rd_path = None, None, None
        if args.rd_checkpoint is not None:
            if source.rounds != 3:
                raise ValueError("Learned RD export requires a three-tier model")
            source_rd, rd_payload = load_rd(args.rd_checkpoint, args.checkpoint)
            source_rd.eval()
            rd_path = temporary / "rd.pth"
            torch.save({"format": RD_FORMAT, "protocol": rd_payload["protocol"],
                        "native_y2_supported": bool(rd_payload.get("native_y2_supported", False)),
                        "architecture": rd_payload["architecture"], "state_dict": cpu_state(source_rd),
                        "config": portable_config(rd_payload["config"]), "epoch": rd_payload["epoch"],
                        "base_checkpoint": main_path.name, "base_checkpoint_sha256": main_hash,
                        "optimizer": None, "scheduler": None, "rng_state": None,
                        "best_metric": rd_payload.get("best_metric"), "deployment_weights": True,
                        "provenance": {"source_checkpoint_sha256": sha256(args.rd_checkpoint),
                                       "source_base_checkpoint_sha256": source_hash}}, rd_path)
            metadata["rd"] = {"file": rd_path.name, "sha256": sha256(rd_path),
                              "epoch": rd_payload["epoch"], "protocol": rd_payload["protocol"],
                              "native_y2_supported": bool(rd_payload.get("native_y2_supported", False)),
                              "parameters": sum(p.numel() for p in source_rd.parameters())}
        restored = RHCD(main_path, rd_checkpoint=rd_path)
        receiver = RHCDReceiver(hce_path, rd_checkpoint=rd_path)
        generator = torch.Generator().manual_seed(131)
        base = torch.rand(1, 3, source.sizes[0], source.sizes[0], generator=generator)
        secrets = [torch.rand(1, 3, size, size, generator=generator) for size in source.sizes[:-1]]
        expected = source(base, secrets)
        actual = restored.embed(base, secrets)
        checks = {"embedding_exact": all(torch.equal(x, y) for x, y in zip(expected, actual))}
        checks["native_extraction_exact"] = all(torch.equal(source.decode(x), restored.model.decode(y))
                                                for x, y in zip(expected, actual))
        checks["hce_only_native_exact"] = all(torch.equal(restored.model.decode(carrier),
                                                          receiver.extract(carrier, level, [level])[level])
                                              for level, carrier in enumerate(actual, 1))
        checks["bicubic_extraction_exact"] = all(
            torch.equal(source.decode(source.downgrade(expected[i - 1], i, j)),
                        restored.extract(actual[i - 1], i, [j], "bicubic")[j])
            for i in range(2, source.rounds + 1) for j in range(1, i))
        checks["hce_only_bicubic_exact"] = all(
            torch.equal(restored.extract(actual[i - 1], i, [j], "bicubic")[j],
                        receiver.extract(actual[i - 1], i, [j], "bicubic")[j])
            for i in range(2, source.rounds + 1) for j in range(1, i))
        if source_rd is not None:
            source_outputs = source_rd(expected[2], source.scale)
            restored_outputs = restored.rd(actual[2], source.scale)
            checks["rd_paths_exact"] = all(torch.equal(source_outputs[key], restored_outputs[key])
                                           for key in source_outputs)
            relay = RHCDRelay(rd_path)
            checks["hce_only_learned_exact"] = all(
                torch.equal(restored.extract(actual[2], 3, [level], "learned", route)[level],
                            receiver.extract(actual[2], 3, [level], "learned", route)[level])
                for route in ("direct", "stepwise") for level in (1, 2))
            checks["relay_paths_exact"] = all(
                torch.equal(relay.downgrade(actual[2], 3, route=route)[level],
                            restored.representations(actual[2], 3, route=route)[level])
                for route in ("direct", "stepwise") for level in (1, 2))
            if rd_payload.get("native_y2_supported", False):
                expected_native = source.decode(source_rd(expected[1], source.scale, source_level=2)["pred_1"])
                checks["native_y2_rd_exact"] = torch.equal(expected_native,
                                                           restored.extract(actual[1], 2, [1], "learned")[1])
                checks["hce_only_native_y2_rd_exact"] = torch.equal(expected_native,
                    receiver.extract(actual[1], 2, [1], "learned")[1])
                checks["relay_native_y2_exact"] = torch.equal(
                    source_rd(expected[1], source.scale, source_level=2)["pred_1"],
                    relay.downgrade(actual[1], 2)[1])
        if not all(checks.values()):
            raise ValueError("Exported model output differs: " + repr(checks))
        for index, path in enumerate(args.metrics):
            record = json.loads(path.read_text())
            if record.get("status") != "complete" or record.get("checkpoint_sha256") != source_hash:
                raise ValueError("Metric record does not match the complete source-model evaluation")
            if record.get("pair_count", 0) < 1 or not record.get("means"):
                raise ValueError("Metric record has no evaluated pairs")
            if record.get("rd_checkpoint_sha256"):
                if args.rd_checkpoint is None or record["rd_checkpoint_sha256"] != sha256(args.rd_checkpoint):
                    raise ValueError("Metric RD checkpoint differs from exported RD")
            metric = {key: record[key] for key in ("dataset_label", "pair_count", "manifest_sha256",
                                                  "means", "channel", "metrics", "precision", "rd_input",
                                                  "stepwise_protocol", "learned_rd_protocol", "native_y2_to_y1_operator")
                      if key in record}
            metric["checkpoint_sha256"] = main_hash
            metric["evaluated_source_checkpoint_sha256"] = source_hash
            if record.get("rd_checkpoint_sha256"):
                metric["rd_checkpoint_sha256"] = metadata["rd"]["sha256"]
                metric["evaluated_source_rd_checkpoint_sha256"] = record["rd_checkpoint_sha256"]
            name = f"metrics_{index + 1}.json"
            write_json(temporary / name, metric)
            metadata["metrics"].append({"file": name, "sha256": sha256(temporary / name),
                                        "dataset_label": record.get("dataset_label"),
                                        "pair_count": record["pair_count"]})
        metadata["verification"] = checks
        write_json(temporary / "model.json", metadata)
        (temporary / "config.json").write_text(json.dumps(dict(config), indent=2) + "\n")
        notes = ["# RHCD weights", "", f"Architecture: `{source.architecture_version}`.",
                 f"Geometry: `{' -> '.join(map(str, source.sizes))}`.", "",
                 "Run these commands from the RHCD source directory:", "", "```bash",
                 "python test.py embed --checkpoint /path/to/weights/rhcd.pth \\",
                 "  --cover /path/cover.png --secrets " + " ".join(f"/path/s{i}.png" for i in range(1, source.rounds + 1)) + " \\",
                 "  --output outputs/embed", "```", "",
                 "The receiver uses `hce.pth`. When RD is exported, a relay uses `rd.pth` for resolution downgrade.",
                 "`model.json` records weight checksums, geometry, configuration and metric files.",
                 "Carrier files use RGB HWC float32 NPY. Extraction follows the checkpoint configuration.", ""]
        (temporary / "README.md").write_text("\n".join(notes))
        temporary.rename(args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(args.output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--rd-checkpoint", type=Path)
    parser.add_argument("--metrics", nargs="*", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
