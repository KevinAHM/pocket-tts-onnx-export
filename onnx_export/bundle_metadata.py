import json
import shutil
from pathlib import Path

import numpy as np
import torch

from pocket_tts.default_parameters import MAX_TOKEN_PER_CHUNK
from pocket_tts.utils.utils import PREDEFINED_VOICES, download_if_necessary


def _tensor_fill_kind(tensor: torch.Tensor) -> str:
    if tensor.numel() == 0:
        return "empty"
    if tensor.dtype == torch.bool:
        if torch.all(tensor):
            return "ones"
        if not torch.any(tensor):
            return "zeros"
        return "mixed"
    if tensor.dtype.is_floating_point:
        if torch.isnan(tensor).all():
            return "nan"
        if torch.all(tensor == 0):
            return "zeros"
        return "mixed"
    if torch.all(tensor == 0):
        return "zeros"
    if torch.all(tensor == 1):
        return "ones"
    return "mixed"


def build_state_manifest(state: dict) -> list[dict]:
    manifest: list[dict] = []

    def walk(node: dict, prefix: str = ""):
        for key in sorted(node):
            value = node[key]
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                walk(value, path)
                continue
            module_name, tensor_key = path.rsplit("/", 1)
            manifest.append(
                {
                    "module": module_name,
                    "key": tensor_key,
                    "path": path,
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "shape": list(value.shape),
                    "fill": _tensor_fill_kind(value),
                }
            )

    walk(state)
    for index, entry in enumerate(manifest):
        entry["index"] = index
        entry["input_name"] = f"state_{index}"
        entry["output_name"] = f"out_state_{index}"
    return manifest


def _load_existing_metadata(output_dir: Path) -> dict:
    metadata_path = output_dir / "bundle.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    return {}


def _save_metadata(output_dir: Path, metadata: dict):
    metadata_path = output_dir / "bundle.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _copy_tokenizer(tts_model, output_dir: Path) -> str:
    tokenizer_src = download_if_necessary(tts_model.config.flow_lm.lookup_table.tokenizer_path)
    tokenizer_dest = output_dir / "tokenizer.model"
    if tokenizer_src.resolve() != tokenizer_dest.resolve():
        shutil.copy2(tokenizer_src, tokenizer_dest)
    return tokenizer_dest.name


def _save_bos_before_voice(tts_model, output_dir: Path) -> str | None:
    if not getattr(tts_model.flow_lm, "insert_bos_before_voice", False):
        return None
    if not hasattr(tts_model.flow_lm, "bos_before_voice"):
        return None
    dest = output_dir / "bos_before_voice.npy"
    np.save(dest, tts_model.flow_lm.bos_before_voice.detach().cpu().numpy())
    return dest.name


def write_bundle_metadata(
    output_dir: str | Path,
    tts_model,
    bundle_name: str,
    flow_state: dict | None = None,
    mimi_state: dict | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_existing_metadata(output_dir)
    metadata.update(
        {
            "schema_version": 2,
            "bundle_name": bundle_name,
            "language": tts_model.origin.stem if tts_model.origin is not None else bundle_name,
            "sample_rate": tts_model.config.mimi.sample_rate,
            "frame_rate": tts_model.config.mimi.frame_rate,
            "samples_per_frame": int(
                round(tts_model.config.mimi.sample_rate / tts_model.config.mimi.frame_rate)
            ),
            "latent_dim": tts_model.flow_lm.ldim,
            "conditioning_dim": tts_model.flow_lm.dim,
            "pad_with_spaces_for_short_inputs": tts_model.pad_with_spaces_for_short_inputs,
            "remove_semicolons": tts_model.remove_semicolons,
            "model_recommended_frames_after_eos": tts_model.model_recommended_frames_after_eos,
            "max_token_per_chunk": MAX_TOKEN_PER_CHUNK,
            "insert_bos_before_voice": bool(
                getattr(tts_model.flow_lm, "insert_bos_before_voice", False)
            ),
            "tokenizer_file": _copy_tokenizer(tts_model, output_dir),
            "bos_before_voice_file": _save_bos_before_voice(tts_model, output_dir),
            "predefined_voices": sorted(PREDEFINED_VOICES),
        }
    )

    if flow_state is not None:
        metadata["flow_lm_state_manifest"] = build_state_manifest(flow_state)
    if mimi_state is not None:
        metadata["mimi_state_manifest"] = build_state_manifest(mimi_state)

    _save_metadata(output_dir, metadata)
