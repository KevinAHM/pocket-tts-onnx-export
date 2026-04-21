import argparse
import os
import subprocess
import sys
from pathlib import Path

from pocket_tts.default_parameters import DEFAULT_LANGUAGE

OUTPUT_DIR = Path("pocket-tts-onnx") / "onnx"
SCRIPTS_DIR = Path("scripts")


def install_check():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("❌ huggingface_hub is missing. Please run: sfw pip install -r requirements.txt")
        sys.exit(1)


def _model_label(language: str | None, config: str | None) -> str:
    if language:
        return language
    if config:
        return Path(config).stem
    return DEFAULT_LANGUAGE


def run_export_scripts(language: str | None, config: str | None, output_dir: Path, exact: bool):
    print("\n--- Running Export Scripts ---")
    output_dir.mkdir(exist_ok=True, parents=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = "." + os.pathsep + env.get("PYTHONPATH", "")

    selector_args: list[str]
    if config is not None:
        selector_args = ["--config", config]
    else:
        selector_args = ["--language", language or DEFAULT_LANGUAGE]

    verify_args = ["--exact"] if exact else []

    cmd1 = [
        sys.executable,
        str(SCRIPTS_DIR / "export_mimi_and_conditioner.py"),
        "--output_dir",
        str(output_dir),
        *selector_args,
        *verify_args,
    ]
    print("\n[1/2] Exporting Mimi & Text Conditioner...")
    subprocess.run(cmd1, check=True, env=env)
    print("✅ Mimi/Conditioner Export Success")

    cmd2 = [
        sys.executable,
        str(SCRIPTS_DIR / "export_flow_lm.py"),
        "--output_dir",
        str(output_dir),
        *selector_args,
        *verify_args,
    ]
    print("\n[2/2] Exporting FlowLM...")
    subprocess.run(cmd2, check=True, env=env)
    print("✅ FlowLM Export Success")


def run_quantization(output_dir: Path):
    print("\n--- [Optional] Running Quantization ---")
    if not any(output_dir.glob("*.onnx")):
        print("⚠️ No models found in output directory to quantize.")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = "." + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "quantize.py"),
        "--input_dir",
        str(output_dir),
        "--output_dir",
        str(output_dir),
    ]

    subprocess.run(cmd, check=True, env=env)
    print(f"✅ Quantization Success! INT8 models in: {output_dir.absolute()}")


def print_summary(output_dir: Path):
    print(f"\n✅ All Done! Models are in: {output_dir.absolute()}")
    if not output_dir.exists():
        print("⚠️ Output directory missing.")
        return
    for path in sorted(output_dir.glob("*.onnx")):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f" - {path.name:<30} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified ONNX export for PocketTTS")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Model language/config name.")
    parser.add_argument("--config", default=None, help="Path to a local YAML config file.")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), help="Base output directory.")
    parser.add_argument("--quantize", action="store_true", help="Run INT8 quantization after export.")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Require exact torch vs ONNX equality during verification.",
    )
    args = parser.parse_args()

    install_check()
    label = _model_label(args.language, args.config)
    final_output_dir = Path(args.output_dir) / label
    run_export_scripts(args.language, args.config, final_output_dir, exact=args.exact)

    if args.quantize:
        run_quantization(final_output_dir)

    print_summary(final_output_dir)
