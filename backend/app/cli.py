import argparse
import asyncio
import json
import os
import shutil
import time
import uuid
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"


def _load_env_defaults():
    if ENV_FILE.is_file():
        for raw in ENV_FILE.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                os.environ.setdefault(key, value)
    os.environ.setdefault("GEQNMR_DATA_ROOT", str(ROOT_DIR / "outputs"))
    os.environ.setdefault("GEQNMR_MODELS_DIR", str(ROOT_DIR / "models"))


_load_env_defaults()
warnings.filterwarnings("ignore", message="A NumPy version .* is required for this version of SciPy.*", category=UserWarning)

DATA_ROOT = Path(os.environ["GEQNMR_DATA_ROOT"])
MODELS_DIR = Path(os.environ["GEQNMR_MODELS_DIR"])
OUTPUT_DIR = DATA_ROOT
PREPARED_DIR = OUTPUT_DIR / "prepared_inputs"
RESULT_EXTENSIONS = {".pdb", ".xyz", ".zip"}
RESULT_METADATA_SUFFIX = ".meta.json"

for directory in (OUTPUT_DIR, MODELS_DIR, PREPARED_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    try:
        directory.chmod(0o755)
    except OSError:
        pass


def _print_json(payload):
    print(json.dumps(payload, indent=2, default=str))


def _copy_input(src: Path, dest_dir: Path) -> Path:
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    try:
        dest.chmod(0o644)
    except OSError:
        pass
    return dest


def _prediction_metadata_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.name}{RESULT_METADATA_SUFFIX}")


def _write_prepared_manifest(prepared_dir: Path, manifest: dict):
    manifest_path = prepared_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    try:
        prepared_dir.chmod(0o755)
        manifest_path.chmod(0o644)
    except OSError:
        pass


def _load_prepared_manifest(prepared_dir: Path) -> dict:
    return json.loads((prepared_dir / "manifest.json").read_text())


def _prepared_summary(prepared_dir: Path) -> dict:
    manifest = _load_prepared_manifest(prepared_dir)
    stat = prepared_dir.stat()
    return {
        "id": prepared_dir.name,
        "name": manifest.get("name") or manifest.get("input_file") or prepared_dir.name,
        "input_file": manifest.get("input_file"),
        "trajectory_file": manifest.get("trajectory_file"),
        "npz_file": manifest.get("npz_file"),
        "num_molecules": manifest.get("num_molecules", 0),
        "created": manifest.get("created", stat.st_ctime),
        "modified": stat.st_mtime,
    }


def _prepared_detail(prepared_id: str) -> dict:
    return _prepared_summary(_resolve_prepared_dir(prepared_id))


def _resolve_prepared_dir(prepared_id: str) -> Path:
    prepared_dir = PREPARED_DIR / Path(prepared_id).name
    if not prepared_dir.is_dir():
        raise SystemExit(f"Prepared input not found: {prepared_id}")
    return prepared_dir


def _list_available_models():
    return sorted(path for path in MODELS_DIR.rglob("*.pth") if path.is_file())


def _list_available_results():
    results = []
    for path in OUTPUT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in RESULT_EXTENSIONS:
            stat = path.stat()
            results.append({
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
                "has_predictions": _prediction_metadata_path(path).is_file(),
            })
    results.sort(key=lambda item: item["modified"], reverse=True)
    return results


def cmd_list_prepared(_args):
    prepared = []
    for path in sorted(PREPARED_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.is_dir() and (path / "manifest.json").is_file():
            try:
                prepared.append(_prepared_summary(path))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
    _print_json({"prepared": prepared})


def cmd_list_models(_args):
    models = [str(path.relative_to(MODELS_DIR)) for path in _list_available_models()]
    _print_json({"models": models, "default_model": models[0] if models else None})


def cmd_list_results(_args):
    _print_json({"results": _list_available_results()})


def cmd_delete_prepared(args):
    prepared_dir = _resolve_prepared_dir(args.prepared_id)
    shutil.rmtree(prepared_dir, ignore_errors=True)
    _print_json({"deleted": args.prepared_id})


def cmd_prepare(args):
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    trajectory_path = Path(args.trajectory).expanduser().resolve() if args.trajectory else None
    if trajectory_path and not trajectory_path.is_file():
        raise SystemExit(f"Trajectory file not found: {trajectory_path}")

    prepared_id = uuid.uuid4().hex
    prepared_dir = PREPARED_DIR / prepared_id
    prepared_dir.mkdir(parents=True, exist_ok=True)
    try:
        prepared_dir.chmod(0o755)
    except OSError:
        pass

    stored_input = _copy_input(input_path, prepared_dir)
    stored_trajectory = _copy_input(trajectory_path, prepared_dir) if trajectory_path else None

    def progress(value: float, message: str):
        print(f"{int(float(value) * 100):3d}% - {message}", flush=True)

    try:
        from . import processing

        npz_output_path, num_molecules = processing.process_uploaded_file(
            stored_input,
            prepared_dir,
            stored_trajectory,
            progress_callback=progress,
            num_workers=max(1, int(args.workers)),
        )
        manifest = {
            "id": prepared_id,
            "name": args.name or input_path.name,
            "input_file": stored_input.name,
            "trajectory_file": stored_trajectory.name if stored_trajectory else None,
            "npz_file": npz_output_path.name,
            "num_molecules": num_molecules,
            "num_workers": max(1, int(args.workers)),
            "created": time.time(),
        }
        _write_prepared_manifest(prepared_dir, manifest)
        _print_json({"prepared": _prepared_detail(prepared_id)})
    except Exception:
        shutil.rmtree(prepared_dir, ignore_errors=True)
        raise


async def _infer_prepared(args):
    from . import main

    prepared_dir = main._resolve_prepared_dir(args.prepared_id)
    manifest = main._load_prepared_manifest(prepared_dir)
    input_path = prepared_dir / manifest["input_file"]
    npz_path = prepared_dir / manifest["npz_file"]
    trajectory_path = prepared_dir / manifest["trajectory_file"] if manifest.get("trajectory_file") else None
    model_path = main.resolve_model_path(args.model) if args.model else main.get_default_model_path()
    output_path, atoms_predicted = await main.run_inference_workflow(
        input_path=input_path,
        output_dir=main.OUTPUT_DIR,
        destandardize=args.destandardize,
        model_path=model_path,
        trajectory_path=trajectory_path,
        prepared_npz_path=npz_path,
        frame_slice=args.frame_slice,
        prepared_context=main._prepared_summary(prepared_dir),
        device_name=args.device,
        batch_size=args.batch_size,
    )
    _print_json({
        "message": "Inference complete. Structure file with predictions generated.",
        "input_file": manifest["input_file"],
        "prepared_id": args.prepared_id,
        "model": str(model_path),
        "output_file": str(output_path),
        "atoms_predicted": atoms_predicted,
    })


def cmd_infer_prepared(args):
    asyncio.run(_infer_prepared(args))


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m backend.app.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-prepared").set_defaults(func=cmd_list_prepared)
    subparsers.add_parser("list-models").set_defaults(func=cmd_list_models)
    subparsers.add_parser("list-results").set_defaults(func=cmd_list_results)

    delete_parser = subparsers.add_parser("delete-prepared")
    delete_parser.add_argument("prepared_id")
    delete_parser.set_defaults(func=cmd_delete_prepared)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", required=True)
    prepare_parser.add_argument("--trajectory")
    prepare_parser.add_argument("--name")
    prepare_parser.add_argument("--workers", type=int, default=8)
    prepare_parser.set_defaults(func=cmd_prepare)

    infer_parser = subparsers.add_parser("infer-prepared")
    infer_parser.add_argument("prepared_id")
    infer_parser.add_argument("--model")
    infer_parser.add_argument("--destandardize", action=argparse.BooleanOptionalAction, default=True)
    infer_parser.add_argument("--frame-slice")
    infer_parser.add_argument("--device", default="cuda")
    infer_parser.add_argument("--batch-size", type=int, default=1)
    infer_parser.set_defaults(func=cmd_infer_prepared)

    return parser


def main_cli():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main_cli()
