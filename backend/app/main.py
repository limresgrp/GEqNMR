from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import zipfile
import shutil
import tempfile
import json
import os
import time
import uuid
from pathlib import Path
import traceback
import yaml
import torch
import numpy as np
from typing import Optional, List

# Import core components and utility functions
from . import processing

# --- geqtrain Imports ---
from geqtrain.utils import Config
from geqtrain.data.dataloader import DataLoader
from geqtrain.train.components.dataset_builder import DatasetBuilder
from geqtrain.data import AtomicDataDict # Assuming this holds keys like 'pos', 'node_types' etc.
from geqtrain.train.components.inference import run_inference as geq_run_inference
from geqtrain.utils.deploy import load_deployed_model
from geqtrain.utils.inference_metadata import INFERENCE_METADATA_KEY, load_inference_metadata_bundle
from geqtrain.utils._global_options import apply_global_config

# --- Configuration ---
DATA_ROOT = Path(os.environ.get("GEQNMR_DATA_ROOT", "/workspaces/GEqNMR/outputs"))
MODELS_DIR = Path(os.environ.get("GEQNMR_MODELS_DIR", "/workspaces/GEqNMR/models"))
TEMPLATE_CONFIG = Path(__file__).parent / "template.yaml"

# --- existing app definition ---
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define output directories ---
OUTPUT_DIR = DATA_ROOT
RESULT_EXTENSIONS = {".pdb", ".xyz", ".zip"}
RESULT_METADATA_SUFFIX = ".meta.json"
PREPARED_DIR = OUTPUT_DIR / "prepared_inputs"
PREPARE_JOBS = {}

# Ensure directories exist on startup
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREPARED_DIR.mkdir(parents=True, exist_ok=True)
for directory in (OUTPUT_DIR, MODELS_DIR, PREPARED_DIR):
    try:
        directory.chmod(0o755)
    except OSError:
        pass

_SUMMARY_SAMPLE_SIZE = 20
_SUMMARY_HIST_BINS = 20
_SUMMARY_HIST_SAMPLE_SIZE = 10000


def list_available_models():
    return sorted(path for path in MODELS_DIR.rglob("*.pth") if path.is_file())


def list_available_results():
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


def _prediction_metadata_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.name}{RESULT_METADATA_SUFFIX}")


def _load_atom_labels_from_npz(npz_path: Path) -> List[str]:
    with np.load(npz_path, allow_pickle=True) as data:
        if "atom_labels" not in data.files:
            return []
        labels = data["atom_labels"]
        if labels.ndim == 0:
            return [str(labels.item())]
        if labels.ndim >= 2:
            labels = labels[0]
        return [str(label) for label in labels.reshape(-1).tolist() if str(label)]


def _write_prediction_metadata(
    result_path: Path,
    input_path: Path,
    model_path: Path,
    npz_path: Path,
    predictions: np.ndarray,
    destandardize: bool,
    frame_slice: Optional[str] = None,
    frame_indices: Optional[List[int]] = None,
    prepared: Optional[dict] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    normalization_source: Optional[str] = None,
):
    atom_labels = _load_atom_labels_from_npz(npz_path)
    if not atom_labels:
        return
    n_atoms = len(atom_labels)
    flat_predictions = predictions.reshape(-1).astype(np.float64)
    if flat_predictions.size % n_atoms != 0:
        print(
            f"Skipping prediction metadata for {result_path.name}: "
            f"{flat_predictions.size} predictions are not divisible by {n_atoms} atoms."
        )
        return
    n_frames = int(flat_predictions.size // n_atoms)
    values_by_frame = flat_predictions.reshape(n_frames, n_atoms)
    atoms = [
        {
            "id": atom_label,
            "label": atom_label,
            "index": atom_index,
            "values": values_by_frame[:, atom_index].tolist(),
        }
        for atom_index, atom_label in enumerate(atom_labels)
    ]
    payload = {
        "result": result_path.name,
        "input_file": input_path.name,
        "model": str(model_path),
        "destandardized": bool(destandardize),
        "frame_slice": frame_slice,
        "frame_indices": frame_indices,
        "prepared": prepared,
        "device": device,
        "batch_size": batch_size,
        "normalization_source": normalization_source,
        "num_frames": n_frames,
        "num_atoms": n_atoms,
        "atoms": atoms,
    }
    metadata_path = _prediction_metadata_path(result_path)
    metadata_path.write_text(json.dumps(payload, indent=2))
    try:
        metadata_path.chmod(0o644)
    except OSError:
        pass


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return float(value)


def _summarize_array(name: str, array: np.ndarray) -> dict:
    summary = {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }

    if array.size == 0:
        return summary

    if name == "Lattice":
        lattice = None
        if array.shape == (3, 3):
            lattice = array
        elif array.size == 9:
            lattice = array.reshape(3, 3)
        if lattice is not None:
            summary["matrix"] = lattice.astype(np.float64).tolist()

    kind = array.dtype.kind
    flat = array.reshape(-1)

    if kind in {"i", "u"}:
        values = flat.astype(np.int64, copy=False)
        summary["sample"] = values[:_SUMMARY_SAMPLE_SIZE].tolist()
        summary["stats"] = {
            "min": int(np.min(values)),
            "max": int(np.max(values)),
            "mean": _safe_float(np.mean(values)),
            "std": _safe_float(np.std(values)),
        }

        hist_values = values
        if hist_values.size > _SUMMARY_HIST_SAMPLE_SIZE:
            rng = np.random.default_rng(42)
            hist_values = rng.choice(hist_values, size=_SUMMARY_HIST_SAMPLE_SIZE, replace=False)

        min_val = int(np.min(hist_values))
        max_val = int(np.max(hist_values))
        range_size = max_val - min_val + 1
        if range_size <= _SUMMARY_HIST_BINS:
            edges = np.arange(min_val, max_val + 2, dtype=np.int64)
        else:
            step = int(np.ceil(range_size / _SUMMARY_HIST_BINS))
            edges = np.arange(min_val, max_val + step + 1, step, dtype=np.int64)
        counts, edges = np.histogram(hist_values, bins=edges)
        summary["histogram"] = {
            "bins": edges.astype(int).tolist(),
            "counts": counts.astype(int).tolist(),
        }
    elif kind == "f":
        values = flat.astype(np.float64, copy=False)
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        if finite_values.size == 0:
            return summary

        sample_values = values[:_SUMMARY_SAMPLE_SIZE]
        summary["sample"] = [_safe_float(val) for val in sample_values.tolist()]
        summary["stats"] = {
            "min": _safe_float(np.min(finite_values)),
            "max": _safe_float(np.max(finite_values)),
            "mean": _safe_float(np.mean(finite_values)),
            "std": _safe_float(np.std(finite_values)),
        }

        hist_values = finite_values
        if hist_values.size > _SUMMARY_HIST_SAMPLE_SIZE:
            rng = np.random.default_rng(42)
            hist_values = rng.choice(hist_values, size=_SUMMARY_HIST_SAMPLE_SIZE, replace=False)
        counts, edges = np.histogram(hist_values, bins=_SUMMARY_HIST_BINS)
        summary["histogram"] = {
            "bins": [_safe_float(val) for val in edges.tolist()],
            "counts": counts.astype(int).tolist(),
        }
    elif kind == "b":
        true_count = int(np.count_nonzero(flat))
        false_count = int(flat.size - true_count)
        summary["sample"] = flat[:_SUMMARY_SAMPLE_SIZE].astype(bool).tolist()
        summary["counts"] = {"true": true_count, "false": false_count}
        summary["histogram"] = {
            "bins": [0, 1],
            "counts": [false_count, true_count],
        }
    else:
        sample_values = [str(val) for val in flat[:_SUMMARY_SAMPLE_SIZE]]
        summary["sample"] = sample_values

    return summary


def _summarize_array_with_batch(name: str, array: np.ndarray, batch_index: Optional[int], num_molecules: int) -> dict:
    if batch_index is None:
        return _summarize_array(name, array)
    if array.ndim == 0 or array.shape[0] != num_molecules:
        raise HTTPException(status_code=400, detail="Batch selection is not available for this key.")
    if batch_index < 0 or batch_index >= num_molecules:
        raise HTTPException(status_code=400, detail="Batch index out of range.")
    slice_array = array[batch_index]
    summary = _summarize_array(name, slice_array)
    summary["batch_index"] = batch_index
    return summary


def summarize_npz(npz_path: Path) -> list:
    summary = []
    with np.load(npz_path, allow_pickle=True) as data:
        for key in sorted(data.files):
            summary.append(_summarize_array(key, data[key]))
    return summary


def resolve_model_path(model_name: str) -> Path:
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required.")
    candidate = Path(model_name)
    if not candidate.is_absolute():
        candidate = MODELS_DIR / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(MODELS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Model path must be within the models directory.")
    if candidate.suffix.lower() != ".pth" or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Model not found: {candidate}")
    return candidate


def get_default_model_path() -> Path:
    model_paths = list_available_models()
    if not model_paths:
        raise HTTPException(
            status_code=500,
            detail=f"No .pth models found in {MODELS_DIR}. Add a model to run inference.",
        )
    return model_paths[0]


def _disable_dataset_target_requirements(config: Config):
    """Keep output normalization metadata, but do not require labels in uploaded inference inputs."""
    config["loss_coeffs"] = []
    normalization = config.get("normalization")
    if not isinstance(normalization, dict):
        return
    for field, spec in list(normalization.items()):
        if isinstance(spec, dict):
            updated_spec = dict(spec)
            updated_spec["apply_on_dataset"] = False
            normalization[field] = updated_spec
        elif isinstance(spec, str):
            normalization[field] = {
                "mode": spec,
                "apply_on_dataset": False,
            }


def _make_inference_dataset_config(config: Config) -> Config:
    dataset_config = Config(config.as_dict())
    dataset_config["loss_coeffs"] = []
    dataset_config["normalization"] = {}
    for key in ("train_dataset_list", "validation_dataset_list", "dataset_list"):
        if key in dataset_config:
            dataset_config.pop(key)
    return dataset_config


@app.get("/")
def read_root():
    """Root endpoint to check if the backend is alive."""
    return {"message": "Hello from the FastAPI Backend! (Connection Successful)"}


@app.get("/models")
def list_models():
    model_paths = list_available_models()
    model_names = [str(path.relative_to(MODELS_DIR)) for path in model_paths]
    default_model = model_names[0] if model_names else None
    return {"models": model_names, "default_model": default_model}


@app.get("/results")
def list_results():
    return {"results": list_available_results()}


@app.get("/prediction-results")
def list_prediction_results():
    return {"results": [result for result in list_available_results() if result.get("has_predictions")]}


@app.get("/prediction-results/{filename:path}")
def get_prediction_result(filename: str):
    result_path = resolve_result_path(filename)
    metadata_path = _prediction_metadata_path(result_path)
    if not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="Prediction metadata not found for this result.")
    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid prediction metadata: {e}")


def resolve_result_path(filename: str) -> Path:
    candidate = Path(filename)
    if candidate.is_absolute():
        candidate = candidate.name
    candidate = (OUTPUT_DIR / candidate).resolve()
    try:
        candidate.relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid result path.")
    if candidate.suffix.lower() not in RESULT_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported result file type.")
    return candidate


@app.delete("/results/{filename:path}")
def delete_result(filename: str):
    file_path = resolve_result_path(filename)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        file_path.unlink()
        _prediction_metadata_path(file_path).unlink(missing_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")
    return {"deleted": file_path.name}


def _write_prepared_manifest(prepared_dir: Path, manifest: dict):
    manifest_path = prepared_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    try:
        prepared_dir.chmod(0o755)
        manifest_path.chmod(0o644)
    except OSError:
        pass


def _load_prepared_manifest(prepared_dir: Path) -> dict:
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Prepared input manifest not found.")
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid prepared input manifest: {e}")


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


def _resolve_prepared_dir(prepared_id: str) -> Path:
    prepared_dir = PREPARED_DIR / Path(prepared_id).name
    if not prepared_dir.is_dir():
        raise HTTPException(status_code=404, detail="Prepared input not found.")
    return prepared_dir


def _prepared_detail(prepared_id: str) -> dict:
    prepared_dir = _resolve_prepared_dir(prepared_id)
    manifest = _load_prepared_manifest(prepared_dir)
    npz_path = prepared_dir / manifest["npz_file"]
    if not npz_path.is_file():
        raise HTTPException(status_code=404, detail="Prepared dataset not found.")
    return {
        **_prepared_summary(prepared_dir),
        "keys": summarize_npz(npz_path),
    }


def _set_prepare_job(job_id: str, status: str, progress: float, message: str, result: Optional[dict] = None, error: Optional[str] = None):
    PREPARE_JOBS[job_id] = {
        **PREPARE_JOBS.get(job_id, {}),
        "id": job_id,
        "status": status,
        "progress": max(0.0, min(1.0, float(progress))),
        "message": message,
        "result": result,
        "error": error,
    }


def _parse_frame_slice(frame_slice: Optional[str], num_frames: int) -> Optional[List[int]]:
    if frame_slice is None or str(frame_slice).strip() == "":
        return None
    raw = str(frame_slice).strip()
    parts = raw.split(":")
    if len(parts) > 3:
        raise HTTPException(status_code=400, detail="Frame slice must use start:stop:step syntax.")
    values = []
    for part in parts:
        values.append(None if part == "" else int(part))
    while len(values) < 3:
        values.append(None)
    start, stop, step = values
    if step == 0:
        raise HTTPException(status_code=400, detail="Frame slice step cannot be zero.")
    indices = list(range(num_frames))[slice(start, stop, step)]
    if not indices:
        raise HTTPException(status_code=400, detail="Frame slice selected no frames.")
    return indices


def _slice_prepared_npz(source_npz: Path, output_dir: Path, frame_indices: Optional[List[int]], num_frames: int) -> Path:
    if frame_indices is None:
        return source_npz
    sliced_npz = output_dir / f"{source_npz.stem}_frames_{uuid.uuid4().hex[:8]}.npz"
    with np.load(source_npz, allow_pickle=True) as data:
        save_dict = {}
        for key in data.files:
            value = data[key]
            if value.ndim > 0 and value.shape[0] == num_frames:
                save_dict[key] = value[frame_indices]
            else:
                save_dict[key] = value
    np.savez(sliced_npz, **save_dict)
    try:
        sliced_npz.chmod(0o644)
    except OSError:
        pass
    return sliced_npz


def _progress_iter(iterable, total: int, description: str):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=description, unit="batch")
    except Exception:
        class _SimpleProgress:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __iter__(self):
                started = time.time()
                for idx, item in enumerate(self._wrapped, 1):
                    print(f"{description}: {idx}/{total} batch(es) elapsed={time.time() - started:.1f}s", flush=True)
                    yield item

            def close(self):
                return None

        return _SimpleProgress(iterable)


def _process_prepare_job(job_id: str, prepared_id: str, prepared_dir: Path, input_path: Path, input_filename: str, trajectory_input_path: Optional[Path], trajectory_filename: Optional[str], custom_name: Optional[str], num_workers: int):
    try:
        def progress(value: float, message: str):
            _set_prepare_job(job_id, "running", value, message)

        _set_prepare_job(job_id, "running", 0.01, "Saved uploaded files")
        npz_output_path, num_molecules = processing.process_uploaded_file(
            input_path,
            prepared_dir,
            trajectory_input_path,
            progress_callback=progress,
            num_workers=num_workers,
        )
        manifest = {
            "id": prepared_id,
            "name": custom_name or input_filename,
            "input_file": input_filename,
            "trajectory_file": trajectory_filename,
            "npz_file": npz_output_path.name,
            "num_molecules": num_molecules,
            "num_workers": num_workers,
            "created": time.time(),
        }
        _write_prepared_manifest(prepared_dir, manifest)
        result = {
            **_prepared_detail(prepared_id),
        }
        _set_prepare_job(job_id, "completed", 1.0, "Prepared input is ready", result=result)
    except Exception as e:
        traceback.print_exc()
        shutil.rmtree(prepared_dir, ignore_errors=True)
        _set_prepare_job(job_id, "failed", 1.0, f"Failed to prepare input: {e}", error=str(e))


@app.post("/prepare")
async def prepare_input(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    trajectory_file: UploadFile = File(None),
    name: str = Form(None),
    num_workers: int = Form(1),
):
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ["pdb", "gro", "xyz"]:
        raise HTTPException(status_code=400, detail="Only .pdb, .gro, or .xyz files are supported.")

    prepared_id = uuid.uuid4().hex
    job_id = uuid.uuid4().hex
    try:
        num_workers = max(1, int(num_workers))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="num_workers must be a positive integer.")
    prepared_dir = PREPARED_DIR / prepared_id
    prepared_dir.mkdir(parents=True, exist_ok=True)
    try:
        prepared_dir.chmod(0o755)
    except OSError:
        pass

    input_filename = Path(file.filename).name
    input_path = prepared_dir / input_filename
    trajectory_input_path = None
    trajectory_filename = None

    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        input_path.chmod(0o644)
    except Exception as e:
        shutil.rmtree(prepared_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")
    finally:
        file.file.close()

    if trajectory_file:
        trajectory_filename = Path(trajectory_file.filename).name
        trajectory_input_path = prepared_dir / trajectory_filename
        try:
            with trajectory_input_path.open("wb") as buffer:
                shutil.copyfileobj(trajectory_file.file, buffer)
            trajectory_input_path.chmod(0o644)
        except Exception as e:
            shutil.rmtree(prepared_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"Could not save uploaded trajectory file: {e}")
        finally:
            trajectory_file.file.close()

    _set_prepare_job(job_id, "queued", 0.0, "Queued input preparation")
    background_tasks.add_task(
        _process_prepare_job,
        job_id,
        prepared_id,
        prepared_dir,
        input_path,
        input_filename,
        trajectory_input_path,
        trajectory_filename,
        name.strip() if isinstance(name, str) and name.strip() else None,
        num_workers,
    )
    return {"job_id": job_id, "prepared_id": prepared_id, "status": "queued"}


@app.get("/prepare/jobs/{job_id}")
def get_prepare_job(job_id: str):
    job = PREPARE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Prepare job not found.")
    return job


@app.get("/prepared")
def list_prepared_inputs():
    prepared = []
    for prepared_dir in sorted(PREPARED_DIR.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        if prepared_dir.is_dir() and (prepared_dir / "manifest.json").is_file():
            try:
                prepared.append(_prepared_summary(prepared_dir))
            except HTTPException:
                continue
    return {"prepared": prepared}


@app.get("/prepared/{prepared_id}")
def get_prepared_input(prepared_id: str):
    return _prepared_detail(prepared_id)


@app.delete("/prepared/{prepared_id}")
def delete_prepared_input(prepared_id: str):
    prepared_dir = _resolve_prepared_dir(prepared_id)
    shutil.rmtree(prepared_dir, ignore_errors=True)
    return {"deleted": prepared_id}


@app.get("/prepare/{prepared_id}/keys/{key_name}")
def get_prepared_key(
    prepared_id: str,
    key_name: str,
    batch_index: Optional[int] = None,
):
    prepared_dir = _resolve_prepared_dir(prepared_id)

    manifest = _load_prepared_manifest(prepared_dir)
    npz_path = prepared_dir / manifest["npz_file"]
    if not npz_path.is_file():
        raise HTTPException(status_code=404, detail="Prepared dataset not found.")

    with np.load(npz_path, allow_pickle=True) as data:
        if key_name not in data.files:
            raise HTTPException(status_code=404, detail="Key not found in prepared dataset.")
        summary = _summarize_array_with_batch(
            key_name,
            data[key_name],
            batch_index,
            manifest.get("num_molecules", 0),
        )
    return {"key": summary}


@app.post("/prepare/{prepared_id}/lattice")
def set_prepared_lattice(
    prepared_id: str,
    payload: dict = Body(...),
):
    prepared_dir = _resolve_prepared_dir(prepared_id)

    manifest = _load_prepared_manifest(prepared_dir)
    npz_path = prepared_dir / manifest["npz_file"]
    if not npz_path.is_file():
        raise HTTPException(status_code=404, detail="Prepared dataset not found.")

    matrix = payload.get("matrix")
    if matrix is None:
        raise HTTPException(status_code=400, detail="Missing lattice matrix.")

    flat_values: List[float] = []
    if isinstance(matrix, list) and len(matrix) == 3 and all(isinstance(row, list) for row in matrix):
        for row in matrix:
            if len(row) != 3:
                raise HTTPException(status_code=400, detail="Lattice matrix must be 3x3.")
            for value in row:
                flat_values.append(float(value))
    elif isinstance(matrix, list) and len(matrix) == 9:
        flat_values = [float(value) for value in matrix]
    else:
        raise HTTPException(status_code=400, detail="Lattice matrix must be a 3x3 list or flat list of 9 values.")

    lattice_matrix = np.array(flat_values, dtype=np.float32).reshape(3, 3)
    num_molecules = int(manifest.get("num_molecules", 0))
    if num_molecules <= 0:
        raise HTTPException(status_code=400, detail="Invalid number of molecules for prepared input.")

    with np.load(npz_path, allow_pickle=True) as data:
        save_dict: dict = {key: data[key] for key in data.files}

    save_dict["Lattice"] = np.tile(lattice_matrix, (num_molecules, 1, 1))
    np.savez(npz_path, **save_dict)

    manifest["lattice_override"] = True
    _write_prepared_manifest(prepared_dir, manifest)

    return {
        "message": "Lattice matrix applied.",
        "keys": summarize_npz(npz_path),
    }


@app.post("/infer/prepared/{prepared_id}")
async def infer_prepared_input(
    prepared_id: str,
    model_name: str = Form(None),
    destandardize: bool = Form(True),
    frame_slice: str = Form(None),
    device: str = Form(None),
    batch_size: int = Form(1),
):
    prepared_dir = _resolve_prepared_dir(prepared_id)

    manifest = _load_prepared_manifest(prepared_dir)
    input_path = prepared_dir / manifest["input_file"]
    npz_path = prepared_dir / manifest["npz_file"]
    trajectory_input_path = None
    if manifest.get("trajectory_file"):
        trajectory_input_path = prepared_dir / manifest["trajectory_file"]

    if not input_path.is_file() or not npz_path.is_file():
        raise HTTPException(status_code=404, detail="Prepared input files are missing.")

    try:
        model_path = resolve_model_path(model_name) if model_name else get_default_model_path()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve model path: {e}")

    try:
        output_path, num_atoms_predicted = await run_inference_workflow(
            input_path=input_path,
            output_dir=OUTPUT_DIR,
            destandardize=destandardize,
            model_path=model_path,
            trajectory_path=trajectory_input_path,
            prepared_npz_path=npz_path,
            frame_slice=frame_slice,
            prepared_context=_prepared_summary(prepared_dir),
            device_name=device,
            batch_size=batch_size,
        )
        return {
            "message": "Inference complete. Structure file with predictions generated.",
            "input_file": manifest["input_file"],
            "prepared_id": prepared_id,
            "model": str(model_path),
            "output_file": str(output_path),
            "atoms_predicted": num_atoms_predicted,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Internal Server Error during inference: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred during inference: {e}")


# --- UTILITY: Simplified Inference Workflow ---
async def run_inference_workflow(
    input_path: Path,
    output_dir: Path,
    destandardize: bool,
    model_path: Path,
    trajectory_path: Path = None,
    prepared_npz_path: Path = None,
    frame_slice: Optional[str] = None,
    prepared_context: Optional[dict] = None,
    device_name: Optional[str] = None,
    batch_size: int = 1,
):
    """
    Orchestrates loading, inference, and file saving for an uploaded structure file.
    It saves PDB/GRO input as PDB (with B-factors) and XYZ input as Extended XYZ.
    """
    print(f"--- Starting Inference Workflow (De-standardize: {destandardize}) ---")
    
    try:
        batch_size = max(1, int(batch_size))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="batch_size must be a positive integer.")

    # Check if a CUDA device is available and use it unless explicitly overridden.
    if device_name and str(device_name).strip():
        try:
            device = torch.device(str(device_name).strip())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid device '{device_name}': {e}")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise HTTPException(status_code=400, detail="CUDA was requested but is not available.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load deployed model and metadata
    try:
        extra_metadata_keys = {key: "" for key in processing.STANDARDIZATION_METADATA_KEYS}
        model, metadata = load_deployed_model(
            model_path,
            device=device,
            extra_metadata=extra_metadata_keys,
        )
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Use an internal error code (500) since this is a setup issue
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model checkpoint. Ensure model path is correct: {model_path}. Error: {e}",
        )

    metadata_stats = processing.extract_standardization_stats(metadata)
    inference_metadata = load_inference_metadata_bundle(metadata.get(INFERENCE_METADATA_KEY, ""))
    has_inference_normalization_stats = bool(
        inference_metadata.get("normalization_stats_by_ensemble")
    )
    normalization_source = "none"
    if has_inference_normalization_stats:
        normalization_source = "deployed_model_inference_metadata_v1"
        print("Using normalization statistics from deployed model inference_metadata_v1.")
    elif metadata_stats:
        normalization_source = "deployed_model_legacy_flat_metadata"
        print("Using normalization statistics from deployed model legacy flat metadata.")

    # 2. Load metadata config and merge template overrides
    try:
        if not TEMPLATE_CONFIG.exists():
             raise FileNotFoundError(f"Template config not found at {TEMPLATE_CONFIG}")

        metadata_config_raw = metadata.get("config", "")
        if not metadata_config_raw:
            raise ValueError("Deployed model metadata is missing 'config'.")
        config = Config(yaml.safe_load(metadata_config_raw))

        test_config = Config.from_file(TEMPLATE_CONFIG)
        config.update(test_config)
        _disable_dataset_target_requirements(config)
        # Set batch size for inference
        config['batch_size'] = batch_size
        config["denormalize_inference_outputs"] = bool(destandardize and has_inference_normalization_stats)
        apply_global_config(config.as_dict(), warn_on_override=False)
        print("Configuration loaded and merged.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load or merge configuration: {e}")
    def run_inference_from_npz(npz_output_path: Path, cleanup_npz: bool):
        # 3. Create the dataset from the generated NPZ
        frame_indices = None
        inference_npz_path = npz_output_path
        try:
            with np.load(npz_output_path, allow_pickle=True) as npz_data:
                num_frames = int(npz_data["pos"].shape[0]) if "pos" in npz_data.files and npz_data["pos"].ndim > 0 else 0
            frame_indices = _parse_frame_slice(frame_slice, num_frames)
            # Temporarily update the dataset_input in the config to point to the new NPZ file
            config['test_dataset_list'][0]['dataset_input'] = str(inference_npz_path)
            if frame_indices is not None:
                config['test_dataset_list'][0]['include_frames'] = frame_indices
                print(
                    f"Using frame slice {frame_slice}: {len(frame_indices)} of {num_frames} frame(s).",
                    flush=True,
                )
            elif 'include_frames' in config['test_dataset_list'][0]:
                config['test_dataset_list'][0].pop('include_frames')

            dataset_started = time.time()
            dataset_config = _make_inference_dataset_config(config)
            builder = DatasetBuilder(dataset_config, np.random.default_rng(config.get('dataset_seed', 42)))
            inference_dset = builder.build_test()
            dataloader = DataLoader(inference_dset, batch_size=config['batch_size'], shuffle=False)
            print(
                f"DatasetBuilder created DataLoader with {len(inference_dset)} structures "
                f"in {time.time() - dataset_started:.1f}s.",
                flush=True,
            )
        except Exception as e:
            if cleanup_npz:
                npz_output_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Failed to build inference dataset: {e}")

        # 4. Run Inference
        mean_per_type, std_per_type = None, None
        if destandardize and not has_inference_normalization_stats:
            # --- Load standardization stats from deployment metadata ---
            try:
                mean_per_type_np = metadata_stats.get("_mean_.per_type.cs_iso")
                std_per_type_np = metadata_stats.get("_std_.per_type.cs_iso")
                if mean_per_type_np is None or std_per_type_np is None:
                    raise KeyError("Missing _mean_.per_type.cs_iso or _std_.per_type.cs_iso in metadata.")

                mean_per_type = torch.from_numpy(mean_per_type_np).to(device)
                std_per_type = torch.from_numpy(std_per_type_np).to(device)

                print("Successfully loaded de-standardization statistics from metadata.")
                
                # --- Diagnostic Check for Uniform Statistics (Addressing User Query) ---
                # Checks if all elements in the mean vector are numerically close to the first element.
                if mean_per_type.numel() > 0 and torch.allclose(mean_per_type, mean_per_type[0]):
                    print("\nWARNING: Loaded mean statistics are uniform across all element types.")
                    print("This indicates that the de-standardization is currently using a single, non-element-specific mean/std.")
                    print("To fix this, ensure the deployed model metadata contains valid, non-uniform, per-type statistics from the training set.")
                    print("Current mean shape:", mean_per_type.shape)
                # --- End Diagnostic Check ---
                
            except KeyError:
                if cleanup_npz:
                    npz_output_path.unlink(missing_ok=True)
                # Re-raise the original Key Error for missing stats keys
                raise HTTPException(
                    status_code=500, 
                    detail="De-standardization failed: '_mean_.per_type.cs_iso' or '_std_.per_type.cs_iso' not found in model metadata."
                )

        all_predictions = []
        model.eval()

        try:
            with torch.no_grad():
                total_batches = len(dataloader)
                print(f"Starting model inference on {total_batches} batch(es) using {device}.", flush=True)
                inference_started = time.time()
                progress = _progress_iter(dataloader, total_batches, "Inference")
                for data in progress:
                    data = data.to(device)
                    # Run inference without loss/metrics logic
                    out, _, _, _ = geq_run_inference(
                        model=model, data=data, device=device,
                        loss_fn=None, config=config.as_dict(), is_train=False,
                        inference_metadata=inference_metadata,
                    )
                    
                    # Extract predicted isotropic component ('cs_iso')
                    if 'cs_tensor_spherical' in out:
                        # Isotropic component is the first (l=0) component
                        predicted_cs_iso_std = out['cs_tensor_spherical'][:, 0:1].flatten()
                    elif 'cs_iso' in out:
                         # If the model directly outputs cs_iso
                        predicted_cs_iso_std = out['cs_iso'].flatten()
                    else:
                        raise KeyError("Model output is missing 'cs_tensor_spherical' or 'cs_iso'. Cannot extract isotropic shift.")
                    
                    # --- Conditionally De-standardize the predictions ---
                    if destandardize and not has_inference_normalization_stats:
                        # 1. Get atom types for the current batch
                        atom_types = data[AtomicDataDict.NODE_TYPE_KEY].flatten()
                        
                        # 2. Gather the corresponding mean and std for each atom
                        # mean_per_type has shape [num_atom_types, 1], atom_types has shape [num_atoms_in_batch]
                        # This gathers the correct mean/std for each atom based on its type.
                        means = mean_per_type[atom_types].flatten()
                        stds = std_per_type[atom_types].flatten()
                        
                        # 3. Apply de-standardization: original = (standardized * std) + mean
                        predicted_cs_iso_destd = (predicted_cs_iso_std * stds) + means
                        
                        all_predictions.append(predicted_cs_iso_destd.cpu().numpy())
                    else:
                        # If not de-standardizing, append the raw standardized output
                        all_predictions.append(predicted_cs_iso_std.cpu().numpy())
                if hasattr(progress, "close"):
                    progress.close()
                print(f"Model inference finished in {time.time() - inference_started:.1f}s.", flush=True)
                    
            # Concatenate all batches
            print("Concatenating predictions.", flush=True)
            final_predictions = np.concatenate(all_predictions, axis=0)
            
            # 5. Save Predictions to Output File(s)
            print("Writing inference output files.", flush=True)
            file_extension = input_path.suffix.lower()
            if file_extension in [".pdb", ".gro"]:
                output_paths, is_trajectory = processing.save_predictions_to_pdb(
                    input_path=input_path,
                    predictions_np=final_predictions,
                    output_dir=output_dir,
                    frame_indices=frame_indices,
                )

                # If it's a PDB trajectory, zip the individual frame files
                if is_trajectory:
                    zip_filename = f"{input_path.stem}_inferred_frames.zip"
                    zip_path = output_dir / zip_filename
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for file_path in output_paths:
                            zipf.write(file_path, arcname=file_path.name)
                            file_path.unlink() # Clean up individual PDBs
                    
                    final_output_path = zip_path
                else:
                    final_output_path = output_paths[0]
            elif file_extension == ".xyz":
                # For XYZ, save_predictions_to_xyz handles both single and multi-frame cases, returning one file.
                final_output_path = processing.save_predictions_to_xyz(
                    input_xyz_path=input_path,
                    predictions_np=final_predictions,
                    output_dir=output_dir,
                    frame_indices=frame_indices,
                )
            else:
                raise ValueError(f"Unsupported file type for saving predictions: {file_extension}")

            try:
                final_output_path.chmod(0o644)
            except OSError:
                pass

            _write_prediction_metadata(
                result_path=final_output_path,
                input_path=input_path,
                model_path=model_path,
                npz_path=npz_output_path,
                predictions=final_predictions,
                destandardize=destandardize,
                frame_slice=frame_slice,
                frame_indices=frame_indices,
                prepared=prepared_context,
                device=str(device),
                batch_size=batch_size,
                normalization_source=normalization_source,
            )

            # 6. Cleanup temporary NPZ file
            if cleanup_npz:
                npz_output_path.unlink(missing_ok=True)
            
            print("--- Inference Workflow Complete ---")
            return final_output_path, final_predictions.shape[0]
            
        except Exception as e:
            print(f"Error during inference execution: {e}")
            traceback.print_exc()
            if cleanup_npz:
                npz_output_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Error during model inference or file saving: {e}")

    if prepared_npz_path:
        config['root'] = str(prepared_npz_path.parent)
        return run_inference_from_npz(prepared_npz_path, cleanup_npz=False)

    with tempfile.TemporaryDirectory() as processing_dir:
        processing_dir_path = Path(processing_dir)
        config['root'] = str(processing_dir_path)

        # 3. Process the uploaded PDB/GRO/XYZ file to NPZ format (temporarily on disk)
        npz_output_path = processing_dir_path / f"temp_{input_path.stem}.npz"
        try:
            # This will create a temporary NPZ file containing atom coordinates and types
            npz_output_path, _num_molecules = processing.process_uploaded_file(
                input_path,
                processing_dir_path,
                trajectory_path,
                metadata_statistics=metadata_stats,
            )
            print(f"Structure processed into temporary NPZ: {npz_output_path}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process structure file into NPZ format for inference: {e}")

        return run_inference_from_npz(npz_output_path, cleanup_npz=True)


@app.post("/infer/pdb/")
async def infer_pdb_file(
    file: UploadFile = File(...),
    trajectory_file: UploadFile = File(None),
    model_name: str = Form(None),
    destandardize: bool = Form(True), # Default to True if not provided
    device: str = Form(None),
    batch_size: int = Form(1),
):
    """
    Endpoint to upload a PDB/GRO/XYZ file, run inference, and return the path to the modified structure file.
    """
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ["pdb", "gro", "xyz"]:
        raise HTTPException(status_code=400, detail="Only .pdb, .gro, or .xyz files are supported for this inference endpoint.")
    
    # 1. Securely save the uploaded file to a temporary location
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_filename = Path(file.filename).name
        input_path = temp_dir_path / input_filename
        try:
            with input_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            print(f"Error saving file: {e}")
            raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")
        finally:
            file.file.close()

        trajectory_input_path = None
        if trajectory_file:
            trajectory_filename = Path(trajectory_file.filename).name
            trajectory_input_path = temp_dir_path / trajectory_filename
            try:
                with trajectory_input_path.open("wb") as buffer:
                    shutil.copyfileobj(trajectory_file.file, buffer)
            except Exception as e:
                print(f"Error saving trajectory file: {e}")
                raise HTTPException(status_code=500, detail=f"Could not save uploaded trajectory file: {e}")
            finally:
                trajectory_file.file.close()
            print(f"Trajectory file saved to: {trajectory_input_path}")

        print(f"File saved to: {input_path}")
        
        # 2. Resolve model selection
        try:
            model_path = resolve_model_path(model_name) if model_name else get_default_model_path()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to resolve model path: {e}")

        # 3. Run Inference Workflow
        try:
            output_path, num_atoms_predicted = await run_inference_workflow(
                input_path=input_path, 
                output_dir=OUTPUT_DIR,
                destandardize=destandardize,
                model_path=model_path,
                trajectory_path=trajectory_input_path,
                device_name=device,
                batch_size=batch_size,
            )
            
            return {
                "message": "Inference complete. Structure file with predictions generated.",
                "input_file": file.filename,
                "model": str(model_path),
                "output_file": str(output_path),
                "atoms_predicted": num_atoms_predicted
            }
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"Internal Server Error during inference: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An internal error occurred during inference: {e}")


@app.get("/download/{filename:path}")
def download_file(filename: str):
    """
    Endpoint to allow downloading of the output files.
    """
    # Note: filename path is relative to the OUTPUT_DIR, e.g., "my_file_inferred.pdb"
    file_path = OUTPUT_DIR / filename
    
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
        
    # Set the media type based on the file extension
    media_type = "application/octet-stream"
    if file_path.suffix.lower() == ".pdb":
        media_type = "chemical/x-pdb"
    elif file_path.suffix.lower() == ".npz":
        media_type = "application/zip" # NPZ is often treated as a zip file type
    elif file_path.suffix.lower() == ".zip":
        media_type = "application/zip"
    elif file_path.suffix.lower() == ".xyz":
        media_type = "chemical/x-xyz"
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name
    )
