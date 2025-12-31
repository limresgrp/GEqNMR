from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import zipfile
import shutil
import tempfile
import json
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
from geqtrain.utils._global_options import apply_global_config

# --- Configuration (Hardcoded for this example) ---
MODELS_DIR = Path("/workspaces/GEqNMR/models")
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
OUTPUT_DIR = Path("/workspaces/GEqNMR/outputs")
RESULT_EXTENSIONS = {".pdb", ".xyz", ".zip"}
PREPARED_DIR = OUTPUT_DIR / "prepared_inputs"

# Ensure directories exist on startup
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREPARED_DIR.mkdir(parents=True, exist_ok=True)

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
            })
    results.sort(key=lambda item: item["modified"], reverse=True)
    return results


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")
    return {"deleted": file_path.name}


def _write_prepared_manifest(prepared_dir: Path, manifest: dict):
    manifest_path = prepared_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _load_prepared_manifest(prepared_dir: Path) -> dict:
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Prepared input manifest not found.")
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid prepared input manifest: {e}")


@app.post("/prepare")
async def prepare_input(
    file: UploadFile = File(...),
    trajectory_file: UploadFile = File(None),
):
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ["pdb", "gro", "xyz"]:
        raise HTTPException(status_code=400, detail="Only .pdb, .gro, or .xyz files are supported.")

    prepared_id = uuid.uuid4().hex
    prepared_dir = PREPARED_DIR / prepared_id
    prepared_dir.mkdir(parents=True, exist_ok=True)

    input_filename = Path(file.filename).name
    input_path = prepared_dir / input_filename
    trajectory_input_path = None
    trajectory_filename = None

    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
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
        except Exception as e:
            shutil.rmtree(prepared_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"Could not save uploaded trajectory file: {e}")
        finally:
            trajectory_file.file.close()

    try:
        npz_output_path, num_molecules = processing.process_uploaded_file(
            input_path,
            prepared_dir,
            trajectory_input_path,
        )
        summary = summarize_npz(npz_output_path)
    except Exception as e:
        shutil.rmtree(prepared_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to prepare input for inference: {e}")

    manifest = {
        "input_file": input_filename,
        "trajectory_file": trajectory_filename,
        "npz_file": npz_output_path.name,
        "num_molecules": num_molecules,
    }
    _write_prepared_manifest(prepared_dir, manifest)

    return {
        "id": prepared_id,
        "input_file": input_filename,
        "trajectory_file": trajectory_filename,
        "npz_file": npz_output_path.name,
        "num_molecules": num_molecules,
        "keys": summary,
    }


@app.get("/prepare/{prepared_id}/keys/{key_name}")
def get_prepared_key(
    prepared_id: str,
    key_name: str,
    batch_index: Optional[int] = None,
):
    prepared_dir = PREPARED_DIR / prepared_id
    if not prepared_dir.is_dir():
        raise HTTPException(status_code=404, detail="Prepared input not found.")

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
    prepared_dir = PREPARED_DIR / prepared_id
    if not prepared_dir.is_dir():
        raise HTTPException(status_code=404, detail="Prepared input not found.")

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
    np.savez_compressed(npz_path, **save_dict)

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
):
    prepared_dir = PREPARED_DIR / prepared_id
    if not prepared_dir.is_dir():
        raise HTTPException(status_code=404, detail="Prepared input not found.")

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
        )
        shutil.rmtree(prepared_dir, ignore_errors=True)
        return {
            "message": "Inference complete. Structure file with predictions generated.",
            "input_file": manifest["input_file"],
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
):
    """
    Orchestrates loading, inference, and file saving for an uploaded structure file.
    It saves PDB/GRO input as PDB (with B-factors) and XYZ input as Extended XYZ.
    """
    print(f"--- Starting Inference Workflow (De-standardize: {destandardize}) ---")
    
    # Check if a CUDA device is available and use it
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
        # Set batch size for inference
        config['batch_size'] = config.get('batch_size', 1)
        apply_global_config(config.as_dict(), warn_on_override=False)
        print("Configuration loaded and merged.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load or merge configuration: {e}")
    def run_inference_from_npz(npz_output_path: Path, cleanup_npz: bool):
        # 3. Create the dataset from the generated NPZ
        try:
            # Temporarily update the dataset_input in the config to point to the new NPZ file
            config['test_dataset_list'][0]['dataset_input'] = str(npz_output_path)

            builder = DatasetBuilder(config, np.random.default_rng(config.get('dataset_seed', 42)))
            inference_dset = builder.build_test()
            dataloader = DataLoader(inference_dset, batch_size=config['batch_size'], shuffle=False)
            print(f"DatasetBuilder created DataLoader with {len(inference_dset)} structures.")
        except Exception as e:
            if cleanup_npz:
                npz_output_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Failed to build inference dataset: {e}")

        # 4. Run Inference
        mean_per_type, std_per_type = None, None
        if destandardize:
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
                    print("To fix this, ensure your 'template.npz' contains valid, non-uniform, per-type statistics from the training set.")
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
                for data in dataloader:
                    data = data.to(device)
                    # Run inference without loss/metrics logic
                    out, _, _, _ = geq_run_inference(
                        model=model, data=data, device=device,
                        loss_fn=None, config=config.as_dict(), is_train=False
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
                    if destandardize:
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
                    
            # Concatenate all batches
            final_predictions = np.concatenate(all_predictions, axis=0)
            
            # 5. Save Predictions to Output File(s)
            file_extension = input_path.suffix.lower()
            if file_extension in [".pdb", ".gro"]:
                output_paths, is_trajectory = processing.save_predictions_to_pdb(
                    input_path=input_path,
                    predictions_np=final_predictions,
                    output_dir=output_dir
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
                    output_dir=output_dir
                )
            else:
                raise ValueError(f"Unsupported file type for saving predictions: {file_extension}")

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
    destandardize: bool = Form(True) # Default to True if not provided
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
                trajectory_path=trajectory_input_path
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
