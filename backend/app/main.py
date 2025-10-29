from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import zipfile
import shutil
from pathlib import Path
import traceback
import torch
import numpy as np

# Import core components and utility functions
from . import processing

# --- geqtrain Imports ---
from geqtrain.utils import Config
from geqtrain.data.dataloader import DataLoader
from geqtrain.train.components.dataset_builder import DatasetBuilder
from geqtrain.train.components.checkpointing import CheckpointHandler
from geqtrain.data import AtomicDataDict # Assuming this holds keys like 'pos', 'node_types' etc.
from geqtrain.train.components.inference import run_inference as geq_run_inference

# --- Configuration (Hardcoded for this example) ---
# **IMPORTANT**: Update these paths to point to a valid geqtrain checkpoint directory
MODEL_DIR = Path("/workspaces/GEqNMR/train_session/default_model")
MODEL_CHECKPOINT = "best_model.pth"
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

# --- Define upload and output directories ---
UPLOAD_DIR = Path("/workspaces/GEqNMR/uploads")
OUTPUT_DIR = Path("/workspaces/GEqNMR/outputs")

# Ensure directories exist on startup
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def read_root():
    """Root endpoint to check if the backend is alive."""
    return {"message": "Hello from the FastAPI Backend! (Connection Successful)"}


# --- UTILITY: Simplified Inference Workflow ---
async def run_inference_workflow(input_path: Path, output_dir: Path, destandardize: bool, trajectory_path: Path = None):
    """
    Orchestrates loading, inference, and file saving for an uploaded structure file.
    It saves PDB/GRO input as PDB (with B-factors) and XYZ input as Extended XYZ.
    """
    print(f"--- Starting Inference Workflow (De-standardize: {destandardize}) ---")
    
    # Check if a CUDA device is available and use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model and its original training config
    try:
        model, config = CheckpointHandler.load_model_from_training_session(
            traindir=MODEL_DIR, model_name=MODEL_CHECKPOINT, device=device.type
        )
        print(f"Model {MODEL_CHECKPOINT} loaded successfully from {MODEL_DIR}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Use an internal error code (500) since this is a setup issue
        raise HTTPException(status_code=500, detail=f"Failed to load model checkpoint. Ensure model path is correct: {MODEL_DIR}/{MODEL_CHECKPOINT}. Error: {e}")

    # 2. Load Template Config and merge it to original training config
    try:
        if not TEMPLATE_CONFIG.exists():
             raise FileNotFoundError(f"Template config not found at {TEMPLATE_CONFIG}")
             
        test_config = Config.from_file(TEMPLATE_CONFIG)
        config.update(test_config)
        # Set batch size for inference
        config['batch_size'] = config.get('batch_size', 1) 
        print("Configuration loaded and merged.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load or merge configuration: {e}")
    # 3. Process the uploaded PDB/GRO/XYZ file to NPZ format (temporarily on disk)
    npz_output_path = OUTPUT_DIR / f"temp_{input_path.stem}.npz"
    try:
        # This will create a temporary NPZ file containing atom coordinates and types
        npz_output_path, num_molecules = processing.process_uploaded_file(input_path, OUTPUT_DIR, trajectory_path)
        print(f"Structure processed into temporary NPZ: {npz_output_path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process structure file into NPZ format for inference: {e}")
    
    # 4. Create the dataset from the generated NPZ
    try:
        # Temporarily update the dataset_input in the config to point to the new NPZ file
        config['test_dataset_list'][0]['dataset_input'] = str(npz_output_path)
        
        builder = DatasetBuilder(config, np.random.default_rng(config.get('dataset_seed', 42)))
        inference_dset = builder.build_test()
        dataloader = DataLoader(inference_dset, batch_size=config['batch_size'], shuffle=False)
        print(f"DatasetBuilder created DataLoader with {len(inference_dset)} structures.")
    except Exception as e:
        npz_output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to build inference dataset: {e}")
    
    # 5. Run Inference
    mean_per_type, std_per_type = None, None
    if destandardize:
        # --- Load standardization stats from the NPZ file for de-standardization ---
        try:
            with np.load(npz_output_path) as data:
                # Load numpy arrays for diagnostics
                mean_per_type_np = data['_mean_.per_type.cs_iso']
                std_per_type_np = data['_std_.per_type.cs_iso']
                
                # Convert to PyTorch tensors for use in de-standardization
                mean_per_type = torch.from_numpy(mean_per_type_np).to(device)
                std_per_type = torch.from_numpy(std_per_type_np).to(device)
                
            print("Successfully loaded de-standardization statistics (_mean_ and _std_).")
            
            # --- Diagnostic Check for Uniform Statistics (Addressing User Query) ---
            # Checks if all elements in the mean vector are numerically close to the first element.
            if mean_per_type.numel() > 0 and torch.allclose(mean_per_type, mean_per_type[0]):
                print("\nWARNING: Loaded mean statistics are uniform across all element types.")
                print("This indicates that the de-standardization is currently using a single, non-element-specific mean/std.")
                print("To fix this, ensure your 'template.npz' contains valid, non-uniform, per-type statistics from the training set.")
                print("Current mean shape:", mean_per_type.shape)
            # --- End Diagnostic Check ---
            
        except KeyError:
            npz_output_path.unlink(missing_ok=True)
            # Re-raise the original Key Error for missing stats keys
            raise HTTPException(
                status_code=500, 
                detail="De-standardization failed: '_mean_.per_type.cs_iso' or '_std_.per_type.cs_iso' not found in the generated NPZ file. Ensure 'template.npz' is available and contains these keys."
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
        
        # 6. Save Predictions to Output File(s)
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

        # 7. Cleanup temporary NPZ file
        npz_output_path.unlink(missing_ok=True)
        
        print("--- Inference Workflow Complete ---")
        return final_output_path, final_predictions.shape[0]
        
    except Exception as e:
        print(f"Error during inference execution: {e}")
        traceback.print_exc()
        npz_output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error during model inference or file saving: {e}")


@app.post("/infer/pdb/")
async def infer_pdb_file(
    file: UploadFile = File(...),
    trajectory_file: UploadFile = File(None),
    destandardize: bool = Form(True) # Default to True if not provided
):
    """
    Endpoint to upload a PDB/GRO/XYZ file, run inference, and return the path to the modified structure file.
    """
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ["pdb", "gro", "xyz"]:
        raise HTTPException(status_code=400, detail="Only .pdb, .gro, or .xyz files are supported for this inference endpoint.")
    
    # 1. Securely save the uploaded file
    input_path = UPLOAD_DIR / file.filename
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
        trajectory_input_path = UPLOAD_DIR / trajectory_file.filename
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
    
    # 2. Run Inference Workflow
    try:
        output_path, num_atoms_predicted = await run_inference_workflow(
            input_path=input_path, 
            output_dir=OUTPUT_DIR,
            destandardize=destandardize,
            trajectory_path=trajectory_input_path
        )
        
        return {
            "message": "Inference complete. Structure file with predictions generated.",
            "input_file": file.filename,
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
