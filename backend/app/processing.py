import numpy as np
import shlex
from pathlib import Path
import torch
from e3nn.o3 import Irreps
import MDAnalysis as mda
from MDAnalysis.exceptions import NoDataError
import yaml

# --- Import from geqtrain (installed via pip) ---
from .cartesian_to_spherical import convert_cartesian_to_spherical
from geqtrain.utils.pytorch_scatter import scatter_mean, scatter_std

# --- Configuration (from build_dataset.py) ---
STANDARDIZATION_CONFIG = {
    "cs_tensor_spherical": "per_type:1x0e+1x1o+1x2e",
}
MEAN_KEY_PREFIX = "_mean_"
STD_KEY_PREFIX = "_std_"
PER_TYPE_PREFIX = "per_type"
GLOBAL_PREFIX = "global"

ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36
    # Add more if needed
}
MAX_ATOMIC_NUMBER = max(ATOMIC_NUMBERS.values())

PERIODIC_TABLE_INFO = {
    'H': (1, 1), 'He': (1, 18), 'Li': (2, 1), 'Be': (2, 2), 'B': (2, 13), 'C': (2, 14),
    'N': (2, 15), 'O': (2, 16), 'F': (2, 17), 'Ne': (2, 18), 'Na': (3, 1), 'Mg': (3, 2),
    'Al': (3, 13), 'Si': (3, 14), 'P': (3, 15), 'S': (3, 16), 'Cl': (3, 17), 'Ar': (3, 18),
    'K': (4, 1), 'Ca': (4, 2), 'Sc': (4, 3), 'Ti': (4, 4), 'V': (4, 5), 'Cr': (4, 6),
    'Mn': (4, 7), 'Fe': (4, 8), 'Co': (4, 9), 'Ni': (4, 10), 'Cu': (4, 11), 'Zn': (4, 12),
    'Ga': (4, 13), 'Ge': (4, 14), 'As': (4, 15), 'Se': (4, 16), 'Br': (4, 17), 'Kr': (4, 18)
    # Add more if needed
}

# --- NEW UTILITY: Reverse Map (Atomic Number -> Symbol) ---
# Used for writing XYZ files based on the atomic number used for de-standardization.
ATOMIC_SYMBOLS = {v: k for k, v in ATOMIC_NUMBERS.items()}
# Add symbol for unrecognized atoms (index 0)
ATOMIC_SYMBOLS[0] = 'X' 

# --- NEW UTILITY: Load YAML Config ---
def load_config_from_yaml(config_path: Path):
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise IOError(f"Failed to load config from {config_path}: {e}")

# --- FUNCTION: Save Predictions to PDB (Used for PDB/GRO input) ---

def save_predictions_to_pdb(input_path: Path, predictions_np: np.ndarray, output_dir: Path):
    """
    Loads a structure, updates the B-factor column with predictions, and saves new PDB file(s).
    If the input is a trajectory (multi-frame), it saves one PDB per frame.
    Returns a list of output paths and a boolean indicating if it was a trajectory.
    """
    print(f"--- Writing predictions from {input_path.name} to PDB format ---")
    
    universe = mda.Universe(str(input_path))
    atoms = universe.atoms
    
    # Calculate the total number of atoms in the entire trajectory
    n_atoms_per_frame = len(atoms)
    total_frames = len(universe.trajectory)
    total_atoms_predicted = n_atoms_per_frame * total_frames
    
    # Ensure predictions match the total number of atoms in all frames
    if len(predictions_np) != total_atoms_predicted:
        raise ValueError(
            f"Prediction count ({len(predictions_np)}) does not match total atom count ({total_atoms_predicted}) "
            f"in the structure file ({n_atoms_per_frame} atoms * {total_frames} frames). Aborting PDB writing."
        )
    
    # MDAnalysis automatically handles B-factor (tempfactors) as a float array
    # FIX: Explicitly reshape predictions_np to a 1D vector before slicing/clipping.
    b_factors = predictions_np.reshape(-1).astype(np.float32)
    b_factors_clipped = np.clip(b_factors, -999.99, 999.99)
    
    is_trajectory = total_frames > 1
    output_paths = []

    if is_trajectory:
        print(f"Detected trajectory with {total_frames} frames. Saving each frame as a separate PDB.")
        atom_cursor = 0
        for i, ts in enumerate(universe.trajectory):
            frame_predictions = b_factors_clipped[atom_cursor : atom_cursor + n_atoms_per_frame]
            universe.atoms.tempfactors = frame_predictions
            
            output_filename = f"{input_path.stem}_inferred_frame_{i+1}.pdb"
            frame_output_path = output_dir / output_filename
            
            # Select only the current frame for writing
            universe.trajectory[i]
            atoms.write(str(frame_output_path))
            output_paths.append(frame_output_path)
            atom_cursor += n_atoms_per_frame
        print(f"Saved {len(output_paths)} PDB files.")
    else:
        # Single frame file
        output_filename = f"{input_path.stem}_inferred.pdb"
        output_path = output_dir / output_filename
        universe.atoms.tempfactors = b_factors_clipped
        atoms.write(str(output_path))
        output_paths.append(output_path)
        print(f"Predictions saved as B-factors to: {output_path}")

    return output_paths, is_trajectory


def save_predictions_to_xyz(input_xyz_path: Path, predictions_np: np.ndarray, output_dir: Path):
    """
    Reads an XYZ structure, appends cs_iso predictions as a new column,
    and saves an extended XYZ file.
    
    CRITICAL FIX: This function now re-parses the structure to reliably get atomic numbers
    for element symbol lookup, ensuring the output element symbol matches the de-standardized
    prediction.
    """
    print(f"--- Writing predictions to {input_xyz_path.name} as extended XYZ ---")

    # 1. Load the universe to access elements and positions, and iterate over frames
    try:
        universe = mda.Universe(str(input_xyz_path))
    except Exception as e:
        raise IOError(f"Failed to read structure for XYZ writing: {e}")
        
    n_atoms_per_frame = len(universe.atoms)
    total_frames = len(universe.trajectory)
    total_atoms_predicted = n_atoms_per_frame * total_frames
    
    if len(predictions_np) != total_atoms_predicted:
        raise ValueError(
            f"Prediction count ({len(predictions_np)}) does not match total atom count ({total_atoms_predicted}) "
            "in the structure file. Aborting XYZ writing."
        )

    output_filename = f"{input_xyz_path.stem}_inferred.xyz"
    output_path = output_dir / output_filename
    
    # 2. Re-run parsing logic to get the reliable atomic number array (atom_types)
    # This ensures consistency with the atom_types used by the DataLoader
    all_molecules_data = parse_structure_file_mdanalysis(input_xyz_path)
    
    if not all_molecules_data:
        raise ValueError("Failed to re-parse atomic data for XYZ writing.")
        
    # Collate all atom_types into a single flat array
    atomic_numbers_array = np.concatenate([mol['atom_types'] for mol in all_molecules_data])
    
    # 3. Prepare data for writing
    # FIX: Explicitly reshape predictions_np to a 1D vector
    cs_iso_predictions = predictions_np.reshape(-1).astype(np.float64) 

    with open(output_path, 'w') as f:
        atom_cursor = 0
        for ts_idx, ts in enumerate(universe.trajectory):
            atoms = universe.atoms
            
            # --- Write Header ---
            f.write(f"{n_atoms_per_frame}\n")
            
            header_parts = ["pbc=F"]
            
            # Check if dimensions are available AND non-zero before attempting np.allclose
            if ts.dimensions is not None and not np.allclose(ts.dimensions, 0.0):
                 # Format: "a 0 0 0 b 0 0 0 c"
                 lattice_str = ' '.join([f"{ts.dimensions[i]:.6f}" if i%4 != 0 else f"{ts.dimensions[i]:.6f}" for i in range(9)])
                 header_parts.append(f"Lattice=\"{lattice_str}\"")

            # Define atom-level properties: species (string, 1 dim), pos (real, 3 dim), cs_iso (real, 1 dim)
            header_parts.append("Properties=species:S:1:pos:R:3:cs_iso:R:1")
            f.write(f"{' '.join(header_parts)}\n")
            
            # --- Write Atom Lines ---
            for i in range(n_atoms_per_frame):
                # CRITICAL: Use the pre-calculated atomic number to look up the element symbol
                current_atom_index = atom_cursor + i
                atomic_num = atomic_numbers_array[current_atom_index]
                
                # Look up symbol using the confirmed atomic number (index)
                element = ATOMIC_SYMBOLS.get(atomic_num, 'X')
                
                # Positions (x, y, z)
                pos = atoms.positions[i]
                
                # Predicted cs_iso 
                cs_iso_pred = cs_iso_predictions[current_atom_index]
                
                # Write line: Element X Y Z cs_iso
                f.write(f"{element} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {cs_iso_pred:.4f}\n")
            
            atom_cursor += n_atoms_per_frame
            
    print(f"Predictions saved as 'cs_iso' property to: {output_path}")
    return output_path

# --- Existing Function: MDAnalysis Parser (for PDB/GRO) ---
def parse_structure_file_mdanalysis(filepath: Path, trajectory_path: Path = None):
    """
    Parses PDB or GRO files using MDAnalysis for inference.
    Only extracts positions and atom types (atomic numbers).
    If a trajectory_path is provided, it's loaded with the structure.
    """
    print(f"--- Parsing {filepath.name} with MDAnalysis (Inference Mode) ---")
    if trajectory_path:
        print(f"--- Loading trajectory from {trajectory_path.name} ---")
    
    try:
        if trajectory_path:
            universe = mda.Universe(str(filepath), str(trajectory_path))
        else:
            universe = mda.Universe(str(filepath))
    except Exception as e:
        print(f"Error: MDAnalysis failed to read {filepath}. Reason: {e}")
        return []

    molecules_data = []
    # Process each frame (timestep) in the file
    for ts in universe.trajectory:
        try:
            atoms = universe.atoms
            num_atoms = len(atoms)
            if num_atoms == 0:
                continue

            mol_dict = {'num_atoms': num_atoms}

            # Get positions
            mol_dict['pos'] = atoms.positions.astype(np.float32)

            # Get atom types (Atomic Numbers)
            atom_types = np.zeros(num_atoms, dtype=np.int64)
            atom_rows = np.zeros(num_atoms, dtype=np.int8)
            atom_cols = np.zeros(num_atoms, dtype=np.int8)

            # Try to get element symbols (best)
            try:
                # Use elements attribute directly if available and reliable
                elements = atoms.elements
                
                for i, el in enumerate(elements):
                    symbol = str(el).capitalize() # Ensure compatibility with ATOMIC_NUMBERS keys
                    
                    # Store atomic number (e.g., C=6, O=8, Ca=20)
                    atomic_num = ATOMIC_NUMBERS.get(symbol, 0)
                    atom_types[i] = atomic_num
                    
                    # Look up row/col
                    row, col = PERIODIC_TABLE_INFO.get(symbol, (0, 0))
                    atom_rows[i] = row
                    atom_cols[i] = col
                    
            except (NoDataError, AttributeError, ValueError):
                # Fallback to atom names (less reliable)
                print("Warning: 'elements' not found. Falling back to 'names' for atom types.")
                atom_names = atoms.names
                for i, name in enumerate(atom_names):
                    # Robust symbol extraction: try 2-letter elements first, then 1-letter, ignore digits
                    symbol_upper = ''.join(filter(str.isalpha, name)).upper()
                    symbol = 'X' # Default unknown
                    
                    if len(symbol_upper) >= 2 and symbol_upper[:2] in [s.upper() for s in ATOMIC_NUMBERS.keys() if len(s) == 2]:
                        # e.g., 'CLA' -> 'CL' -> 'Cl' (atomic number 17)
                        symbol = symbol_upper[:2].capitalize()
                    elif len(symbol_upper) >= 1 and symbol_upper[0] in [s.upper() for s in ATOMIC_NUMBERS.keys() if len(s) == 1]:
                        # e.g., 'C1' -> 'C'
                        symbol = symbol_upper[0].capitalize()
                    
                    # Store atomic number (e.g., C=6, O=8, Ca=20)
                    atomic_num = ATOMIC_NUMBERS.get(symbol, 0)
                    atom_types[i] = atomic_num
                    
                    # Look up row/col
                    row, col = PERIODIC_TABLE_INFO.get(symbol, (0, 0))
                    atom_rows[i] = row
                    atom_cols[i] = col

            mol_dict['atom_types'] = atom_types
            mol_dict['atom_rows'] = atom_rows
            mol_dict['atom_cols'] = atom_cols

            # Add center_atoms_mask (assume all atoms are relevant for inference)
            mol_dict['center_atoms_mask'] = np.ones(num_atoms, dtype=bool)
            
            molecules_data.append(mol_dict)

        except Exception as e:
            print(f"Error processing frame {ts.frame}: {e}")
            
    print(f"Successfully parsed {len(molecules_data)} frames/molecules.")
    # DIAGNOSTIC: Check extracted atomic numbers
    unique_types = np.unique(np.concatenate([mol['atom_types'] for mol in molecules_data]))
    print(f"Extracted unique atomic numbers (indices): {unique_types}")
    if np.any(unique_types == 0):
        print("CRITICAL WARNING: Zero (0) is present in unique atomic numbers. This likely means some atoms were not recognized and will use the statistics for index 0.")
        
    return molecules_data


# --- Functions from build_dataset.py (refactored) ---

def parse_properties(properties_str: str):
    """Parses the 'Properties' string from the XYZ header."""
    props = properties_str.split(':')
    prop_info = {}
    current_col = 0
    for i in range(0, len(props), 3):
        name = props[i]
        try:
            ptype = props[i+1]
            dim = int(props[i+2])
        except IndexError:
            print(f"Warning: Malformed Properties string fragment near '{name}'. Skipping subsequent properties.")
            break

        col_slice = slice(current_col, current_col + dim)

        dtype = str
        if ptype == 'R': dtype = np.float32
        elif ptype == 'L': dtype = bool
        elif ptype == 'I': dtype = np.int_
        elif ptype == 'S': dtype = np.str_ # Explicitly handle string type
        else:
             print(f"Warning: Unknown property type '{ptype}' for property '{name}'. Treating as string.")

        prop_info[name] = {"slice": col_slice, "dim": dim, "dtype": dtype}
        current_col += dim
    return prop_info

def parse_extxyz_file(filepath: Path, cartesian_to_spherical_converter_func):
    """
    Reads an extended XYZ file, converts Cartesian CS tensor to spherical,
    and calculates CS iso from the spherical tensor.
    """
    molecules_data = []
    print(f"--- Reading and parsing {filepath.name} (XYZ Mode) ---")
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error: Could not read file {filepath}. Reason: {e}")
        return []

    line_idx = 0
    mol_count = 0
    while line_idx < len(lines):
        try:
            # --- Header Parsing ---
            if not lines[line_idx].strip(): # Skip empty lines
                line_idx += 1
                continue
            num_atoms = int(lines[line_idx].strip())
            header = lines[line_idx+1].strip()

            mol_dict = {'num_atoms': num_atoms}
            properties_str = ""

            header_parts = shlex.split(header)
            for part in header_parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    if key == 'Properties':
                        properties_str = value
                    else:
                        try: mol_dict[key] = float(value)
                        except ValueError:
                            try: mol_dict[key] = np.fromstring(value.replace('"',''), sep=' ', dtype=np.float32) # Handle potential quotes and arrays
                            except (ValueError, TypeError): mol_dict[key] = value # Store as string if cannot convert

            if not properties_str:
                 raise ValueError("Missing 'Properties' definition in header.")

            prop_map = parse_properties(properties_str)
            atom_lines = lines[line_idx+2 : line_idx+2 + num_atoms]
            if len(atom_lines) != num_atoms:
                raise ValueError(f"Expected {num_atoms} atom lines, but found {len(atom_lines)}.")


            # --- Atom Data Parsing ---
            raw_atom_data = {}
            for key, info in prop_map.items():
                shape = (num_atoms, info['dim']) if info['dim'] > 1 else (num_atoms,)
                 # Handle string type specifically for species
                if info['dtype'] == np.str_:
                     raw_atom_data[key] = np.empty(shape, dtype=object)
                else:
                    raw_atom_data[key] = np.zeros(shape, dtype=info['dtype'])

            mol_dict['atom_types'] = np.zeros(num_atoms, dtype=np.int64) # Use int64 for PyTorch compatibility
            mol_dict['atom_rows'] = np.zeros(num_atoms, dtype=np.int8)
            mol_dict['atom_cols'] = np.zeros(num_atoms, dtype=np.int8)

            for i, line in enumerate(atom_lines):
                parts = line.split()
                current_part_idx = 0
                processed_keys = set() 

                sorted_prop_keys = sorted(prop_map.keys(), key=lambda k: prop_map[k]['slice'].start)

                for key in sorted_prop_keys:
                    if key in processed_keys: continue
                    info = prop_map[key]
                    num_parts_for_key = info['dim']
                    end_part_idx = current_part_idx + num_parts_for_key

                    if end_part_idx > len(parts):
                         raise ValueError(f"Line {line_idx+2+i}: Not enough columns for property '{key}'. Expected {num_parts_for_key} value(s), found {len(parts)-current_part_idx}.")

                    raw_vals = parts[current_part_idx : end_part_idx]

                    if key == 'species':
                         species_symbol = raw_vals[0]
                         raw_atom_data[key][i] = species_symbol
                         
                         # Store atomic number (e.g., C=6, O=8, Ca=20)
                         atomic_num = ATOMIC_NUMBERS.get(species_symbol, 0)
                         mol_dict['atom_types'][i] = atomic_num
                         
                         row, col = PERIODIC_TABLE_INFO.get(species_symbol, (0, 0))
                         mol_dict['atom_rows'][i] = row
                         mol_dict['atom_cols'][i] = col
                    elif info['dtype'] == bool:
                         val = True if raw_vals[0].upper() == 'T' else False
                         raw_atom_data[key][i] = val
                    elif info['dtype'] == np.str_:
                         val = " ".join(raw_vals) if info['dim'] > 1 else raw_vals[0]
                         raw_atom_data[key][i] = val
                    else: # Numerical types
                         try:
                             val = np.array(raw_vals, dtype=info['dtype'])
                             if info['dim'] > 1: raw_atom_data[key][i, :] = val
                             else: raw_atom_data[key][i] = val
                         except ValueError as conversion_error:
                              raise ValueError(f"Line {line_idx+2+i}, Property '{key}': Could not convert '{raw_vals}' to {info['dtype']}. Error: {conversion_error}")

                    processed_keys.add(key)
                    current_part_idx = end_part_idx

            # --- Process Chemical Shielding ---
            if 'cs_tensor' in raw_atom_data:
                cs_tensor_cartesian_flat = raw_atom_data['cs_tensor']
                if cs_tensor_cartesian_flat.shape[-1] != 9:
                    raise ValueError(f"Molecule {mol_count}: 'cs_tensor' should have 9 components, found {cs_tensor_cartesian_flat.shape[-1]}")

                cs_tensor_cartesian_3x3_np = cs_tensor_cartesian_flat.reshape(num_atoms, 3, 3)
                cs_tensor_cartesian_3x3_torch = torch.from_numpy(cs_tensor_cartesian_3x3_np).to(torch.float32)

                # Convert to spherical
                cs_tensor_spherical_torch = cartesian_to_spherical_converter_func(cs_tensor_cartesian_3x3_torch)
                mol_dict['cs_tensor_spherical'] = cs_tensor_spherical_torch.numpy()

                mol_dict['cs_iso'] = mol_dict['cs_tensor_spherical'][:, 0:1]

                if 'pos' in raw_atom_data: mol_dict['pos'] = raw_atom_data['pos']
                if 'forces' in raw_atom_data: mol_dict['forces'] = raw_atom_data['forces']
                if 'center_atoms_mask' in raw_atom_data: mol_dict['center_atoms_mask'] = raw_atom_data['center_atoms_mask']

            else:
                # This branch is taken for inference XYZ files that don't have ground truth
                print(f"Info: Molecule {mol_count} does not contain 'cs_tensor'. Assuming inference mode for this file.")
                if 'pos' in raw_atom_data: mol_dict['pos'] = raw_atom_data['pos']
                # Add center_atoms_mask (assume all atoms are relevant for inference)
                mol_dict['center_atoms_mask'] = np.ones(num_atoms, dtype=bool)

            molecules_data.append(mol_dict)
            mol_count += 1
            line_idx += num_atoms + 2
        except (ValueError, IndexError, KeyError) as e:
            print(f"Error: A parsing error occurred for molecule starting near line {line_idx+1}. Skipping molecule. Error: {e}")
            try:
                potential_num_atoms = int(lines[line_idx].strip())
                line_idx += potential_num_atoms + 2
            except:
                line_idx += 1 

    print(f"Successfully parsed {mol_count} molecules.")
    # DIAGNOSTIC: Check extracted atomic numbers
    unique_types = np.unique(np.concatenate([mol['atom_types'] for mol in molecules_data]))
    print(f"Extracted unique atomic numbers (indices): {unique_types}")
    if np.any(unique_types == 0):
        print("CRITICAL WARNING: Zero (0) is present in unique atomic numbers. This likely means some atoms were not recognized and will use the statistics for index 0.")
        
    return molecules_data


def compute_statistics(molecules, config):
    """Computes standardization statistics for specified fields."""
    print("\n--- Computing Standardization Statistics ---")
    statistics = {}
    num_types = MAX_ATOMIC_NUMBER + 1 

    for field, mode_str in config.items():
        print(f"Processing field: {field} with mode: {mode_str}")
        if not any(field in mol for mol in molecules):
            print(f"Warning: Field '{field}' not found in any molecule. Skipping statistics calculation.")
            continue

        parts = mode_str.split(':', 1)
        mode = parts[0]
        irreps_str = parts[1] if len(parts) > 1 else None
        irreps = Irreps(irreps_str) if irreps_str else None

        all_field_values_list = []
        all_node_types_list = []

        for i, mol in enumerate(molecules):
             if field in mol:
                 all_field_values_list.append(torch.from_numpy(mol[field]))
                 all_node_types_list.append(torch.from_numpy(mol['atom_types']))

        if not all_field_values_list:
             print(f"Warning: Field '{field}' has no data points. Skipping statistics.")
             continue

        all_field_values = torch.cat(all_field_values_list, dim=0).to(torch.float32)
        all_node_types = torch.cat(all_node_types_list, dim=0)

        if mode == 'per_type':
            if irreps:
                norm_values_list = []
                current_idx = 0
                for mol_field_tensor in all_field_values_list:
                    norms = []
                    for (mul, ir), slice_ in zip(irreps, irreps.slices()):
                        tensor_slice = mol_field_tensor[:, slice_]
                        if ir.l == 0:
                            norms.append(tensor_slice)
                        else:
                            norms.append(torch.linalg.norm(tensor_slice, dim=-1, keepdim=True))
                    norm_values_list.append(torch.cat(norms, dim=-1))
                values_for_stats = torch.cat(norm_values_list, dim=0).to(torch.float32)
            else:
                 values_for_stats = all_field_values

            means = scatter_mean(values_for_stats, all_node_types, dim=0, dim_size=num_types)
            stds = scatter_std(values_for_stats, all_node_types, dim=0, dim_size=num_types)

            stds = torch.where(torch.isnan(stds) | (stds < 1e-8), torch.ones_like(stds), stds)
            means = torch.nan_to_num(means, nan=0.0) 

            mean_key = f"{MEAN_KEY_PREFIX}.{PER_TYPE_PREFIX}.{field}"
            std_key = f"{STD_KEY_PREFIX}.{PER_TYPE_PREFIX}.{field}"
            statistics[mean_key] = means.numpy()
            statistics[std_key] = stds.numpy()

            if field == "cs_tensor_spherical":
                mean_key = f"{MEAN_KEY_PREFIX}.{PER_TYPE_PREFIX}.cs_iso"
                std_key = f"{STD_KEY_PREFIX}.{PER_TYPE_PREFIX}.cs_iso"
                statistics[mean_key] = means.numpy()[..., :1]
                statistics[std_key] = stds.numpy()[..., :1]
            print(f"  Computed per-type stats. Mean shape: {means.shape}, Std shape: {stds.shape}")

        elif mode == 'global':
            if irreps:
                norm_values_list = []
                for mol_field_tensor in all_field_values_list:
                     norms = []
                     for (mul, ir), slice_ in zip(irreps, irreps.slices()):
                         tensor_slice = mol_field_tensor[:, slice_]
                         if ir.l == 0: norms.append(tensor_slice)
                         else: norms.append(torch.linalg.norm(tensor_slice, dim=-1, keepdim=True))
                     norm_values_list.append(torch.cat(norms, dim=-1))
                values_for_stats = torch.cat(norm_values_list, dim=0).to(torch.float32)
            else:
                 values_for_stats = all_field_values

            mean_val = torch.mean(values_for_stats)
            std_val = torch.std(values_for_stats)
            std_val = torch.ones_like(std_val) if std_val < 1e-8 else std_val 

            mean_key = f"{MEAN_KEY_PREFIX}.{GLOBAL_PREFIX}.{field}"
            std_key = f"{STD_KEY_PREFIX}.{GLOBAL_PREFIX}.{field}"
            statistics[mean_key] = mean_val.numpy()
            statistics[std_key] = std_val.numpy()
            
            if field == "cs_tensor_spherical":
                mean_key = f"{MEAN_KEY_PREFIX}.{GLOBAL_PREFIX}.cs_iso"
                std_key = f"{STD_KEY_PREFIX}.{GLOBAL_PREFIX}.cs_iso"
                statistics[mean_key] = mean_val.numpy()[..., :1]
                statistics[std_key] = std_val.numpy()[..., :1]
            print(f"  Computed global stats. Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        else:
             print(f"Warning: Unknown standardization mode '{mode}' for field '{field}'. Skipping.")

    return statistics


def standardize_data(molecules, statistics, config):
    """Applies standardization to specified fields."""
    print("\n--- Applying Standardization ---")
    num_types = MAX_ATOMIC_NUMBER + 1 

    for field, mode_str in config.items():
        if not any(field in mol for mol in molecules): continue 

        parts = mode_str.split(':', 1)
        mode = parts[0]
        irreps_str = parts[1] if len(parts) > 1 else None
        irreps = Irreps(irreps_str) if irreps_str else None

        mean_val, std_val = None, None
        means_per_type, stds_per_type = None, None

        if mode == 'per_type':
            mean_key = f"{MEAN_KEY_PREFIX}.{PER_TYPE_PREFIX}.{field}"
            std_key = f"{STD_KEY_PREFIX}.{PER_TYPE_PREFIX}.{field}"
            if mean_key in statistics and std_key in statistics:
                means_per_type = torch.from_numpy(statistics[mean_key]).float()
                stds_per_type = torch.from_numpy(statistics[std_key]).float()
            else:
                 print(f"Warning: Per-type stats not found for '{field}'. Skipping standardization.")
                 continue
        elif mode == 'global':
            mean_key = f"{MEAN_KEY_PREFIX}.{GLOBAL_PREFIX}.{field}"
            std_key = f"{STD_KEY_PREFIX}.{GLOBAL_PREFIX}.{field}"
            if mean_key in statistics and std_key in statistics:
                mean_val = torch.from_numpy(statistics[mean_key]).float()
                std_val = torch.from_numpy(statistics[std_key]).float()
            else:
                 print(f"Warning: Global stats not found for '{field}'. Skipping standardization.")
                 continue
        else: continue 

        print(f"Standardizing field: {field}")
        for mol in molecules:
            if field not in mol: continue

            values_torch = torch.from_numpy(mol[field]).float().clone()
            node_types = torch.from_numpy(mol['atom_types'])

            if mode == 'per_type':
                means_expanded = means_per_type[node_types]
                stds_expanded = stds_per_type[node_types]
                if irreps:
                    i = 0
                    for (mul, ir), slice_ in zip(irreps, irreps.slices()):
                        mean_bc = means_expanded[:, i:i+1]
                        std_bc = stds_expanded[:, i:i+1]
                        if ir.l == 0:
                            values_torch[:, slice_] = (values_torch[:, slice_] - mean_bc) / std_bc
                        else: # l > 0
                            tensor_slice = values_torch[:, slice_]
                            norm = torch.linalg.norm(tensor_slice, dim=-1, keepdim=True)
                            safe_norm = norm.clamp(min=1e-8)
                            safe_std = std_bc.clamp(min=1e-8)
                            standardized_norm = (norm - mean_bc) / safe_std
                            scale = standardized_norm / safe_norm
                            values_torch[:, slice_] = tensor_slice * scale
                        i += 1
                else: # Scalar per-type
                     values_torch = (values_torch - means_expanded) / stds_expanded.clamp(min=1e-8)

            elif mode == 'global':
                 if std_val > 1e-8:
                     if irreps:
                         i = 0
                         for (mul, ir), slice_ in zip(irreps, irreps.slices()):
                             mean_component = mean_val[i] 
                             std_component = std_val[i].clamp(min=1e-8)
                             if ir.l == 0:
                                  values_torch[:, slice_] = (values_torch[:, slice_] - mean_component) / std_component
                             else:
                                 tensor_slice = values_torch[:, slice_]
                                 norm = torch.linalg.norm(tensor_slice, dim=-1, keepdim=True)
                                 safe_norm = norm.clamp(min=1e-8)
                                 standardized_norm = (norm - mean_component) / std_component
                                 scale = standardized_norm / safe_norm
                                 values_torch[:, slice_] = tensor_slice * scale
                             i += 1
                     else: # Scalar global
                          values_torch = (values_torch - mean_val) / std_val

            standardized_field_name = f"{field}_std"
            mol[standardized_field_name] = values_torch.numpy()

            if field == "cs_tensor_spherical":
                mol["cs_iso_std"] = values_torch.numpy()[:, 0:1]

def create_and_save_masked_npz(molecules, output_path: Path, statistics: dict):
    """Creates and saves masked/standardized arrays and statistics to NPZ."""
    if not molecules:
        print("No data to save. Aborting.")
        return

    print("\n--- Merging and Saving Standardized Datasets ---")
    total_mols = len(molecules)
    max_n_atoms = max(m['num_atoms'] for m in molecules)
    print(f"Processing {total_mols} structures, padding up to {max_n_atoms} atoms.")

    all_keys = set(key for mol in molecules for key in mol.keys())
    atom_level_keys = set()
    graph_level_keys = set()

    expected_atom_keys = {'pos', 'cs_tensor_spherical', 'cs_iso', 'center_atoms_mask',
                          'forces', 'atom_types', 'atom_rows', 'atom_cols',
                          'cs_tensor_spherical_std', 'cs_iso_std'} # Add std keys

    for key in all_keys:
        if key in expected_atom_keys:
            atom_level_keys.add(key)
        elif key != 'num_atoms':
            val = molecules[0].get(key)
            if isinstance(val, np.ndarray) and len(val.shape) > 0 and val.shape[0] == molecules[0]['num_atoms']:
                 atom_level_keys.add(key)
            else:
                 graph_level_keys.add(key)

    # --- Load template NPZ file ---
    # The template file should be placed where the application can access it.
    # Here, we assume it's in the same directory as this script.
    template_npz_path = Path(__file__).parent / "template.npz"
    save_dict = {}
    if template_npz_path.exists():
        print(f"Loading data from template file: {template_npz_path}")
        with np.load(template_npz_path) as template_data:
            for key in template_data.keys():
                save_dict[key] = template_data[key]
        print(f"Loaded keys from template: {list(save_dict.keys())}")
    else:
        print(f"Warning: Template file not found at {template_npz_path}. Proceeding without it.")
    # --- End of template loading ---

    for key in sorted(list(atom_level_keys)):
        print(f"Processing atom-level property: {key}")
        first_mol_prop = next((m[key] for m in molecules if key in m), None)
        if first_mol_prop is None:
             print(f"  -> Key '{key}' not found in any molecule, skipping.")
             continue

        if first_mol_prop.ndim > 1:
            shape = (total_mols, max_n_atoms, first_mol_prop.shape[1])
        else:
            shape = (total_mols, max_n_atoms)

        masked_array = np.ma.masked_all(shape, dtype=first_mol_prop.dtype)
        masked_array.data[...] = 0 

        for i, mol_data in enumerate(molecules):
            if key in mol_data:
                n_atoms = mol_data['num_atoms']
                data = mol_data[key]
                if data.ndim > 1:
                    masked_array.data[i, :n_atoms, :] = data
                    masked_array.mask[i, :n_atoms, :] = False
                else:
                    masked_array.data[i, :n_atoms] = data
                    masked_array.mask[i, :n_atoms] = False

        save_dict[key] = masked_array.data
        save_dict[f"{key}__mask__"] = masked_array.mask

    for key in sorted(list(graph_level_keys)):
        print(f"Processing graph-level property: {key}")
        data_list = [mol.get(key) for mol in molecules]

        if all(item is None for item in data_list):
            print(f"  -> Key '{key}' is None for all molecules, skipping.")
            continue

        try:
            if any(isinstance(item, np.ndarray) for item in data_list):
                first_valid_item = next(item for item in data_list if isinstance(item, np.ndarray))
                default_shape = first_valid_item.shape
                default_dtype = first_valid_item.dtype
                
                if np.issubdtype(default_dtype, np.floating):
                     default_val = np.full(default_shape, np.nan, dtype=default_dtype)
                elif np.issubdtype(default_dtype, np.integer):
                     default_val = np.full(default_shape, 0, dtype=default_dtype)
                else:
                     default_val = np.full(default_shape, None, dtype=object)

                processed_list = [item if isinstance(item, np.ndarray) else default_val for item in data_list]

                if all(arr.shape == default_shape for arr in processed_list if isinstance(arr,np.ndarray)):
                     stacked_array = np.stack(processed_list, axis=0)
                     save_dict[key] = stacked_array
                else:
                     print(f"Warning: Cannot stack '{key}' due to inconsistent shapes. Saving as object array.")
                     save_dict[key] = np.array(data_list, dtype=object)

            else: # Not arrays
                save_dict[key] = np.array(data_list, dtype=object)
                try: save_dict[key] = save_dict[key].astype(np.float32)
                except (ValueError, TypeError): pass 

            if key == 'Lattice' and key in save_dict and isinstance(save_dict[key], np.ndarray) and save_dict[key].shape == (total_mols, 9):
                save_dict[key] = save_dict[key].reshape(total_mols, 3, 3)
                print(f"  -> Reshaped 'Lattice' to {save_dict[key].shape}")

        except Exception as e:
            print(f"Warning: Could not collate graph-level property '{key}'. Reason: {e}. Saving as object array.")
            save_dict[key] = np.array(data_list, dtype=object)

    print("Adding standardization statistics to NPZ file.")
    for key, value in statistics.items():
        save_dict[key] = np.asarray(value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"\n✅ Dataset saved successfully to: {output_path}")
    print(f"Saved fields: {list(save_dict.keys())}")


# --- Main Wrapper Function ---

def process_uploaded_file(input_path: Path, output_dir: Path, trajectory_path: Path = None):
    """
    Main function to process an uploaded file, route to the correct parser,
    and save the output.
    """
    output_filename = input_path.stem + ".npz"
    output_path = output_dir / output_filename
    
    file_extension = input_path.suffix.lower()
    
    all_molecules_data = []
    statistics = {}

    if file_extension in [".pdb", ".gro", ".xyz"]:
        # Full processing for XYZ files (with ground truth)
        all_molecules_data = parse_structure_file_mdanalysis(input_path, trajectory_path)
        if not all_molecules_data:
            raise ValueError(f"Failed to parse any molecules from {file_extension.upper()} file.")
        
        # Only compute stats if cs_tensor was present
        if any("cs_tensor_spherical" in mol for mol in all_molecules_data):
            statistics = compute_statistics(all_molecules_data, STANDARDIZATION_CONFIG)
            standardize_data(all_molecules_data, statistics, STANDARDIZATION_CONFIG)
        else:
            print("No 'cs_tensor' found in XYZ, skipping statistics and standardization.")

    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Please upload .xyz, .pdb, or .gro")

    # Save whatever was processed
    create_and_save_masked_npz(all_molecules_data, output_path, statistics)
    
    return output_path, len(all_molecules_data)
