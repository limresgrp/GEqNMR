import math
import ase.io
import numpy as np
import torch

def convert_spherical_to_cartesian(spherical_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of tensors from a spherical representation (l=0, l=1, l=2)
    to a 3x3 Cartesian tensor representation, flattened.

    The conversion follows the direct summation of isotropic, antisymmetric,
    and symmetric-traceless parts.

    Args:
        spherical_tensor: A tensor of shape (n_batch, 9) where:
                          - column 0: scalar part (l=0)
                          - columns 1-3: vector part for l=1 (vx, vy, vz)
                          - columns 4-8: 5 components for l=2

    Returns:
        A tensor of shape (n_batch, 9) representing the flattened 3x3 matrix.
    """
    if spherical_tensor.shape[1] != 9:
        raise ValueError("Input tensor must have 9 columns (1+3+5 components).")

    n_batch = spherical_tensor.shape[0]
    device = spherical_tensor.device
    dtype = spherical_tensor.dtype # Preserve the dtype (e.g., float64)

    # --- 1. Extract Components ---
    # Isotropic part (l=0)
    s_iso = spherical_tensor[:, 0]

    # Antisymmetric part (l=1)
    v_x, v_y, v_z = spherical_tensor[:, 1], spherical_tensor[:, 2], spherical_tensor[:, 3]

    # Symmetric-traceless part (l=2)
    c1, c2, c3, c4, c5 = spherical_tensor[:, 4], spherical_tensor[:, 5], spherical_tensor[:, 6], spherical_tensor[:, 7], spherical_tensor[:, 8]

    # --- 2. Build the Cartesian Tensor ---
    # Initialize an empty (n_batch, 3, 3) tensor
    cartesian_tensor = torch.zeros((n_batch, 3, 3), device=device, dtype=dtype)
    
    # Pre-calculate the constant for the l=2 part
    sqrt6 = math.sqrt(6)

    # Reconstruct the matrix by adding the components for each element
    # T_final = T_iso + T_anti + T_sym_traceless

    # Row 1
    cartesian_tensor[:, 0, 0] = s_iso + c1 - c2 / sqrt6
    cartesian_tensor[:, 0, 1] = -v_z + c3
    cartesian_tensor[:, 0, 2] = v_y + c4

    # Row 2
    cartesian_tensor[:, 1, 0] = v_z + c3
    cartesian_tensor[:, 1, 1] = s_iso - c1 - c2 / sqrt6
    cartesian_tensor[:, 1, 2] = -v_x + c5

    # Row 3
    cartesian_tensor[:, 2, 0] = -v_y + c4
    cartesian_tensor[:, 2, 1] = v_x + c5
    cartesian_tensor[:, 2, 2] = s_iso + 2 * c2 / sqrt6

    # --- 3. Flatten the Tensor ---
    # Reshape from (n_batch, 3, 3) to (n_batch, 9)
    return cartesian_tensor.reshape(n_batch, 9)


def convert_cartesian_to_spherical(cartesian_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of 3x3 Cartesian tensors to a spherical representation (l=0, l=1, l=2),
    flattened into a 9-component vector.

    The conversion follows the decomposition into isotropic, antisymmetric,
    and symmetric-traceless parts.

    Args:
        cartesian_tensor: A tensor of shape (n_batch, 3, 3) representing the 3x3 matrices.

    Returns:
        A tensor of shape (n_batch, 9) where:
                          - column 0: scalar part (l=0)
                          - columns 1-3: vector part for l=1 (vx, vy, vz)
                          - columns 4-8: 5 components for l=2 (c1, c2, c3, c4, c5)
    """
    if cartesian_tensor.shape[1:] != (3, 3):
        raise ValueError("Input tensor must be of shape (n_batch, 3, 3).")

    n_batch = cartesian_tensor.shape[0]
    device = cartesian_tensor.device
    dtype = cartesian_tensor.dtype # Preserve the dtype (e.g., float64)
    sqrt6 = math.sqrt(6)

    # --- 1. Isotropic part (l=0) ---
    # s_iso = (T_xx + T_yy + T_zz) / 3
    s_iso = torch.diagonal(cartesian_tensor, dim1=-2, dim2=-1).sum(dim=-1) / 3.0

    # --- 2. Antisymmetric part (l=1) ---
    # Calculate the antisymmetric part explicitly: A = (T - T^T) / 2
    antisymmetric_part = (cartesian_tensor - cartesian_tensor.transpose(-2, -1)) / 2.0
    v_x = antisymmetric_part[:, 2, 1] # T_anti_zy
    v_y = antisymmetric_part[:, 0, 2] # T_anti_xz
    v_z = antisymmetric_part[:, 1, 0] # T_anti_yx

    # --- 3. Symmetric part (S = (T + T^T) / 2) ---
    symmetric_part = (cartesian_tensor + cartesian_tensor.transpose(-2, -1)) / 2.0

    # Derive c1, c2, c3, c4, c5 from symmetric_part and s_iso
    c1 = (symmetric_part[:, 0, 0] - symmetric_part[:, 1, 1]) / 2.0
    c2 = (sqrt6 / 2.0) * (symmetric_part[:, 2, 2] - s_iso)
    c3 = symmetric_part[:, 0, 1]
    c4 = symmetric_part[:, 0, 2]
    c5 = symmetric_part[:, 1, 2]

    # Combine all components into a (n_batch, 9) tensor
    spherical_tensor_components = torch.stack(
        [s_iso, v_x, v_y, v_z, c1, c2, c3, c4, c5], dim=-1
    )

    return spherical_tensor_components.to(dtype=dtype)


def test_spherical_cartesian_conversion(input_cartesian_tensors: torch.Tensor):
    """
    Tests the correctness of convert_spherical_to_cartesian and
    convert_cartesian_to_spherical by round-tripping and checking decomposition.
    This test is designed to run with float64 precision for rigor.

    Args:
        input_cartesian_tensors: A batch of 3x3 Cartesian tensors to use for testing.
                                 Expected to be torch.float64.
    """
    print("\n--- Testing Spherical-Cartesian Conversion (float64) ---")
    
    # Define a strict tolerance for float64 numerical precision
    STRICT_ATOL = 1e-9

    if input_cartesian_tensors.dtype != torch.float64:
        print(f"Warning: Input tensor is not float64. Forcing to float64 for rigorous test.")
        input_cartesian_tensors = input_cartesian_tensors.to(torch.float64)

    # Step 1: Convert Cartesian to Spherical
    spherical_components = convert_cartesian_to_spherical(input_cartesian_tensors)
    print(f"Original Cartesian tensor shape: {input_cartesian_tensors.shape}, dtype: {input_cartesian_tensors.dtype}")
    print(f"Spherical components shape: {spherical_components.shape}, dtype: {spherical_components.dtype}")

    # Step 2: Convert Spherical back to Cartesian (flattened)
    reconstructed_cartesian_flat = convert_spherical_to_cartesian(spherical_components)
    # Reshape back to (N, 3, 3) for direct comparison
    reconstructed_cartesian_tensors = reconstructed_cartesian_flat.reshape(-1, 3, 3)
    print(f"Reconstructed Cartesian tensor shape: {reconstructed_cartesian_tensors.shape}, dtype: {reconstructed_cartesian_tensors.dtype}")

    # Step 3: Verify round-trip (original vs reconstructed Cartesian)
    # We compare the (N, 3, 3) tensors directly.
    assert torch.allclose(input_cartesian_tensors, reconstructed_cartesian_tensors, atol=STRICT_ATOL), \
        "Round-trip conversion failed: Original and reconstructed Cartesian tensors do not match."
    print(f"✅ Round-trip conversion (Cartesian -> Spherical -> Cartesian) successful (atol={STRICT_ATOL}).")

    # Step 4: Verify decomposition components (trace, antisymmetric, symmetric-traceless)

    # Check trace: Tr(T) = 3 * s_iso
    original_traces = torch.diagonal(input_cartesian_tensors, dim1=-2, dim2=-1).sum(dim=-1)
    # Ensure s_iso is float64 (it should be from the conversion function)
    s_iso_from_spherical = spherical_components[:, 0] 
    assert torch.allclose(original_traces, 3 * s_iso_from_spherical, atol=STRICT_ATOL), \
        "Trace verification failed: Tr(T) != 3 * s_iso."
    print(f"✅ Trace verification successful: Tr(T) == 3 * s_iso (atol={STRICT_ATOL}).")

    # Check antisymmetric part:
    original_antisymmetric_part = (input_cartesian_tensors - input_cartesian_tensors.transpose(-2, -1)) / 2.0
    # Use the (N, 3, 3) reconstructed tensor
    reconstructed_antisymmetric_part = (reconstructed_cartesian_tensors - reconstructed_cartesian_tensors.transpose(-2, -1)) / 2.0
    assert torch.allclose(original_antisymmetric_part, reconstructed_antisymmetric_part, atol=STRICT_ATOL), \
        "Antisymmetric part verification failed."
    print(f"✅ Antisymmetric part verification successful (atol={STRICT_ATOL}).")

    # Check symmetric-traceless part:
    # Ensure torch.eye is created with the correct dtype
    eye = torch.eye(3, device=input_cartesian_tensors.device, dtype=torch.float64)
    original_symmetric_traceless_part = (input_cartesian_tensors + input_cartesian_tensors.transpose(-2, -1)) / 2.0 - (original_traces / 3.0).unsqueeze(-1).unsqueeze(-1) * eye
    
    # Use the (N, 3, 3) reconstructed tensor
    reconstructed_symmetric_traceless_part = (reconstructed_cartesian_tensors + reconstructed_cartesian_tensors.transpose(-2, -1)) / 2.0 - (s_iso_from_spherical).unsqueeze(-1).unsqueeze(-1) * eye
    assert torch.allclose(original_symmetric_traceless_part, reconstructed_symmetric_traceless_part, atol=STRICT_ATOL), \
        "Symmetric-traceless part verification failed."
    print(f"✅ Symmetric-traceless part verification successful (atol={STRICT_ATOL}).")

    print("\n✅ All tests passed with float64 precision.")

