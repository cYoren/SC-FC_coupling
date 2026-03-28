"""Cross-parcellation robustness: compare SC-FC coupling across atlas granularities."""

import logging
from dataclasses import dataclass

import numpy as np
import nibabel as nib
from scipy import stats
from scipy.ndimage import label as ndimage_label

from .coupling import CouplingResult, analyze_coupling

log = logging.getLogger(__name__)


@dataclass
class RobustnessResult:
    """Cross-parcellation robustness summary."""
    coupling_results: dict[str, CouplingResult]
    global_r_values: dict[str, float]
    global_p_values: dict[str, float]

    # Spatial overlap analysis (Schaefer only — hierarchical nesting)
    voxelwise_coupling: dict[str, np.ndarray] | None  # atlas_key → 3D volume
    consistency_map: np.ndarray | None                  # mean across atlases
    variability_map: np.ndarray | None                  # std across atlases

    # Network-level summary (Schaefer only)
    network_coupling: dict[str, dict[str, float]] | None  # atlas → {network: mean_r}


def _parse_schaefer_network_labels(atlas_path, n_parcels: int) -> dict[int, str]:
    """
    Assign Yeo 7-network labels to Schaefer parcels using the freeview LUT.

    LUT format: idx  7Networks_LH_Vis_1  R  G  B  0
    Network is the 3rd underscore-separated field in the label name.
    """
    # Map short LUT names → display names
    net_name_map = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "DorsAttn",
        "SalVentAttn": "SalVentAttn",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    lut_dir = atlas_path.parent / "freeview_lut"
    lut_file = None
    if lut_dir.exists():
        # Match e.g. Schaefer2018_100Parcels_7Networks_order.txt
        candidates = [f for f in lut_dir.glob(f"*_{n_parcels}Parcels_7Networks_order.txt")]
        if candidates:
            lut_file = candidates[0]

    labels = {}
    if lut_file and lut_file.exists():
        with open(lut_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    idx = int(parts[0])
                    name = parts[1]
                    # e.g. "7Networks_LH_Vis_1" → split by _ → ['7Networks','LH','Vis','1']
                    name_parts = name.split("_")
                    if len(name_parts) >= 3:
                        net_short = name_parts[2]
                        labels[idx] = net_name_map.get(net_short, net_short)
                    else:
                        labels[idx] = "Unknown"

    return labels


def parcels_to_voxelwise(
    regional_values: np.ndarray,
    atlas_path,
) -> np.ndarray:
    """Map parcel-level values back to voxel-space (3D volume)."""
    img = nib.load(str(atlas_path))
    data = np.asarray(img.dataobj, dtype=np.int32)
    out = np.full(data.shape, np.nan, dtype=np.float64)

    for i, val in enumerate(regional_values):
        parcel_id = i + 1  # parcels are 1-indexed
        mask = data == parcel_id
        if mask.any():
            out[mask] = val

    return out


def compute_network_coupling(
    coupling_result: CouplingResult,
    atlas_path,
    n_parcels: int,
) -> dict[str, float]:
    """Compute mean regional coupling per Yeo network."""
    labels = _parse_schaefer_network_labels(atlas_path, n_parcels)
    if not labels:
        return {}

    rc = coupling_result.regional_coupling
    network_means = {}
    for net in set(labels.values()):
        parcel_ids = [pid for pid, n in labels.items() if n == net]
        # Convert 1-indexed parcel IDs to 0-indexed array indices
        indices = [pid - 1 for pid in parcel_ids if pid - 1 < len(rc)]
        vals = rc[indices]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            network_means[net] = float(np.mean(vals))

    return network_means


def analyze_robustness(
    sc_matrices: dict[str, np.ndarray],
    fc_matrices: dict[str, np.ndarray],
    n_perm: int = 1000,
) -> RobustnessResult:
    """
    Run coupling analysis across all atlases and compare.

    Builds:
    - Per-atlas global and regional coupling
    - Voxel-wise consistency maps (how stable is coupling across parcellations)
    - Network-level coupling summary
    """
    from .config import ATLASES

    common_keys = sorted(set(sc_matrices) & set(fc_matrices))
    log.info("Robustness analysis across %d atlases: %s", len(common_keys), common_keys)

    coupling_results = {}
    global_r = {}
    global_p = {}
    voxelwise = {}
    network_coupling = {}

    for key in common_keys:
        result = analyze_coupling(sc_matrices[key], fc_matrices[key], key, n_perm=n_perm)
        coupling_results[key] = result
        global_r[key] = result.global_r
        global_p[key] = result.global_p

        atlas = ATLASES[key]

        # Voxel-wise coupling map
        vox = parcels_to_voxelwise(result.regional_coupling, atlas.path)
        voxelwise[key] = vox

        # Network coupling (Schaefer only)
        if "schaefer" in key:
            net_coup = compute_network_coupling(result, atlas.path, atlas.n_parcels)
            network_coupling[key] = net_coup

    # Consistency and variability maps (only for atlases with same voxel grid)
    consistency_map = None
    variability_map = None
    if len(voxelwise) >= 2:
        # Stack all voxelwise maps — they should share the MNI grid
        shapes = [v.shape for v in voxelwise.values()]
        common_shape = shapes[0]
        compatible = [k for k, v in voxelwise.items() if v.shape == common_shape]

        if len(compatible) >= 2:
            stack = np.stack([voxelwise[k] for k in compatible], axis=0)
            with np.errstate(all="ignore"):
                consistency_map = np.nanmean(stack, axis=0)
                variability_map = np.nanstd(stack, axis=0)

    return RobustnessResult(
        coupling_results=coupling_results,
        global_r_values=global_r,
        global_p_values=global_p,
        voxelwise_coupling=voxelwise,
        consistency_map=consistency_map,
        variability_map=variability_map,
        network_coupling=network_coupling if network_coupling else None,
    )
