"""Structural connectivity: registration + tractogram-based connectome construction."""

import subprocess
import logging
from pathlib import Path

import numpy as np
import nibabel as nib

from .config import (
    MRTRIX_DIR, B0_MEAN, TRACTOGRAM, SIFT2_WEIGHTS, MNI_TEMPLATE,
    REGISTRATION_DIR, MATRICES_DIR, ATLASES, ACTIVE_ATLASES,
    AtlasDef, ensure_dirs,
)

log = logging.getLogger(__name__)

# FSL binary directory — use binaries directly (wrapper scripts may be broken)
_FSL_BIN = Path.home() / "fsl" / "bin"


def _fsl(tool: str) -> str:
    """Return path to FSL binary, falling back to PATH."""
    direct = _FSL_BIN / tool
    if direct.exists():
        return str(direct)
    return tool


def _run(cmd: list[str], desc: str = "") -> subprocess.CompletedProcess:
    """Run a shell command, raise on failure."""
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{desc or cmd[0]} failed:\n{result.stderr}")
    return result


# ── Step 1: Convert b0 to NIfTI ────────────────────────────────────────

def convert_b0_to_nifti(force: bool = False) -> Path:
    """Convert b0_mean.mif → single 3D NIfTI for registration."""
    ensure_dirs()
    b0_nii = REGISTRATION_DIR / "b0_mean_3d.nii.gz"
    if b0_nii.exists() and not force:
        log.info("b0 NIfTI already exists: %s", b0_nii)
        return b0_nii

    # b0_mean.mif has multiple b0 volumes — average to single 3D
    b0_avg = REGISTRATION_DIR / "b0_avg.mif"
    _run(["mrmath", str(B0_MEAN), "mean", str(b0_avg), "-axis", "3", "-force"],
         desc="mrmath average b0")
    _run(["mrconvert", str(b0_avg), str(b0_nii), "-force"],
         desc="mrconvert b0→nifti")
    return b0_nii


# ── Step 2: Register b0 → MNI (get transform) ──────────────────────────

def register_b0_to_mni(force: bool = False) -> tuple[Path, Path]:
    """
    Use FSL flirt to register b0 → MNI152.
    Returns (b0_to_mni_mat, mni_to_b0_mat).
    """
    ensure_dirs()
    b0_nii = convert_b0_to_nifti(force=force)

    b0_to_mni_mat = REGISTRATION_DIR / "b0_to_mni.mat"
    mni_to_b0_mat = REGISTRATION_DIR / "mni_to_b0.mat"

    if b0_to_mni_mat.exists() and mni_to_b0_mat.exists() and not force:
        log.info("Registration matrices already exist")
        return b0_to_mni_mat, mni_to_b0_mat

    # flirt: b0 → MNI
    _run([
        _fsl("flirt"),
        "-in", str(b0_nii),
        "-ref", str(MNI_TEMPLATE),
        "-omat", str(b0_to_mni_mat),
        "-dof", "12",
        "-cost", "mutualinfo",
    ], desc="flirt b0→MNI")

    # Invert: MNI → b0
    _run([
        _fsl("convert_xfm"),
        "-omat", str(mni_to_b0_mat),
        "-inverse", str(b0_to_mni_mat),
    ], desc="invert registration matrix")

    log.info("Registration complete: %s, %s", b0_to_mni_mat, mni_to_b0_mat)
    return b0_to_mni_mat, mni_to_b0_mat


# ── Step 3: Warp atlas to DWI space ────────────────────────────────────

def warp_atlas_to_dwi(atlas: AtlasDef, force: bool = False) -> Path:
    """Apply MNI→b0 transform to bring atlas into DWI space."""
    ensure_dirs()
    b0_nii = convert_b0_to_nifti()
    _, mni_to_b0_mat = register_b0_to_mni()

    atlas_dwi = REGISTRATION_DIR / f"{atlas.path.stem}_dwi.nii.gz"
    if atlas_dwi.exists() and not force:
        log.info("Warped atlas exists: %s", atlas_dwi)
        return atlas_dwi

    _run([
        _fsl("flirt"),
        "-in", str(atlas.path),
        "-ref", str(b0_nii),
        "-applyxfm",
        "-init", str(mni_to_b0_mat),
        "-interp", "nearestneighbour",
        "-out", str(atlas_dwi),
    ], desc=f"warp {atlas.name} → DWI space")

    log.info("Warped atlas: %s", atlas_dwi)
    return atlas_dwi


# ── Step 4: Compute SC matrix via tck2connectome ───────────────────────

def compute_sc_matrix(
    atlas_key: str,
    use_sift2: bool = True,
    force: bool = False,
) -> np.ndarray:
    """
    Compute structural connectivity matrix for a given atlas.

    Uses tck2connectome with optional SIFT2 weights.
    Returns symmetric SC matrix (n_parcels x n_parcels).
    """
    ensure_dirs()
    atlas = ATLASES[atlas_key]
    atlas_dwi = warp_atlas_to_dwi(atlas, force=force)

    suffix = "_sift2" if use_sift2 else ""
    sc_path = MATRICES_DIR / f"sc_{atlas_key}{suffix}.csv"

    if sc_path.exists() and not force:
        log.info("SC matrix exists: %s", sc_path)
        return np.loadtxt(sc_path, delimiter=",")

    cmd = [
        "tck2connectome",
        str(TRACTOGRAM),
        str(atlas_dwi),
        str(sc_path),
        "-symmetric",
        "-zero_diagonal",
        "-force",
    ]
    if use_sift2 and SIFT2_WEIGHTS.exists():
        cmd.extend(["-tck_weights_in", str(SIFT2_WEIGHTS)])

    _run(cmd, desc=f"tck2connectome {atlas_key}")

    # tck2connectome outputs comma-separated; load accordingly
    sc = np.loadtxt(sc_path, delimiter=",")
    log.info("SC matrix computed: %s, shape=%s", atlas_key, sc.shape)
    return sc


def compute_all_sc(
    atlas_keys: list[str] | None = None,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Compute SC matrices for all active atlases."""
    keys = atlas_keys or ACTIVE_ATLASES
    results = {}
    for key in keys:
        log.info("Computing SC for %s...", key)
        results[key] = compute_sc_matrix(key, force=force)
    return results
