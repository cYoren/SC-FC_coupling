"""Functional connectivity: parcellated time-series extraction and FC computation."""

import logging
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

from .config import (
    ADHD_DATA, NILEARN_DIR, ATLASES, ACTIVE_ATLASES,
    MATRICES_DIR, AtlasDef, ensure_dirs,
)

log = logging.getLogger(__name__)


def _find_fmri_files() -> list[Path]:
    """Find available resting-state fMRI files from ADHD200 dataset."""
    files = []
    if ADHD_DATA.exists():
        for subj_dir in sorted(ADHD_DATA.iterdir()):
            if subj_dir.is_dir():
                for f in subj_dir.glob("*rest*mni*.nii.gz"):
                    files.append(f)
    # Also check development_fmri
    dev_dir = NILEARN_DIR / "development_fmri" / "development_fmri"
    if dev_dir.exists():
        for f in dev_dir.glob("*bold*.nii.gz"):
            files.append(f)
    return files


def extract_timeseries(
    fmri_path: Path,
    atlas: AtlasDef,
    confounds_path: Path | None = None,
    standardize: str = "zscore_sample",
    low_pass: float = 0.1,
    high_pass: float = 0.01,
    t_r: float | None = None,
) -> np.ndarray:
    """
    Extract parcellated time series from fMRI using a given atlas.

    Returns array of shape (n_timepoints, n_parcels).
    """
    masker = NiftiLabelsMasker(
        labels_img=str(atlas.path),
        standardize=standardize,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        resampling_target="data",
    )
    confounds = None
    if confounds_path and confounds_path.exists():
        import pandas as pd
        # Try tab-separated first (ADHD200 format), then comma
        confounds = pd.read_csv(confounds_path, sep="\t")
        if confounds.shape[1] <= 1:
            confounds = pd.read_csv(confounds_path, sep=",")

    ts = masker.fit_transform(str(fmri_path), confounds=confounds)
    log.info("Extracted timeseries: %s → shape %s", atlas.name, ts.shape)
    return ts


def compute_fc_matrix(
    timeseries: np.ndarray,
    kind: str = "correlation",
) -> np.ndarray:
    """
    Compute functional connectivity from parcellated time series.

    kind: 'correlation', 'partial correlation', 'tangent', 'covariance'
    """
    conn = ConnectivityMeasure(kind=kind)
    fc = conn.fit_transform([timeseries])[0]
    return fc


def compute_fc_for_atlas(
    atlas_key: str,
    fmri_files: list[Path] | None = None,
    t_r: float = 2.0,
    force: bool = False,
) -> np.ndarray:
    """
    Compute FC matrix for a given atlas, averaging across available subjects.

    Returns FC matrix (n_parcels x n_parcels).
    """
    ensure_dirs()
    atlas = ATLASES[atlas_key]
    fc_path = MATRICES_DIR / f"fc_{atlas_key}.csv"

    if fc_path.exists() and not force:
        log.info("FC matrix exists: %s", fc_path)
        return np.loadtxt(fc_path, delimiter=",")

    if fmri_files is None:
        fmri_files = _find_fmri_files()

    if not fmri_files:
        raise FileNotFoundError("No resting-state fMRI files found")

    fc_matrices = []
    for fmri_path in fmri_files:
        log.info("Processing %s with %s...", fmri_path.name, atlas.name)

        # Detect confounds file
        confounds = None
        stem = fmri_path.stem.replace(".nii", "")
        for pattern in [f"{stem}_confounds*", "*regressors*"]:
            candidates = list(fmri_path.parent.glob(pattern))
            if candidates:
                confounds = candidates[0]
                break

        ts = extract_timeseries(
            fmri_path, atlas, confounds_path=confounds, t_r=t_r,
        )
        fc = compute_fc_matrix(ts)
        fc_matrices.append(fc)

    # Average FC across subjects
    fc_avg = np.mean(fc_matrices, axis=0)
    np.savetxt(fc_path, fc_avg, delimiter=",", fmt="%.6f")
    log.info("FC matrix computed: %s, shape=%s", atlas_key, fc_avg.shape)
    return fc_avg


def compute_all_fc(
    atlas_keys: list[str] | None = None,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Compute FC matrices for all active atlases."""
    keys = atlas_keys or ACTIVE_ATLASES
    fmri_files = _find_fmri_files()
    log.info("Found %d fMRI files", len(fmri_files))

    results = {}
    for key in keys:
        results[key] = compute_fc_for_atlas(key, fmri_files=fmri_files, force=force)
    return results
