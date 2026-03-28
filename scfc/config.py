"""Paths, atlas definitions, and project configuration."""

from pathlib import Path
from dataclasses import dataclass, field

# ── Base paths ──────────────────────────────────────────────────────────
NEURO_ROOT = Path.home() / "Documents" / "neuroscience"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
MATRICES_DIR = OUTPUTS / "matrices"
FIGURES_DIR = OUTPUTS / "figures"
REGISTRATION_DIR = OUTPUTS / "registration"

# ── Data paths ──────────────────────────────────────────────────────────
ATLAS_DIR = NEURO_ROOT / "atlases"
MRTRIX_DIR = NEURO_ROOT / "datasets" / "mrtrix3_output"
NILEARN_DIR = NEURO_ROOT / "datasets" / "nilearn"

TRACTOGRAM = MRTRIX_DIR / "tracks_1M.tck"
SIFT2_WEIGHTS = MRTRIX_DIR / "sift2_weights.txt"
B0_MEAN = MRTRIX_DIR / "b0_mean.mif"
BRAIN_MASK = MRTRIX_DIR / "brain_mask.mif"

MNI_TEMPLATE = ATLAS_DIR / "MNI" / "MNI152_T1_2mm_brain.nii.gz"

ADHD_DATA = NILEARN_DIR / "adhd" / "data"


# ── Atlas registry ──────────────────────────────────────────────────────
@dataclass
class AtlasDef:
    name: str
    path: Path
    n_parcels: int
    networks: dict[int, str] = field(default_factory=dict)
    description: str = ""


# Schaefer 7-network label mapping (parse from atlas integer labels)
SCHAEFER_7_NETWORKS = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Salience/Ventral Attention",
    "Limbic": "Limbic",
    "Cont": "Frontoparietal Control",
    "Default": "Default Mode",
}

ATLASES: dict[str, AtlasDef] = {
    "schaefer100": AtlasDef(
        name="Schaefer 100 (7 Networks)",
        path=ATLAS_DIR / "Schaefer" / "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
        n_parcels=100,
        description="100 cortical parcels, Yeo 7 networks",
    ),
    "schaefer200": AtlasDef(
        name="Schaefer 200 (7 Networks)",
        path=ATLAS_DIR / "Schaefer" / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
        n_parcels=200,
        description="200 cortical parcels, Yeo 7 networks",
    ),
    "schaefer400": AtlasDef(
        name="Schaefer 400 (7 Networks)",
        path=ATLAS_DIR / "Schaefer" / "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
        n_parcels=400,
        description="400 cortical parcels, Yeo 7 networks",
    ),
    "aal": AtlasDef(
        name="AAL",
        path=ATLAS_DIR / "AAL" / "AAL_MNI152_2mm.nii.gz",
        n_parcels=116,
        description="Automated Anatomical Labeling, 116 regions",
    ),
}

# Active atlases for analysis (can be overridden)
ACTIVE_ATLASES = ["schaefer100", "schaefer200", "schaefer400", "aal"]


def get_atlas(key: str) -> AtlasDef:
    """Get an atlas definition by key."""
    if key not in ATLASES:
        raise KeyError(f"Unknown atlas '{key}'. Available: {list(ATLASES)}")
    return ATLASES[key]


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [MATRICES_DIR, FIGURES_DIR, REGISTRATION_DIR]:
        d.mkdir(parents=True, exist_ok=True)
