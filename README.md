# SC-FC Coupling Atlas

**Cross-parcellation structure-function coupling robustness analysis for the human brain.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

Every SC-FC coupling study picks **one** brain parcellation and reports results as if they are atlas-independent. They are not. The parcellation dependence of structure-function findings is a well-known methodological concern ([Zalesky et al., 2010](https://doi.org/10.1016/j.neuroimage.2009.12.036); [de Reus & van den Heuvel, 2013](https://doi.org/10.1016/j.neuroimage.2013.09.022)), yet **no dedicated tool** exists to systematically quantify this.

SC-FC Coupling Atlas fills that gap: it computes structure-function coupling across multiple parcellation schemes simultaneously, identifies which brain regions show **robust** vs. **atlas-dependent** coupling, and annotates the results by canonical brain networks.

## What This Tool Does

1. **Structural Connectivity (SC)** -- registers MNI atlases to DWI space via FSL FLIRT, constructs connectomes from MRtrix3 tractograms with SIFT2 weighting
2. **Functional Connectivity (FC)** -- extracts parcellated time series from resting-state fMRI using nilearn, computes correlation-based FC matrices
3. **SC-FC Coupling Analysis** -- global edge-wise coupling (log-SC vs FC Pearson correlation) and regional per-node coupling, with permutation-based significance testing
4. **Cross-Parcellation Robustness** -- runs the analysis across Schaefer 100/200/400 and AAL simultaneously, produces voxel-wise consistency and variability brain maps
5. **Network Annotation** -- maps regional coupling to Yeo 7-network labels using official Schaefer LUTs
6. **Publication-Ready Figures** -- SC/FC matrix heatmaps, edge scatter plots, axial brain maps, network bar plots, multi-panel summary dashboard

## Example Results

Results from a single-subject demonstration (OpenNeuro ds000114 DWI + ADHD200 resting-state fMRI):

| Atlas | Global SC-FC r | p (perm) | SC Density | Valid Regions |
|-------|---------------|----------|------------|---------------|
| AAL (116 regions) | 0.412 | < 0.001 | 45.2% | 116/116 |
| Schaefer-100 | 0.411 | < 0.001 | 49.2% | 100/100 |
| Schaefer-200 | 0.416 | < 0.001 | 30.8% | 200/200 |
| Schaefer-400 | 0.410 | < 0.001 | 16.5% | 400/400 |

**Key finding:** Global SC-FC coupling is remarkably stable across parcellations (r ~ 0.41), but regional coupling varies by network -- Visual and Somatomotor networks show highest coupling, Default Mode Network shows lowest. This is consistent with the cortical hierarchy hypothesis (Margulies et al., 2016).

## Quick Start

### Installation

```bash
git clone https://github.com/cYoren/SC-FC_coupling.git
cd SC-FC_coupling
pip install -e .
```

### Running the Pipeline

```bash
# Full pipeline
python scripts/run_all.py

# With options
python scripts/run_all.py --atlases schaefer100 schaefer200 --n-perm 5000 --force

# Use cached matrices, just redo analysis + plots
python scripts/run_all.py --skip-sc --skip-fc
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--atlases` | Atlas keys to analyze (default: `schaefer100 schaefer200 schaefer400 aal`) |
| `--n-perm` | Number of permutations for significance testing (default: 1000) |
| `--force` | Recompute even if cached results exist |
| `--skip-sc` | Skip SC computation, use cached matrices |
| `--skip-fc` | Skip FC computation, use cached matrices |
| `--skip-plots` | Skip figure generation |
| `-v` | Verbose logging |

## Python API

```python
from scfc.structural import compute_all_sc
from scfc.functional import compute_all_fc
from scfc.robustness import analyze_robustness

# Compute connectivity matrices across parcellations
sc = compute_all_sc(["schaefer100", "schaefer200", "schaefer400"])
fc = compute_all_fc(["schaefer100", "schaefer200", "schaefer400"])

# Run cross-parcellation robustness analysis
results = analyze_robustness(sc, fc, n_perm=1000)

# Access results
results.global_r_values           # {atlas_key: r}
results.global_p_values           # {atlas_key: p}
results.coupling_results          # {atlas_key: CouplingResult}
results.network_coupling          # {atlas_key: {network: mean_r}}
results.consistency_map           # 3D numpy array, voxel-wise mean coupling
results.variability_map           # 3D numpy array, voxel-wise std coupling
```

## Project Structure

```
SC-FC_coupling/
  scfc/
    __init__.py
    config.py          # Paths, atlas registry, configuration
    structural.py      # SC: DWI registration + tck2connectome
    functional.py      # FC: nilearn time-series + correlation
    coupling.py        # SC-FC coupling: global, regional, permutation tests
    robustness.py      # Cross-parcellation comparison + network annotation
    viz.py             # Publication-ready matplotlib/nilearn figures
    cli.py             # Command-line interface
  scripts/
    run_all.py         # Pipeline entry point
  outputs/
    matrices/          # SC and FC matrices (CSV)
    figures/           # Generated plots (PNG, 300 dpi)
    registration/      # Atlas warps and transform matrices
```

## Requirements

### System Tools

- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/) >= 6.0 (FLIRT for registration)
- [MRtrix3](https://www.mrtrix.org/) >= 3.0 (tck2connectome for tractography-based connectomes)

### Python >= 3.10

```
nibabel >= 5.0
nilearn >= 0.10
numpy >= 1.24
scipy >= 1.10
pandas >= 2.0
matplotlib >= 3.7
seaborn >= 0.12
scikit-learn >= 1.3
```

Install all dependencies:
```bash
pip install -e .
```

## Data Requirements

The tool is designed to work with standard neuroimaging data:

- **Tractogram:** MRtrix3 `.tck` file with optional SIFT2 weights (`.txt`)
- **Resting-state fMRI:** Preprocessed, in MNI space (`.nii.gz`)
- **Brain atlases:** Parcellation volumes in MNI 2mm space (`.nii.gz`)
- **Atlas LUTs** (optional): Schaefer freeview LUT files for network labeling

All paths are configured in `scfc/config.py` -- edit this file to point to your data.

## Adding New Atlases

```python
# In scfc/config.py, add to the ATLASES dict:
ATLASES["my_atlas"] = AtlasDef(
    name="My Custom Atlas",
    path=ATLAS_DIR / "my_atlas" / "my_atlas_MNI152_2mm.nii.gz",
    n_parcels=200,
    description="Custom 200-region parcellation",
)
ACTIVE_ATLASES.append("my_atlas")
```

## Contributing

Contributions are welcome. Some directions for extension:

- Additional parcellations (Glasser HCP 360, Brodmann, DKT, custom atlases)
- Group-level FC averaging across multiple subjects
- Integration with [neuromaps](https://github.com/netneurolab/neuromaps) for spatial transcriptomics annotation
- Alternative SC-FC coupling models (communicability, diffusion, neural mass)
- Support for HCP-style `.dscalar.nii` surface data
- BIDS-app packaging

Please open an issue or pull request.

## Citation

If you use this tool in your research, please cite:

```
SC-FC Coupling Atlas: Cross-parcellation structure-function coupling robustness analysis.
https://github.com/cYoren/SC-FC_coupling
```

## References

- Schaefer, A. et al. (2018). Local-global parcellation of the human cerebral cortex. *Cerebral Cortex*, 28(9), 3095-3114.
- Yeo, B.T.T. et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *J Neurophysiol*, 106(3), 1125-1165.
- Zalesky, A. et al. (2010). Whole-brain anatomical networks: Does the choice of nodes matter? *NeuroImage*, 50(3), 970-983.
- de Reus, M.A. & van den Heuvel, M.P. (2013). The parcellation-based connectome: Limitations and extensions. *NeuroImage*, 80, 397-404.
- Margulies, D.S. et al. (2016). Situating the default-mode network along a principal gradient of macroscale cortical organization. *PNAS*, 113(44), 12574-12579.

## Credits

Developed by [cYoren](https://github.com/cYoren).

Built with: [nilearn](https://nilearn.github.io/), [nibabel](https://nipy.org/nibabel/), [MRtrix3](https://www.mrtrix.org/), [FSL](https://fsl.fmrib.ox.ac.uk/fsl/), [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/).

## License

MIT License

Copyright (c) 2026 cYoren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
