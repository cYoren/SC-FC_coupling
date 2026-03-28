"""Visualization: publication-ready SC-FC coupling figures."""

import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from scipy import stats

from .config import FIGURES_DIR, ATLASES, ensure_dirs
from .coupling import CouplingResult

log = logging.getLogger(__name__)

# Style
sns.set_theme(style="white", font_scale=1.1)
CMAP_COUPLING = "RdBu_r"
CMAP_MATRIX = "RdBu_r"
CMAP_SC = "hot"


def save_fig(fig, name: str, dpi: int = 300):
    """Save figure to outputs/figures/."""
    ensure_dirs()
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", path)
    return path


# ── 1. SC and FC matrices side by side ──────────────────────────────────

def plot_matrices(
    sc: np.ndarray,
    fc: np.ndarray,
    atlas_key: str,
) -> Path:
    """Plot SC and FC matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # SC (log scale)
    sc_plot = sc.copy()
    sc_plot[sc_plot > 0] = np.log10(sc_plot[sc_plot > 0])
    im0 = axes[0].imshow(sc_plot, cmap=CMAP_SC, aspect="equal")
    axes[0].set_title(f"Structural Connectivity (log₁₀)\n{ATLASES[atlas_key].name}")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # FC
    vmax = np.percentile(np.abs(fc), 98)
    im1 = axes[1].imshow(fc, cmap=CMAP_MATRIX, vmin=-vmax, vmax=vmax, aspect="equal")
    axes[1].set_title(f"Functional Connectivity\n{ATLASES[atlas_key].name}")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    for ax in axes:
        ax.set_xlabel("Parcel")
        ax.set_ylabel("Parcel")

    fig.suptitle(f"SC and FC Matrices — {ATLASES[atlas_key].name}", fontsize=14, y=1.02)
    return save_fig(fig, f"matrices_{atlas_key}")


# ── 2. Global SC-FC scatter ─────────────────────────────────────────────

def plot_global_scatter(
    sc: np.ndarray,
    fc: np.ndarray,
    result: CouplingResult,
    atlas_key: str,
) -> Path:
    """Scatter plot of SC vs FC edges with regression line."""
    n = min(sc.shape[0], fc.shape[0])
    sc_use = sc[:n, :n]
    fc_use = fc[:n, :n]

    idx = np.triu_indices(n, k=1)
    sc_vec = sc_use[idx]
    fc_vec = fc_use[idx]

    mask = sc_vec > 0
    sc_log = np.log10(sc_vec[mask])
    fc_masked = fc_vec[mask]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Subsample for plotting if too many points
    n_pts = len(sc_log)
    if n_pts > 5000:
        rng = np.random.default_rng(42)
        sample = rng.choice(n_pts, 5000, replace=False)
        ax.scatter(sc_log[sample], fc_masked[sample], alpha=0.15, s=4, c="steelblue", rasterized=True)
    else:
        ax.scatter(sc_log, fc_masked, alpha=0.2, s=6, c="steelblue", rasterized=True)

    # Regression line
    slope, intercept = np.polyfit(sc_log, fc_masked, 1)
    x_line = np.linspace(sc_log.min(), sc_log.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", lw=2, label="OLS fit")

    ax.set_xlabel("log₁₀(SC weight)")
    ax.set_ylabel("FC (Pearson r)")
    ax.set_title(
        f"SC-FC Coupling — {ATLASES[atlas_key].name}\n"
        f"r = {result.global_r:.3f}, p = {result.global_p:.4f}, "
        f"density = {result.sc_density:.1%}",
        fontsize=12,
    )
    ax.legend()
    sns.despine()

    return save_fig(fig, f"scatter_{atlas_key}")


# ── 3. Regional coupling bar plot by network ────────────────────────────

def plot_network_coupling(
    network_coupling: dict[str, dict[str, float]],
) -> Path:
    """Bar plot of mean regional coupling per Yeo network, across parcellations."""
    network_order = [
        "Visual", "Somatomotor", "DorsAttn", "SalVentAttn",
        "Limbic", "Control", "Default",
    ]
    network_colors = {
        "Visual": "#781286",
        "Somatomotor": "#4682B4",
        "DorsAttn": "#00760E",
        "SalVentAttn": "#C43AFA",
        "Limbic": "#DCF8A4",
        "Control": "#E69422",
        "Default": "#CD3E4E",
    }

    atlas_keys = sorted(network_coupling.keys())
    n_atlases = len(atlas_keys)
    n_nets = len(network_order)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8 / n_atlases
    x = np.arange(n_nets)

    for i, atlas_key in enumerate(atlas_keys):
        nc = network_coupling[atlas_key]
        vals = [nc.get(net, np.nan) for net in network_order]
        offset = (i - n_atlases / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width,
            label=ATLASES[atlas_key].name,
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(network_order, rotation=30, ha="right")
    ax.set_ylabel("Mean Regional SC-FC Coupling (r)")
    ax.set_title("SC-FC Coupling by Yeo Network Across Parcellations")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    sns.despine()

    return save_fig(fig, "network_coupling")


# ── 4. Cross-parcellation comparison ────────────────────────────────────

def plot_global_comparison(
    global_r: dict[str, float],
    global_p: dict[str, float],
) -> Path:
    """Bar plot comparing global SC-FC r across atlases."""
    keys = list(global_r.keys())
    names = [ATLASES[k].name for k in keys]
    r_vals = [global_r[k] for k in keys]
    p_vals = [global_p[k] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(keys))
    bars = ax.bar(names, r_vals, color=colors, edgecolor="white", linewidth=1)

    # Add significance stars
    for i, (bar, p) in enumerate(zip(bars, p_vals)):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            star, ha="center", va="bottom", fontsize=11,
        )

    ax.set_ylabel("Global SC-FC Coupling (Pearson r)")
    ax.set_title("SC-FC Coupling Strength Across Parcellations")
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=20, ha="right")
    sns.despine()

    return save_fig(fig, "global_comparison")


# ── 5. Brain surface maps (axial slices) ────────────────────────────────

def plot_coupling_brain(
    voxelwise_coupling: np.ndarray,
    atlas_path,
    atlas_key: str,
    title_suffix: str = "",
) -> Path:
    """Plot regional coupling on axial brain slices."""
    from nilearn import plotting, image

    img = nib.load(str(atlas_path))
    coupling_img = nib.Nifti1Image(voxelwise_coupling, img.affine, img.header)

    fig, axes = plt.subplots(1, 1, figsize=(12, 4))

    valid = voxelwise_coupling[~np.isnan(voxelwise_coupling)]
    if len(valid) == 0:
        log.warning("No valid voxels for brain plot: %s", atlas_key)
        plt.close(fig)
        return None

    vmax = np.percentile(np.abs(valid), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    display = plotting.plot_stat_map(
        coupling_img,
        display_mode="z",
        cut_coords=7,
        colorbar=True,
        cmap=CMAP_COUPLING,
        vmax=vmax,
        title=f"Regional SC-FC Coupling — {ATLASES[atlas_key].name}{title_suffix}",
        figure=fig,
    )

    return save_fig(fig, f"brain_{atlas_key}{title_suffix.replace(' ', '_')}")


def plot_consistency_brain(
    consistency_map: np.ndarray,
    variability_map: np.ndarray,
    reference_atlas_path,
) -> Path:
    """Plot cross-parcellation consistency and variability maps."""
    from nilearn import plotting

    img = nib.load(str(reference_atlas_path))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Consistency
    cons_img = nib.Nifti1Image(consistency_map, img.affine, img.header)
    valid = consistency_map[~np.isnan(consistency_map)]
    if len(valid) > 0:
        vmax = np.percentile(np.abs(valid), 95)
        plotting.plot_stat_map(
            cons_img, display_mode="z", cut_coords=7, colorbar=True,
            cmap=CMAP_COUPLING, vmax=vmax,
            title="Mean SC-FC Coupling (Cross-Parcellation Consistency)",
            figure=fig, axes=axes[0],
        )

    # Variability
    var_img = nib.Nifti1Image(variability_map, img.affine, img.header)
    valid_v = variability_map[~np.isnan(variability_map)]
    if len(valid_v) > 0:
        vmax_v = np.percentile(valid_v[valid_v > 0], 95) if np.any(valid_v > 0) else 0.1
        plotting.plot_stat_map(
            var_img, display_mode="z", cut_coords=7, colorbar=True,
            cmap="YlOrRd", vmax=vmax_v,
            title="SC-FC Coupling Variability Across Parcellations",
            figure=fig, axes=axes[1],
        )

    fig.tight_layout()
    return save_fig(fig, "consistency_variability_brain")


# ── 6. Summary dashboard ───────────────────────────────────────────────

def plot_summary_dashboard(
    sc_matrices: dict[str, np.ndarray],
    fc_matrices: dict[str, np.ndarray],
    coupling_results: dict[str, CouplingResult],
) -> Path:
    """Create a multi-panel summary figure."""
    n = len(coupling_results)
    fig = plt.figure(figsize=(6 * n, 10))
    gs = gridspec.GridSpec(2, n, hspace=0.35, wspace=0.3)

    for i, (key, result) in enumerate(coupling_results.items()):
        sc = sc_matrices[key]
        fc = fc_matrices[key]
        n_roi = min(sc.shape[0], fc.shape[0])

        # Top row: SC-FC scatter
        ax_scatter = fig.add_subplot(gs[0, i])
        idx = np.triu_indices(n_roi, k=1)
        sc_vec = sc[:n_roi, :n_roi][idx]
        fc_vec = fc[:n_roi, :n_roi][idx]
        mask = sc_vec > 0
        sc_log = np.log10(sc_vec[mask])
        fc_m = fc_vec[mask]

        if len(sc_log) > 3000:
            rng = np.random.default_rng(42)
            s = rng.choice(len(sc_log), 3000, replace=False)
            ax_scatter.scatter(sc_log[s], fc_m[s], alpha=0.1, s=2, c="steelblue", rasterized=True)
        else:
            ax_scatter.scatter(sc_log, fc_m, alpha=0.15, s=3, c="steelblue", rasterized=True)

        slope, intercept = np.polyfit(sc_log, fc_m, 1)
        xl = np.linspace(sc_log.min(), sc_log.max(), 50)
        ax_scatter.plot(xl, slope * xl + intercept, "r-", lw=1.5)
        ax_scatter.set_title(f"{ATLASES[key].name}\nr={result.global_r:.3f}", fontsize=10)
        ax_scatter.set_xlabel("log₁₀(SC)", fontsize=9)
        ax_scatter.set_ylabel("FC", fontsize=9)

        # Bottom row: regional coupling histogram
        ax_hist = fig.add_subplot(gs[1, i])
        rc = result.regional_coupling
        rc_valid = rc[~np.isnan(rc)]
        ax_hist.hist(rc_valid, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
        ax_hist.axvline(np.median(rc_valid), color="red", linestyle="--", label=f"median={np.median(rc_valid):.3f}")
        ax_hist.set_xlabel("Regional SC-FC Coupling (r)", fontsize=9)
        ax_hist.set_ylabel("Count", fontsize=9)
        ax_hist.set_title(f"Regional Distribution\n{np.sum(~np.isnan(rc))}/{len(rc)} regions", fontsize=10)
        ax_hist.legend(fontsize=8)

    fig.suptitle("SC-FC Coupling Atlas — Cross-Parcellation Summary", fontsize=14, y=1.01)
    return save_fig(fig, "summary_dashboard")
