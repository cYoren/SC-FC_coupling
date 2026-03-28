"""SC-FC coupling analysis: global correlation, regional coupling, significance."""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

log = logging.getLogger(__name__)


@dataclass
class CouplingResult:
    """Results of SC-FC coupling analysis for one atlas."""
    atlas_key: str
    global_r: float
    global_p: float
    regional_coupling: np.ndarray  # per-node coupling strength
    regional_p: np.ndarray         # per-node p-values
    n_edges: int
    sc_density: float              # fraction of non-zero SC edges


def _upper_triangle(mat: np.ndarray) -> np.ndarray:
    """Extract upper triangle (excluding diagonal) as flat vector."""
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


def global_coupling(
    sc: np.ndarray,
    fc: np.ndarray,
    log_transform_sc: bool = True,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute global SC-FC coupling (Pearson r on upper-triangle edges).

    SC is log-transformed by default (standard practice, since SC values
    span orders of magnitude while FC is bounded).

    Returns (r, p, sc_vec, fc_vec) where sc_vec/fc_vec are the edge vectors used.
    """
    n = min(sc.shape[0], fc.shape[0])
    sc = sc[:n, :n]
    fc = fc[:n, :n]

    sc_vec = _upper_triangle(sc)
    fc_vec = _upper_triangle(fc)

    # Only use edges where SC > 0 (no structural connection → uninformative)
    mask = sc_vec > 0
    if mask.sum() < 10:
        log.warning("Very few non-zero SC edges (%d). Results may be unreliable.", mask.sum())

    sc_use = sc_vec[mask]
    fc_use = fc_vec[mask]

    if log_transform_sc:
        sc_use = np.log10(sc_use + 1e-10)

    r, p = stats.pearsonr(sc_use, fc_use)
    return r, p, sc_use, fc_use


def regional_coupling(
    sc: np.ndarray,
    fc: np.ndarray,
    log_transform_sc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-region SC-FC coupling.

    For each node i, correlate its SC profile (row i) with its FC profile (row i).
    This measures how well a region's structural connections predict its
    functional connections.

    Returns (coupling_vector, p_values) of length n_parcels.
    """
    n = min(sc.shape[0], fc.shape[0])
    sc = sc[:n, :n]
    fc = fc[:n, :n]

    coupling = np.zeros(n)
    pvals = np.ones(n)

    for i in range(n):
        sc_row = sc[i, :].copy()
        fc_row = fc[i, :].copy()

        # Exclude self-connection
        sc_row = np.delete(sc_row, i)
        fc_row = np.delete(fc_row, i)

        # Only use non-zero SC entries
        mask = sc_row > 0
        if mask.sum() < 5:
            coupling[i] = np.nan
            pvals[i] = np.nan
            continue

        sc_use = sc_row[mask]
        fc_use = fc_row[mask]

        if log_transform_sc:
            sc_use = np.log10(sc_use + 1e-10)

        r, p = stats.pearsonr(sc_use, fc_use)
        coupling[i] = r
        pvals[i] = p

    return coupling, pvals


def permutation_test_global(
    sc: np.ndarray,
    fc: np.ndarray,
    n_perm: int = 1000,
    log_transform_sc: bool = True,
) -> tuple[float, float]:
    """
    Permutation test for global SC-FC coupling significance.

    Shuffles FC edges while preserving SC structure to generate null distribution.
    Returns (observed_r, p_perm).
    """
    r_obs, _, sc_vec, fc_vec = global_coupling(sc, fc, log_transform_sc)

    null_dist = np.zeros(n_perm)
    rng = np.random.default_rng(42)
    for i in range(n_perm):
        fc_shuffled = rng.permutation(fc_vec)
        null_dist[i] = stats.pearsonr(sc_vec, fc_shuffled)[0]

    p_perm = np.mean(np.abs(null_dist) >= np.abs(r_obs))
    log.info("Permutation test: r=%.4f, p_perm=%.4f (n=%d)", r_obs, p_perm, n_perm)
    return r_obs, p_perm


def analyze_coupling(
    sc: np.ndarray,
    fc: np.ndarray,
    atlas_key: str,
    n_perm: int = 1000,
) -> CouplingResult:
    """Full coupling analysis for one atlas: global + regional + permutation test."""
    log.info("Analyzing SC-FC coupling for %s...", atlas_key)

    r_global, p_global = permutation_test_global(sc, fc, n_perm=n_perm)
    reg_coupling, reg_p = regional_coupling(sc, fc)

    sc_vec = _upper_triangle(sc[:fc.shape[0], :fc.shape[0]])
    n_edges = len(sc_vec)
    sc_density = np.mean(sc_vec > 0)

    result = CouplingResult(
        atlas_key=atlas_key,
        global_r=r_global,
        global_p=p_global,
        regional_coupling=reg_coupling,
        regional_p=reg_p,
        n_edges=n_edges,
        sc_density=sc_density,
    )
    log.info(
        "%s: global r=%.4f (p=%.4f), density=%.2f%%, %d/%d regions valid",
        atlas_key, r_global, p_global, sc_density * 100,
        np.sum(~np.isnan(reg_coupling)), len(reg_coupling),
    )
    return result
