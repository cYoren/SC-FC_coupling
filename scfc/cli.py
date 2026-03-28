"""Command-line interface for SC-FC coupling pipeline."""

import argparse
import logging
import sys

import numpy as np

from .config import ACTIVE_ATLASES, ensure_dirs


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="SC-FC Coupling Atlas: cross-parcellation structure-function analysis",
    )
    parser.add_argument(
        "--atlases", nargs="+", default=ACTIVE_ATLASES,
        help="Atlas keys to analyze (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    parser.add_argument("--n-perm", type=int, default=1000, help="Permutations for significance")
    parser.add_argument("--skip-sc", action="store_true", help="Skip SC computation (use cached)")
    parser.add_argument("--skip-fc", action="store_true", help="Skip FC computation (use cached)")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("scfc")
    ensure_dirs()

    log.info("=" * 60)
    log.info("SC-FC Coupling Atlas — Cross-Parcellation Analysis")
    log.info("Atlases: %s", args.atlases)
    log.info("=" * 60)

    # ── Step 1: Structural Connectivity ──
    if not args.skip_sc:
        log.info("Step 1/4: Computing structural connectivity...")
        from .structural import compute_all_sc
        sc_matrices = compute_all_sc(args.atlases, force=args.force)
    else:
        log.info("Step 1/4: Loading cached SC matrices...")
        from .structural import compute_all_sc
        sc_matrices = compute_all_sc(args.atlases, force=False)

    # ── Step 2: Functional Connectivity ──
    if not args.skip_fc:
        log.info("Step 2/4: Computing functional connectivity...")
        from .functional import compute_all_fc
        fc_matrices = compute_all_fc(args.atlases, force=args.force)
    else:
        log.info("Step 2/4: Loading cached FC matrices...")
        from .functional import compute_all_fc
        fc_matrices = compute_all_fc(args.atlases, force=False)

    # ── Step 3: Coupling + Robustness Analysis ──
    log.info("Step 3/4: SC-FC coupling and robustness analysis...")
    from .robustness import analyze_robustness
    robustness = analyze_robustness(sc_matrices, fc_matrices, n_perm=args.n_perm)

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)
    for key, r_val in robustness.global_r_values.items():
        p_val = robustness.global_p_values[key]
        cr = robustness.coupling_results[key]
        n_valid = int(np.sum(~np.isnan(cr.regional_coupling)))
        log.info(
            "  %s: r=%.4f (p=%.4f), density=%.1f%%, %d/%d valid regions",
            key, r_val, p_val, cr.sc_density * 100,
            n_valid, len(cr.regional_coupling),
        )

    if robustness.network_coupling:
        log.info("\nNetwork-level coupling:")
        for atlas_key, nc in robustness.network_coupling.items():
            log.info("  %s:", atlas_key)
            for net, val in sorted(nc.items(), key=lambda x: -x[1]):
                log.info("    %-20s: %.4f", net, val)

    # ── Step 4: Visualization ──
    if not args.skip_plots:
        log.info("\nStep 4/4: Generating figures...")
        from . import viz
        from .config import ATLASES

        for key in args.atlases:
            if key in sc_matrices and key in fc_matrices:
                viz.plot_matrices(sc_matrices[key], fc_matrices[key], key)
                viz.plot_global_scatter(
                    sc_matrices[key], fc_matrices[key],
                    robustness.coupling_results[key], key,
                )
                # Brain maps
                if key in robustness.voxelwise_coupling:
                    viz.plot_coupling_brain(
                        robustness.voxelwise_coupling[key],
                        ATLASES[key].path, key,
                    )

        viz.plot_global_comparison(
            robustness.global_r_values, robustness.global_p_values,
        )
        viz.plot_summary_dashboard(sc_matrices, fc_matrices, robustness.coupling_results)

        if robustness.network_coupling:
            viz.plot_network_coupling(robustness.network_coupling)

        if robustness.consistency_map is not None:
            ref_key = [k for k in args.atlases if "schaefer" in k][0]
            viz.plot_consistency_brain(
                robustness.consistency_map,
                robustness.variability_map,
                ATLASES[ref_key].path,
            )

        log.info("Figures saved to: %s", viz.FIGURES_DIR)

    log.info("\nDone!")
    return robustness


if __name__ == "__main__":
    main()
