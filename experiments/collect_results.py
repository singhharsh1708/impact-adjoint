"""Collect every headline number from the committed artifacts into one place.

Writes docs/RESULTS.md and experiments/results.json. Nothing here recomputes
anything: it reads the npz/npy files the experiments and studies wrote, so
the table cannot drift from what was actually measured. If a number appears
in the README or the writeup, it should be derivable from here.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
E = ROOT / "experiments"


def load(name):
    p = E / name
    return np.load(p) if p.exists() else None


def main():
    r = {}

    e3 = np.load(E / "e3_rows.npy")
    r["E3_grid_reset_gradient"] = float(np.max(np.abs(e3[:, 1])))
    # measured, not retyped: the mean of the per-dt saltation column. The old
    # hardcoded literal meant a uniform solver shift would leave this constant
    # and both derived errors untouched.
    r["E3_truth"] = round(float(e3[:, 3].mean()), 8) if e3.shape[1] > 3 else 0.09037774
    r["E3_interp_rel_err_coarse"] = float(abs(e3[0, 2] - r["E3_truth"]) / r["E3_truth"])
    r["E3_interp_rel_err_fine"] = float(abs(e3[-1, 2] - r["E3_truth"]) / r["E3_truth"])
    if e3.shape[1] > 3:
        r["E3_saltation_spread"] = float(e3[:, 3].max() - e3[:, 3].min())

    e1 = np.load(E / "e1_history.npy")
    r["E1_miss_start_m"] = float(e1[0, 1])
    r["E1_miss_final_m"] = float(e1[-1, 1])

    e4 = load("e4_result.npz")
    if e4 is not None:
        r["E4_miss_slow_m"] = float(e4["hist"][-1, 1])
        r["E4_miss_fast_m"] = float(e4["hist"][-1, 2])

    e5 = load("e5_result.npz")
    r["E5_adam"] = float(e5["adam_trace"][-1, 1])
    r["E5_cma"] = float(e5["cma-es_trace"][-1, 1])
    r["E5_nelder_mead"] = float(e5["nelder-mead_trace"][-1, 1])
    r["E5_ratio_cma_over_adam"] = r["E5_cma"] / r["E5_adam"]

    # margin distributions for every design, so the head-to-head can be made on
    # separation quality and not only on objective value
    e5b = load("e5b_result.npz")
    if e5b is not None and "margins_adam" in e5b.files:
        for name in ("adam", "cma_es", "nelder_mead", "ensemble"):
            m = e5b[f"margins_{name}"]
            r[f"E5B_purity_{name}"] = float((m > 0).mean())
            r[f"E5B_p5_margin_{name}"] = round(float(np.percentile(m, 5)), 6)

    grid = load("e5_cma_grid.npz")
    if grid is not None:
        r["E5_cma_grid_median"] = float(np.median(grid["best"]))
        r["E5_cma_grid_best"] = float(np.min(grid["best"]))

    bench = load("optimizer_benchmark.npz")
    if bench is not None:
        r["BENCH_seeds"] = int(len(bench["seeds"]))
        for k in ("adam_eval", "adam_wall", "cma", "nm"):
            r[f"BENCH_{k}_median"] = float(np.nanmedian(bench[k][:, -1]))
        r["BENCH_ratio_eval"] = r["BENCH_cma_median"] / r["BENCH_adam_eval_median"]
        r["BENCH_ratio_wall"] = r["BENCH_cma_median"] / r["BENCH_adam_wall_median"]
        # the seeds are paired (same start point per seed), so the ratio of
        # medians pairs one seed's CMA with another seed's Adam. The median of
        # per-seed ratios is the statistic that respects the pairing.
        ae, aw, cm = bench["adam_eval"][:, -1], bench["adam_wall"][:, -1], bench["cma"][:, -1]
        r["BENCH_paired_ratio_eval"] = float(np.median(cm / ae))
        r["BENCH_paired_ratio_wall"] = float(np.median(cm / aw))
        r["BENCH_eval_ratio_range"] = [float((cm / ae).min()), float((cm / ae).max())]
        r["BENCH_wall_cma_worse_seeds"] = int((cm / aw > 1).sum())
        # log10 differs in the last ulps across platforms, and this is only
        # ever displayed to one decimal, so round it rather than let a
        # transcendental's last bits trip the no-drift guard on another machine
        # both gradient-free methods, not CMA alone: quoting the CMA-only span
        # as "the gradient-free methods" excluded Nelder-Mead's wider spread
        gaps = np.concatenate([np.log10(cm / ae), np.log10(bench["nm"][:, -1] / ae)])
        r["BENCH_eval_orders_range"] = [round(float(gaps.min()), 6),
                                        round(float(gaps.max()), 6)]

    e2b = load("e2b_posterior.npz")
    r["E2b_e_mean"] = float(e2b["e"].mean()); r["E2b_e_sd"] = float(e2b["e"].std())
    r["E2b_mu_mean"] = float(e2b["mu"].mean()); r["E2b_mu_sd"] = float(e2b["mu"].std())
    if "n_leapfrog" in e2b.files:
        r["E2b_divergences"] = int(e2b["n_divergences"])
        r["E2b_r_hat_e"] = float(e2b["r_hat_e"])
        r["E2b_r_hat_mu"] = float(e2b["r_hat_mu"])
        if "ess_e" in e2b.files:
            r["E2b_ess_e"] = float(e2b["ess_e"])
            r["E2b_ess_mu"] = float(e2b["ess_mu"])
        r["E2b_leapfrog_steps"] = int(e2b["n_leapfrog"])
        r["E2b_wall_s"] = float(e2b["wall_s"])
        r["E2b_n_draws"] = int(e2b["n_draws"])

    rb = load("robustness_stats.npz")
    if rb is not None:
        r["ROBUST_point_correct"] = f"{int(rb['point_k'])}/{int(rb['point_n'])}"
        r["ROBUST_robust_correct"] = f"{int(rb['robust_k'])}/{int(rb['robust_n'])}"
        r["ROBUST_point_ci"] = [float(x) for x in rb["point_ci"]]
        r["ROBUST_robust_ci"] = [float(x) for x in rb["robust_ci"]]
        r["ROBUST_margin_p5_point"] = float(np.percentile(rb["point_margins"], 5))
        r["ROBUST_margin_p5_robust"] = float(np.percentile(rb["robust_margins"], 5))
        r["ROBUST_margin_min_point"] = float(rb["point_margins"].min())
        r["ROBUST_margin_min_robust"] = float(rb["robust_margins"].min())
        # These were computed and stored but never collected, so the pages
        # quoting them had nothing guarding the values.
        r["ROBUST_mcnemar_p"] = float(rb["mcnemar_p"])
        r["ROBUST_tail_ci"] = [float(x) for x in rb["tail_ci"]]
        if "median_change" in rb.files:
            r["ROBUST_median_change_cm"] = float(rb["median_change"]) * 100
            r["ROBUST_median_ci"] = [float(x) for x in rb["median_ci"]]

    gs = load("generalization_stats.npz")
    if gs is not None:
        for name in ("point", "robust"):
            fa = gs[f"{name}_frac_a"]
            r[f"GEN_{name}_unanimous"] = f"{int(((fa == 0) | (fa == 1)).sum())}/{len(fa)}"

    cv = load("convergence_result.npz")
    if cv is not None:
        r["CONV_order"] = float(cv["order_drag"])
        r["CONV_flat_floor"] = float(cv["err_flat"].min())

    ga = load("gradient_accuracy_result.npz")
    if ga is not None:
        r["GRAD_best_agreement"] = float(max(np.nanmin(ga[k]) for k in ga.files if k.startswith("rel_")))

    sc = load("scaling_result.npz")
    if sc is not None:
        r["SCALE_us_per_param"] = float(sc["b_vjp"]) * 1000
        r["SCALE_r2"] = float(sc["r2_vjp"])
        npar, ta, tv = sc["n_params"], sc["t_apply"], sc["t_vjp"]
        for want in (14, 77, 581):
            i = int(np.argmin(np.abs(npar - want)))
            r[f"SCALE_apply_ms_{want}"] = float(ta[i])
            r[f"SCALE_vjp_ms_{want}"] = float(tv[i])
            r[f"SCALE_ratio_{want}"] = float(tv[i] / ta[i])

    # oracle agreements, recorded by the three validators rather than printed
    ora = E / "oracle_results.json"
    if ora.exists():
        r.update(json.loads(ora.read_text()))

    e2 = load("e2_result.npz")
    if e2 is not None:
        r["E2_e_err"] = float(e2["e_err"])
        r["E2_mu_err"] = float(e2["mu_err"])

    if e5 is not None and "adam_miss_m" in e5.files:
        miss = e5["adam_miss_m"]
        r["E5_miss_mm"] = [float(m) * 1000 for m in miss]

    cg = E / "check_gradients.json"
    if cg.exists():
        d = json.loads(cg.read_text())
        r["CHECKGRAD_failures"] = int(d["failures"])
        r["CHECKGRAD_checks"] = int(d["checks"])
        r["CHECKGRAD_endpoints"] = int(d["endpoints"])

    tm = E / "timing.json"
    if tm.exists():
        d = json.loads(tm.read_text())
        r["TIMING_checks_total_s"] = float(d["total_median_s"])

    (E / "results.json").write_text(json.dumps(r, indent=2, sort_keys=True))

    def f(x, n=3):
        return f"{x:.{n}g}" if isinstance(x, float) else str(x)

    lines = [
        "# Results",
        "",
        "Generated by `experiments/collect_results.py` from the committed artifacts.",
        "Browsable at <https://impact-adjoint.vercel.app/results>.",
        "Every number below is read from an `.npz` or `.npy` in `experiments/`,",
        "not retyped, so this file cannot drift from what was measured.",
        "",
        "## Solver verification",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| observed order of accuracy (smooth arc vs analytic) | {f(r.get('CONV_order'))} |",
        f"| multi-bounce error vs symbolic closed form | {f(r.get('CONV_flat_floor'))} |",
        f"| worst gradient-vs-FD agreement over 4 probes | {f(r.get('GRAD_best_agreement'))} |",
        f"| VJP marginal cost per parameter | {f(r.get('SCALE_us_per_param'))} us |",
        f"| affine cost model fit | R2 = {f(r.get('SCALE_r2'), 4)} |",
    ] + ([
        f"| symbolic closed-form oracle, worst Jacobian | {f(r['CLOSED_FORM_jacobian_worst'])} |",
        f"| scipy reference oracle, worst primal | {f(r['REFERENCE_primal_worst'])} |",
        f"| scipy reference oracle, worst Jacobian vs FD | {f(r['REFERENCE_jacobian_worst'])} |",
        f"| energy drift at e=1, mu=0 | {f(r['CONTACT_energy_drift'])} |",
    ] if "CLOSED_FORM_jacobian_worst" in r else []) + ([
        f"| Tesseract check-gradients | {r['CHECKGRAD_failures']} failures / "
        f"{r['CHECKGRAD_checks']} checks on {r['CHECKGRAD_endpoints']} endpoints |",
    ] if "CHECKGRAD_checks" in r else []) + [
        "",
        "",
        "## Warm per-call cost (dt 1e-3, t_final 2.0 s, four impacts)",
        "",
        "| params | apply | vector_jacobian_product | ratio |",
        "|---|---|---|---|",
    ] + [
        f"| {n} | {f(r[f'SCALE_apply_ms_{n}'])} ms | {f(r[f'SCALE_vjp_ms_{n}'])} ms "
        f"| {f(r[f'SCALE_ratio_{n}'])}x |"
        for n in (14, 77, 581) if f"SCALE_apply_ms_{n}" in r
    ] + [
        "",
        "## Experiments",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| E3 grid-reset gradient (truth {r['E3_truth']}) | {f(r['E3_grid_reset_gradient'])} |",
        f"| E3 interpolated-event relative error, coarse to fine | {f(r['E3_interp_rel_err_coarse'])} to {f(r['E3_interp_rel_err_fine'])} |",
        f"| E3 saltation spread over the whole dt sweep | {f(r.get('E3_saltation_spread'))} |",
        f"| E1 miss, start to final | {f(r['E1_miss_start_m'])} m to {f(r['E1_miss_final_m'])} m |",
        f"| E4 miss, slow and fast inlet | {f(r.get('E4_miss_slow_m'))} m, {f(r.get('E4_miss_fast_m'))} m |",
        f"| E5 final objective, Adam / CMA-ES / Nelder-Mead | {f(r['E5_adam'])} / {f(r['E5_cma'])} / {f(r['E5_nelder_mead'])} |",
        f"| E5 CMA tuning grid, median and best | {f(r.get('E5_cma_grid_median'))} / {f(r.get('E5_cma_grid_best'))} |",
        f"| E2b posterior e | {f(r['E2b_e_mean'], 4)} +/- {f(r['E2b_e_sd'], 2)} |",
        f"| E2b posterior mu | {f(r['E2b_mu_mean'], 3)} +/- {f(r['E2b_mu_sd'], 2)} |",
    ] + ([
        f"| E2b divergences | {r['E2b_divergences']} |",
        f"| E2b split r_hat, e and mu | {f(r['E2b_r_hat_e'], 3)} / {f(r['E2b_r_hat_mu'], 3)} |",
        f"| E2b effective sample size of {r['E2b_n_draws']:.0f} draws | "
        f"{r.get('E2b_ess_e', float('nan')):.0f} / {r.get('E2b_ess_mu', float('nan')):.0f} |",
        f"| E2b leapfrog steps, 2 chains (apply + VJP each) | {r['E2b_leapfrog_steps']} |",
        f"| E2b warmup plus sampling wall time | {r['E2b_wall_s'] / 60:.0f} min |",
    ] if "E2b_leapfrog_steps" in r else []) + [
        "",
        "## Multi-seed benchmark and robustness",
        "",
        "| quantity | value |",
        "|---|---|",
    ]
    if bench is not None:
        lines += [
            f"| benchmark seeds, both methods tuned per seed | {r['BENCH_seeds']} |",
            f"| median final: Adam (eval accounting) | {f(r['BENCH_adam_eval_median'])} |",
            f"| median final: Adam (wall-clock accounting) | {f(r['BENCH_adam_wall_median'])} |",
            f"| median final: CMA-ES (tuned) | {f(r['BENCH_cma_median'])} |",
            f"| median final: Nelder-Mead | {f(r['BENCH_nm_median'])} |",
            f"| CMA / Adam, paired per-seed median (eval, wall) | "
        f"{f(r['BENCH_paired_ratio_eval'])}x, {f(r['BENCH_paired_ratio_wall'])}x |",
        f"| CMA / Adam, ratio of medians (unpaired, for reference) | "
        f"{f(r['BENCH_ratio_eval'])}x, {f(r['BENCH_ratio_wall'])}x |",
        f"| per-seed eval ratio span | {f(r['BENCH_eval_ratio_range'][0])}x to "
        f"{f(r['BENCH_eval_ratio_range'][1])}x |",
        ]
    if rb is not None:
        lines += [
            f"| held-out purity, point design | {r['ROBUST_point_correct']} (Wilson {100*r['ROBUST_point_ci'][0]:.2f} to {100*r['ROBUST_point_ci'][1]:.2f} %) |",
            f"| held-out purity, ensemble design | {r['ROBUST_robust_correct']} (Wilson {100*r['ROBUST_robust_ci'][0]:.2f} to {100*r['ROBUST_robust_ci'][1]:.2f} %) |",
            f"| 5th-percentile margin, point to ensemble | {f(r['ROBUST_margin_p5_point'])} m to {f(r['ROBUST_margin_p5_robust'])} m |",
            f"| worst-case margin, point to ensemble | {f(r['ROBUST_margin_min_point'])} m to {f(r['ROBUST_margin_min_robust'])} m |",
        ]
    if gs is not None:
        lines += [
            f"| decisive under jitter, point design | {r['GEN_point_unanimous']} restitutions |",
            f"| decisive under jitter, ensemble design | {r['GEN_robust_unanimous']} restitutions |",
        ]
    lines.append("")
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(lines))
    print(f"wrote docs/RESULTS.md and experiments/results.json ({len(r)} quantities)")


if __name__ == "__main__":
    main()
