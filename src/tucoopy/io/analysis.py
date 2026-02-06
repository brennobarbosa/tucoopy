"""
# High-level analysis builder for cooperative game artifacts.

This module builds the `analysis` section used by the JSON/animation contracts:

- computes selected point solutions (e.g. Shapley, Banzhaf),
- computes selected set-valued objects (e.g. imputation set, core),
- attaches diagnostics summaries (e.g. core membership of solutions),
- and applies limits (max players, max points, truncation) to keep outputs stable.

The resulting object is intended to be JSON-serializable and suitable for
renderers (e.g. the JS demo) and static reports.

Examples
--------
>>> from tucoopy import Game
>>> from tucoopy.io.analysis import build_analysis
>>> g = Game.from_coalitions(
...     n_players=2,
...     values={(): 0.0, (0,): 0.0, (1,): 0.0, (0, 1): 1.0},
... )
>>> report = build_analysis(g, include_blocking_regions=False, include_bundle=False)
>>> "solutions" in report and "sets" in report
True
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from ..base.types import GameProtocol, is_tabular_game
from ..base.exceptions import tucoopyError


ANALYSIS_CONTRACT_VERSION = "0.1.0"
ENGINE_ID = "tugcoopy"


def _to_jsonable(x: Any) -> Any:
    if is_dataclass(x) and not isinstance(x, type):
        return asdict(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _truncate_list(values: list[Any], max_len: int | None) -> tuple[list[Any], dict[str, Any]]:
    if max_len is None:
        return values, {"count_total": len(values), "count_returned": len(values), "truncated": False}
    m = max(0, int(max_len))
    if len(values) <= m:
        return values, {"count_total": len(values), "count_returned": len(values), "truncated": False}
    return values[:m], {"count_total": len(values), "count_returned": m, "truncated": True}


def build_analysis(
    game: GameProtocol,
    *,
    include_sets: bool = True,
    include_solutions: bool = True,
    include_diagnostics: bool = True,
    include_blocking_regions: bool = True,
    include_weber: bool = False,
    include_bargaining: bool = False,
    include_prekernel_set: bool = False,
    include_kernel_set: bool = False,
    bargaining_n_samples: int = 200,
    bargaining_max_attempts: int | None = None,
    prekernel_n_samples: int = 2000,
    kernel_n_samples: int = 5000,
    prekernel_max_points: int = 50,
    kernel_max_points: int = 50,
    sets_seed: int | None = 0,
    include_bundle: bool = True,
    bundle_max_players: int = 12,
    bundle_include_power_indices: bool = True,
    bundle_include_tau: bool = True,
    bundle_include_approx_solutions: bool = True,
    bundle_shapley_samples: int = 2000,
    bundle_seed: int | None = None,
    include_lp_explanations: bool = False,
    lp_explanations_max_players: int = 4,
    max_players: int = 4,
    max_points: int | None = 5000,
    tol: float = 1e-9,
    diagnostics_top_k: int = 8,
    diagnostics_max_list: int = 256,
) -> dict[str, Any]:
    n = int(game.n_players)
    approx_reasons: list[str] = []

    analysis: dict[str, Any] = {
        "meta": {
            "computed_by": ENGINE_ID,
            "contract_version": ANALYSIS_CONTRACT_VERSION,
            "build_analysis": {
                "include_sets": bool(include_sets),
                "include_solutions": bool(include_solutions),
                "include_diagnostics": bool(include_diagnostics),
                "include_blocking_regions": bool(include_blocking_regions),
                "include_weber": bool(include_weber),
                "include_bargaining": bool(include_bargaining),
                "include_prekernel_set": bool(include_prekernel_set),
                "include_kernel_set": bool(include_kernel_set),
                "bargaining_n_samples": int(bargaining_n_samples),
                "bargaining_max_attempts": int(bargaining_max_attempts) if bargaining_max_attempts is not None else None,
                "prekernel_n_samples": int(prekernel_n_samples),
                "kernel_n_samples": int(kernel_n_samples),
                "prekernel_max_points": int(prekernel_max_points),
                "kernel_max_points": int(kernel_max_points),
                "sets_seed": int(sets_seed) if sets_seed is not None else None,
                "include_bundle": bool(include_bundle),
                "bundle_max_players": int(bundle_max_players),
                "bundle_include_power_indices": bool(bundle_include_power_indices),
                "bundle_include_tau": bool(bundle_include_tau),
                "bundle_include_approx_solutions": bool(bundle_include_approx_solutions),
                "bundle_shapley_samples": int(bundle_shapley_samples),
                "bundle_seed": None if bundle_seed is None else int(bundle_seed),
                "include_lp_explanations": bool(include_lp_explanations),
                "lp_explanations_max_players": int(lp_explanations_max_players),
                "max_players": int(max_players),
                "max_points": None if max_points is None else int(max_points),
                "tol": float(tol),
                "diagnostics_top_k": int(diagnostics_top_k),
                "diagnostics_max_list": int(diagnostics_max_list),
            },
            "limits": {
                "max_players": int(max_players),
                "max_points": None if max_points is None else int(max_points),
                "bundle_max_players": int(bundle_max_players),
                "lp_explanations_max_players": int(lp_explanations_max_players),
                "diagnostics_top_k": int(diagnostics_top_k),
                "diagnostics_max_list": int(diagnostics_max_list),
            },
            "computed": {},
            "skipped": {},
        }
    }

    # ----------------------------
    # solutions
    # ----------------------------
    if include_solutions:
        if n <= int(max_players):
            from ..solutions import normalized_banzhaf_value, shapley_value

            analysis["solutions"] = {
                "shapley": {
                    "allocation": shapley_value(game),
                    "meta": {"computed_by": ENGINE_ID, "method": "shapley_value"},
                },
                "normalized_banzhaf": {
                    "allocation": normalized_banzhaf_value(game),
                    "meta": {"computed_by": ENGINE_ID, "method": "normalized_banzhaf_value"},
                },
            }

            # Optional LP-based classic solution points (when available).
            try:
                from ..solutions.nucleolus import nucleolus

                nu = nucleolus(game, tol=float(tol))
                analysis["solutions"]["nucleolus"] = {
                    "allocation": [float(v) for v in nu.x],
                    "meta": {"computed_by": ENGINE_ID, "method": "nucleolus"},
                }
            except Exception as e:
                analysis["meta"]["skipped"]["solutions.nucleolus"] = str(e)

            analysis["meta"]["computed"]["solutions"] = True
        else:
            analysis["meta"]["computed"]["solutions"] = False
            analysis["meta"]["skipped"]["solutions"] = f"n={n} > max_players={max_players}"
    else:
        analysis["meta"]["computed"]["solutions"] = False

    # ----------------------------
    # diagnostics
    # ----------------------------
    if include_diagnostics:
        diagnostics: dict[str, Any] = {}

        from ..properties import is_essential, is_simple_game

        grand = int(game.grand_coalition)
        v0 = float(game.value(0))
        vN = float(game.value(grand))
        sum_singletons = float(sum(float(game.value(1 << i)) for i in range(n)))
        essential = bool(is_essential(game))

        warnings: list[str] = []
        input_diag: dict[str, Any] = {
            "n_players": int(n),
            "grand_coalition_mask": int(grand),
            "v0": v0,
            "vN": vN,
            "sum_singletons": sum_singletons,
            "essential": essential,
            "simple_game": None,
            "monotone_simple_game": None,
            "monotone_counterexample": None,
            "warnings": warnings,
        }

        if n <= int(max_players):
            max_mask = (1 << n) - 1
            if is_tabular_game(game):
                present = set(int(m) for m in game.v.keys())
                expected = set(range(max_mask + 1))
                missing_all = sorted(expected - present)
                missing_count = len(missing_all)

                input_diag["characteristic_function_complete"] = (missing_count == 0)
                input_diag["missing_coalition_mask_count"] = int(missing_count)

                if missing_count <= max(0, int(diagnostics_max_list)):
                    input_diag["missing_coalition_masks"] = missing_all
                    input_diag["missing_coalition_masks_truncated"] = False
                else:
                    input_diag["missing_coalition_masks"] = missing_all[: max(0, int(diagnostics_max_list))]
                    input_diag["missing_coalition_masks_truncated"] = True
                    approx_reasons.append("diagnostics.input.missing_coalition_masks truncated")
            else:
                input_diag["characteristic_function_complete"] = None
                input_diag["missing_coalition_mask_count"] = None
                input_diag["missing_coalition_masks"] = None
                input_diag["missing_coalition_masks_truncated"] = None
                warnings.append("Characteristic function completeness is unavailable (game does not expose `.v`).")

            simple = bool(is_simple_game(game, tol=0.0, max_players=max_players))
            input_diag["simple_game"] = simple

            if simple:
                mono = True
                counter = None
                for S in range(max_mask + 1):
                    vS = float(game.value(S))
                    for i in range(n):
                        if S & (1 << i):
                            continue
                        T = S | (1 << i)
                        vT = float(game.value(T))
                        if vS > vT + tol:
                            mono = False
                            counter = {
                                "S": int(S),
                                "T": int(T),
                                "added_player": int(i),
                                "vS": float(vS),
                                "vT": float(vT),
                            }
                            break
                    if not mono:
                        break
                input_diag["monotone_simple_game"] = mono
                input_diag["monotone_counterexample"] = counter
                if not mono:
                    warnings.append("Game is simple but not monotone (counterexample provided).")
        else:
            input_diag["characteristic_function_complete"] = None
            input_diag["missing_coalition_mask_count"] = None
            input_diag["missing_coalition_masks"] = None
            input_diag["missing_coalition_masks_truncated"] = None
            input_diag["note"] = f"Some checks are skipped for n>{max_players}."

        diagnostics["input"] = input_diag

        if "solutions" in analysis and n <= int(max_players):
            from ..diagnostics import check_allocation, explain_core_membership

            diagnostics["solutions"] = {
                sol_id: {
                    "core": {
                        **check_allocation(
                            game,
                            sol["allocation"],
                            tol=tol,
                            core_top_k=diagnostics_top_k,
                        ).core.to_dict(),
                        "explanation": explain_core_membership(
                            game,
                            sol["allocation"],
                            tol=tol,
                            top_k=min(3, int(diagnostics_top_k)),
                        ),
                    }
                }
                for sol_id, sol in analysis["solutions"].items()
            }

        if include_lp_explanations:
            if n <= int(lp_explanations_max_players):
                try:
                    from ..diagnostics.linprog_diagnostics import build_lp_explanations

                    lp_diag = build_lp_explanations(game, tol=tol, max_list=int(diagnostics_max_list))

                    if lp_diag.get("balancedness_check", {}).get("weights_meta", {}).get("truncated"):
                        approx_reasons.append("diagnostics.lp.balancedness_check.weights truncated")

                    diagnostics["lp"] = lp_diag
                    analysis["meta"]["computed"]["lp_explanations"] = True
                except ImportError as e:
                    analysis["meta"]["computed"]["lp_explanations"] = False
                    analysis["meta"]["skipped"]["lp_explanations"] = str(e)
                except Exception as e:
                    analysis["meta"]["computed"]["lp_explanations"] = False
                    analysis["meta"]["skipped"]["lp_explanations"] = f"{type(e).__name__}: {e}"
            else:
                analysis["meta"]["computed"]["lp_explanations"] = False
                analysis["meta"]["skipped"]["lp_explanations"] = (
                    f"n={n} > lp_explanations_max_players={lp_explanations_max_players}"
                )

        analysis["diagnostics"] = diagnostics
        analysis["meta"]["computed"]["diagnostics"] = True
    else:
        analysis["meta"]["computed"]["diagnostics"] = False

    # ----------------------------
    # sets (STRICT to tucoopy-animation.schema.json)
    # ----------------------------
    if include_sets and n <= int(max_players):
        from ..geometry import (
            BargainingSet,
            Core,
            CoreCover,
            ImputationSet,
            ReasonableSet,
            mean_marginal_vector,
            weber_marginal_vectors,
        )

        sets: dict[str, Any] = {}

        imp_vertices = ImputationSet(game).vertices(tol=tol)
        imp_vertices, imp_meta = _truncate_list(list(imp_vertices), max_points)
        if imp_meta.get("truncated"):
            approx_reasons.append("sets.imputation.vertices truncated")
        sets["imputation"] = {"vertices": imp_vertices, "meta": {"computed_by": ENGINE_ID, **imp_meta}}

        core_v = Core(game).extreme_points(tol=tol, max_dim=max_players)
        core_v, core_meta = _truncate_list(list(core_v), max_points)
        if core_meta.get("truncated"):
            approx_reasons.append("sets.core.vertices truncated")
        sets["core"] = {"vertices": core_v, "meta": {"computed_by": ENGINE_ID, **core_meta}}

        # Core cover (always cheap: just bounds + efficiency)
        from ..solutions.tau import minimal_rights, utopia_payoff

        M = utopia_payoff(game)
        m = minimal_rights(game, M=M)
        cc_vertices = CoreCover(game).extreme_points(tol=tol, max_dim=max_players)
        cc_vertices, cc_meta = _truncate_list(list(cc_vertices), max_points)
        if cc_meta.get("truncated"):
            approx_reasons.append("sets.core_cover.vertices truncated")
        sets["core_cover"] = {
            "vertices": cc_vertices,
            "bounds": {"minimal_rights": [float(v) for v in m], "utopia_payoff": [float(v) for v in M]},
            "meta": {"computed_by": ENGINE_ID, **cc_meta},
        }

        # Reasonable set (bounds + efficiency)
        lower = [float(game.value(1 << i)) for i in range(n)]
        upper = [float(v) for v in utopia_payoff(game)]
        r_vertices = ReasonableSet(game).vertices(tol=tol, max_players=max_players, max_dim=max_players)
        r_vertices, r_meta = _truncate_list(list(r_vertices), max_points)
        if r_meta.get("truncated"):
            approx_reasons.append("sets.reasonable.vertices truncated")
        sets["reasonable"] = {
            "vertices": r_vertices,
            "bounds": {"imputation_lower_bounds": lower, "utopia_payoff": upper},
            "meta": {"computed_by": ENGINE_ID, **r_meta},
        }

        # Optional Weber points
        if include_weber:
            try:
                pts = weber_marginal_vectors(game, max_permutations=720)
                mu = mean_marginal_vector(game, max_permutations=720)
                pts, pts_meta = _truncate_list(list(pts), max_points)
                if pts_meta.get("truncated"):
                    approx_reasons.append("sets.weber.points truncated")
                sets["weber"] = {
                    "points": pts,
                    "mean_marginal": mu,
                    "meta": {"computed_by": ENGINE_ID, **pts_meta},
                }
            except tucoopyError as e:
                analysis["meta"]["skipped"]["sets.weber"] = str(e)

        # Optional bargaining sample points
        if include_bargaining and n <= 4:
            k = int(bargaining_n_samples)
            if k > 0:
                k = min(k, int(max_points) if max_points is not None else k)
                bs = BargainingSet(game, tol=float(tol), n_max=4)
                pts = bs.sample_points(n_samples=k, seed=sets_seed, max_attempts=bargaining_max_attempts)
                sets["bargaining"] = {
                    "points": pts,
                    "meta": {
                        "computed_by": ENGINE_ID,
                        "count_requested": int(k),
                        "count_returned": int(len(pts)),
                        "seed": int(sets_seed) if sets_seed is not None else None,
                        "max_attempts": int(bargaining_max_attempts) if bargaining_max_attempts is not None else None,
                    },
                }

        # Optional prekernel/kernel sample points (sampling-based, exponential membership checks).
        if (include_prekernel_set or include_kernel_set) and n <= 4:
            from ..geometry.kernel_set import KernelSet, PreKernelSet

            # We treat the number of samples as "work" and the max_points as "output size".
            out_cap = int(max_points) if max_points is not None else None

            if include_prekernel_set and int(prekernel_n_samples) > 0 and int(prekernel_max_points) > 0:
                try:
                    pk = PreKernelSet(game, tol=float(tol), approx_seed=sets_seed)
                    pts = pk.sample_points(
                        n_samples=int(prekernel_n_samples),
                        seed=sets_seed,
                        max_points=int(prekernel_max_points) if out_cap is None else min(int(prekernel_max_points), out_cap),
                        tol=float(tol),
                    )
                    sets["prekernel_set"] = {
                        "points": pts,
                        "meta": {
                            "computed_by": ENGINE_ID,
                            "n_samples": int(prekernel_n_samples),
                            "max_points": int(prekernel_max_points),
                            "count_returned": int(len(pts)),
                            "seed": int(sets_seed) if sets_seed is not None else None,
                        },
                    }
                except tucoopyError as e:
                    analysis["meta"]["skipped"]["sets.prekernel_set"] = str(e)

            if include_kernel_set and int(kernel_n_samples) > 0 and int(kernel_max_points) > 0:
                try:
                    ks = KernelSet(game, tol=float(tol), approx_seed=sets_seed)
                    pts = ks.sample_points(
                        n_samples=int(kernel_n_samples),
                        seed=sets_seed,
                        max_points=int(kernel_max_points) if out_cap is None else min(int(kernel_max_points), out_cap),
                        tol=float(tol),
                    )
                    sets["kernel_set"] = {
                        "points": pts,
                        "meta": {
                            "computed_by": ENGINE_ID,
                            "n_samples": int(kernel_n_samples),
                            "max_points": int(kernel_max_points),
                            "count_returned": int(len(pts)),
                            "seed": int(sets_seed) if sets_seed is not None else None,
                        },
                    }
                except tucoopyError as e:
                    analysis["meta"]["skipped"]["sets.kernel_set"] = str(e)

        analysis["sets"] = sets
        analysis["meta"]["computed"]["sets"] = True
    else:
        analysis["meta"]["computed"]["sets"] = False

    # ----------------------------
    # blocking_regions (n=3 only; future: n=4 approx)
    # ----------------------------
    if include_blocking_regions and n == 3 and n <= int(max_players):
        from ..diagnostics.blocking_regions import blocking_regions

        br = blocking_regions(game, tol=tol)
        regions_out: list[dict[str, Any]] = []
        for rgn in br.regions:
            item: dict[str, Any] = {"coalition_mask": rgn.coalition_mask, "vertices": rgn.vertices}
            if rgn.coalition_excess_at_vertices is not None:
                item["coalition_excess_at_vertices"] = rgn.coalition_excess_at_vertices
            if rgn.max_excess_at_vertices is not None:
                item["max_excess_at_vertices"] = rgn.max_excess_at_vertices
            if rgn.ties is not None:
                item["ties"] = rgn.ties
            regions_out.append(item)
        analysis["blocking_regions"] = {
            "coordinate_system": br.coordinate_system,
            "regions": regions_out,
            "meta": {"computed_by": ENGINE_ID},
        }
        analysis["meta"]["computed"]["blocking_regions"] = True
    else:
        analysis["meta"]["computed"]["blocking_regions"] = False

    # ----------------------------
    # bundle (n>max_players)
    # ----------------------------
    if include_bundle and n > int(max_players):
        grand = int(game.grand_coalition)
        vN = float(game.value(grand))
        sum_singletons = float(sum(float(game.value(1 << i)) for i in range(n)))

        game_labels = getattr(game, "player_labels", None)
        labels = list(game_labels) if game_labels is not None else [f"P{i+1}" for i in range(n)]
        bundle_max_mask: int | None = (1 << n) - 1 if n <= 30 else None
        cf_complete = False
        provided_coalitions: int | None = None
        if is_tabular_game(game):
            provided_coalitions = int(len(game.v))
            if bundle_max_mask is not None:
                cf_complete = len(game.v) == (bundle_max_mask + 1) and all(
                    int(m) in game.v for m in range(bundle_max_mask + 1)
                )
        else:
            cf_complete = False

        bundle: dict[str, Any] = {
            "game_summary": {
                "n_players": int(n),
                "vN": vN,
                "sum_singletons": sum_singletons,
                "essential": bool(vN > sum_singletons + 1e-12),
                "player_labels": labels,
                "provided_coalitions": provided_coalitions,
                "characteristic_function_complete": cf_complete if n <= int(bundle_max_players) else None,
            },
            "notes": [
                f"n={n} > max_players={max_players}: set geometry (vertices) is skipped; prefer tables/lists.",
                "To compute exact exponential solutions (e.g., Shapley) for larger n, call those solvers explicitly.",
            ],
            "meta": {"computed_by": ENGINE_ID},
        }

        if n <= int(bundle_max_players):
            tables: dict[str, Any] = {}
            tables["players"] = [{"player": int(i), "label": labels[i], "v_singleton": float(game.value(1 << i))} for i in range(n)]

            if cf_complete and bundle_include_tau:
                from ..solutions.tau import minimal_rights, utopia_payoff

                M = utopia_payoff(game)
                m = minimal_rights(game, M=M)
                tables["tau_vectors"] = {
                    "utopia_payoff": [{"player": int(i), "value": float(M[i])} for i in range(n)],
                    "minimal_rights": [{"player": int(i), "value": float(m[i])} for i in range(n)],
                }

            if cf_complete and bundle_include_power_indices:
                from ..properties import is_simple_game

                if is_simple_game(game, tol=0.0, max_players=bundle_max_players):
                    from ..power import (
                        banzhaf_index,
                        coleman_initiate_index,
                        coleman_prevent_index,
                        deegan_packel_index,
                        holler_index,
                        johnston_index,
                        rae_index,
                        shapley_shubik_index,
                    )

                    ssi = shapley_shubik_index(game)
                    bpi = banzhaf_index(game, normalized=True)
                    dpi = deegan_packel_index(game)
                    hi = holler_index(game)
                    ji = johnston_index(game)
                    cpi = coleman_prevent_index(game)
                    cii = coleman_initiate_index(game)
                    rae = rae_index(game)
                    tables["power_indices"] = [
                        {
                            "player": int(i),
                            "label": labels[i],
                            "shapley_shubik": float(ssi[i]),
                            "banzhaf": float(bpi[i]),
                            "deegan_packel": float(dpi[i]),
                            "holler": float(hi[i]),
                            "johnston": float(ji[i]),
                            "coleman_prevent": float(cpi[i]),
                            "coleman_initiate": float(cii[i]),
                            "rae": float(rae[i]),
                        }
                        for i in range(n)
                    ]
                else:
                    bundle.setdefault("notes", []).append("Power indices are included only for complete simple games.")
            elif not cf_complete:
                bundle.setdefault("notes", []).append(
                    "Bundle tables that need full v(S) are skipped because characteristic_function is incomplete."
                )

            if cf_complete and bundle_include_approx_solutions:
                from ..solutions.shapley import shapley_value_sample

                phi_hat, stderr = shapley_value_sample(game, n_samples=int(bundle_shapley_samples), seed=bundle_seed)
                tables.setdefault("approx_solutions", {})
                tables["approx_solutions"]["shapley"] = {
                    "allocation": [float(x) for x in phi_hat],
                    "stderr": [float(x) for x in stderr],
                    "meta": {
                        "computed_by": ENGINE_ID,
                        "method": "shapley_value_sample",
                        "n_samples": int(bundle_shapley_samples),
                        "seed": None if bundle_seed is None else int(bundle_seed),
                    },
                }
                approx_reasons.append("bundle.tables.approx_solutions.shapley sampled")

            bundle["tables"] = tables

        analysis["bundle"] = bundle
        analysis["meta"]["computed"]["bundle"] = True
    else:
        analysis["meta"]["computed"]["bundle"] = False

    analysis["meta"]["approx"] = bool(approx_reasons)
    analysis["meta"]["approx_reasons"] = approx_reasons

    return _to_jsonable(analysis)
