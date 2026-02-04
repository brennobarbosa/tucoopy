from __future__ import annotations

"""
Kernel and prekernel (optional fast backend).

Warning
-------
This example requires NumPy at runtime. Install with:
`pip install \"tucoop[fast]\"`.
"""

from _bootstrap import add_src_to_path

add_src_to_path()


def main() -> None:
    try:
        import numpy  # noqa: F401
    except Exception:
        print("This example requires NumPy. Install with: pip install \"tucoop[fast]\"")
        return

    from tucoop.base.game import Game
    from tucoop.solutions.kernel import kernel, prekernel
    from tucoop.solutions.shapley import shapley_value

    # Additive game: v(S)=|S|. Unique imputation/core point is (1,1,1).
    g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))

    print("Shapley:", shapley_value(g))

    pk = prekernel(g, tol=1e-9, max_iter=50)
    print("Prekernel:", pk.x, "residual:", pk.residual, "iters:", pk.iterations)

    k = kernel(g, tol=1e-9, max_iter=50)
    print("Kernel:", k.x, "residual:", k.residual, "iters:", k.iterations, "active_bounds:", sorted(k.active_bounds))


if __name__ == "__main__":
    main()
