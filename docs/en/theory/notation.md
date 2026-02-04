# Notation and conventions

This section summarizes the notation used in the theory pages and in the library.
We follow standard notation for **transferable-utility (TU) cooperative game theory**, with a few small conventions chosen for clarity and ease of implementation.

The goal is not to introduce new concepts, but to establish a common language so that definitions, algorithms, and outputs can be interpreted consistently.

## Players and coalitions

- **Players** are indexed by the finite set $N = \{1, \ldots, n\}$.
- A **coalition** is any subset $S \subseteq N$.
- The **grand coalition** is the set of all players, denoted by $N$ itself.

In the code, players are indexed from `0` to `n-1`, following standard Python conventions.
Coalitions are represented internally as **bitmasks**, but most user-facing functions accept Python iterables (lists, tuples, or sets of player indices).

!!! tip "Intuition"
    A coalition is simply a group of players acting together.
    The grand coalition represents full cooperation among all players.

## Characteristic function

A TU cooperative game is described by a **characteristic function**

$$
v : 2^N \to \mathbb{R},
$$

which assigns a real value to each coalition, with the normalization

$$
v(\emptyset) = 0.
$$

The value $v(S)$ represents the total value that coalition $S$ can generate on its own, assuming its members cooperate fully and can transfer utility freely among themselves.

!!! tip "Intuition"
    Think of $v(S)$ as the "*size of the pie*" available to coalition $S$. How that pie is divided comes later.

## Allocations and efficiency

!!! note "Definition"
    An **allocation** is a vector

    $$
    x = (x_1, \dots, x_n) \in \mathbb{R}^n,
    $$

    where $x_i$ denotes the payoff assigned to player $i$.

    An allocation is **efficient** if

    $$
    \sum_{i \in N} x_i = v(N).
    $$

!!! tip "Intuition"
    Efficiency means that all value created by full cooperation is distributed among the players.
    Nothing is lost and nothing is left undistributed.

??? example "Example"
    For an additive game defined by $v(S) = |S|$ with $n=3$, the grand coalition has value

    $$
    v(N) = 3.
    $$

    A natural efficient allocation is

    $$
    x = (1, 1, 1),
    $$

    where each player receives exactly their standalone contribution.

## Coalition sums and excess

!!! note "Definition"
    Given an allocation $x$ and a coalition $S \subseteq N$, the **coalition sum** is

    $$
    x(S) = \sum_{i \in S} x_i.
    $$

    The **excess** of coalition $S$ at allocation $x$ is defined as

    $$
    e(S, x) = v(S) - x(S).
    $$

!!! tip "Intuition"
    The excess measures how dissatisfied a coalition is.

    - If $e(S, x) > 0$, coalition $S$ can do better on its own than under allocation $x$.
    - If $e(S, x) = 0$, the coalition is exactly satisfied.
    - If $e(S, x) < 0$, the coalition receives more than its standalone value.

The concept of excess is central in many solution concepts, especially those related to **stability**, such as the core, the $\epsilon$-core, and the nucleolus.

