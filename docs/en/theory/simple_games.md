# Simple games and weighted voting

A *simple game* typically has $v(S) \in \{0,1\}$ for every coalition $S$.

Weighted voting games are a common subclass:

- Given weights $w_i$ and a quota $q$, coalition $S$ is winning if $\sum_{i \in S} w_i \ge q$.

Power indices:

- Shapley-Shubik index
- (Normalized) Banzhaf index

In the code, we validate the "simple game" assumption before computing these indices.

## Definition (simple game)

!!! note "Definition"
    A simple game satisfies $v(S) \in \{0,1\}$ for every coalition $S$.
    In general, $v(S)=1$ means "winning" and $v(S)=0$ means "losing".

!!! tip "Intuition"
    Only the yes/no outcome matters; utilities are not cardinal beyond that.

