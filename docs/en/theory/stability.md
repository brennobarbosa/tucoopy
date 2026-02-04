# Stability: blocking and the core

So far, we described the imputation set: divisions that use all the value and guarantee that nobody is worse off than acting alone.

But one essential question remains.

Even if a division looks reasonable, is it **sustainable**?

That is:

> Is there some group of players that would prefer to leave the grand coalition and make a deal on its own?

This is the central idea behind **stability** in cooperative games.

---

## The idea of "deviation" in cooperative games

Consider an imputation $x \in \mathbb{R}^n$.
It says how much each player receives when everyone cooperates.

Now take a coalition $S \subseteq N$. This group knows that, if it breaks away, it can generate $v(S)$.

The question is: can they split $v(S)$ **among themselves** in a way that makes **everyone in $S$ strictly better** than under $x$?

If yes, then $x$ is not sustainable: that group has an incentive to break the agreement.

---

## Total payoff of a coalition

Given a vector $x$, we write the total payoff that $x$ gives to a coalition $S$ as

$$
x(S) := \sum_{i\in S} x_i.
$$

This notation is useful because the coalition compares "what it gets under the current agreement" with "what it can guarantee on its own".

---

## Blocking

We say a coalition $S$ can **block** an imputation $x$ if it can secure enough value to improve the situation of everyone within $S$.

A simple (and widely used) sufficient condition for that is:

$$
v(S) > x(S).
$$

!!! tip "Interpretation"
    Group $S$ can generate more than it is receiving under the current agreement, so it has "slack" to propose an alternative deal that benefits its members.

> If there exists some $S$ that blocks $x$, then $x$ is not stable: a coalitional deviation is plausible.

---

## The core

The **core** is the set of imputations that **cannot be blocked by any coalition**.

In practical terms: these are divisions where *no one*, in any group, has an incentive to abandon the agreement.

Mathematically, the core is the set of allocations $x$ such that:

$$
\sum_{i \in N} x_i = v(N)
\quad\text{and}\quad
x(S) \ge v(S)\; \text{for all } S \subseteq N.
$$

The condition $x(S) \ge v(S)$ says:

> "any coalition $S$ already receives at least what it could guarantee on its own".

---

## A geometric view (the 3-player case)

When $n=3$, the imputation set is a triangle (as you saw earlier).
Each constraint of the form

$$
x(S) \ge v(S)
$$

becomes a **half-plane** cutting that triangle.

The core is simply the **intersection** of all those cuts.

That is why it is often a smaller polygon[^1] inside the triangle -- and sometimes... it may not exist at all.

[^1]: In higher dimensions we call it a polytope: an intersection of half-spaces.

---

## An important fact: the core can be empty

Even when the grand coalition generates a lot of value, it can happen that there is no imputation that simultaneously satisfies all constraints $x(S) \ge v(S)$.

Intuitively, this happens when "partial" coalitions are too strong: there is always some group that can demand more than the current agreement can offer without violating another constraint.

When the core is empty, we need other, more flexible notions of stability -- such as the **least-core** and the **nucleolus**.

(And this is where the theory starts getting really interesting.)

