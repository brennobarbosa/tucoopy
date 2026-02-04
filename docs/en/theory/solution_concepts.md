# Conceitos de solução (visão geral)

Esta biblioteca implementa vários conceitos clássicos de solução. Em alto nível:

- Valores (seleções de ponto único): valor de Shapley, valor de Banzhaf, valor τ.
- Seleções via LP (SciPy opcional): least-core, nucleolus, pré-nucleolus.
- Seleções por complementaridade (NumPy opcional): kernel, prekernel.


## Valor de Shapley

!!! note "Definição"
    O valor de Shapley atribui a cada jogador $i$ a contribuição marginal média sobre todas as permutações:

    $$\varphi_i(v) = \sum_{S \subseteq N\setminus\{i\}} \frac{|S|!(n-|S|-1)!}{n!}\,\bigl(v(S\cup\{i\})-v(S)\bigr).$$

!!! tip "Intuição"
    Cada ordem de chegada dos jogadores é igualmente provável; o pagamento de um jogador é sua contribuição marginal esperada.
