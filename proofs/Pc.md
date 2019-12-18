Take a candidate peace-inducing policy announcement $\tilde{\bm{\tau}}^\star$. Peace and condition \ref{eq:BRa} require that for any action profile $\bm{a}_{i}$ with $a_{ij} = 1$ for some $j \neq i$
$$
G_i(\tilde{\bm{\tau}}^\star) \geq \sum_{\varphi_i \in \Phi} \text{Pr}(\varphi_i; \bm{a}_i) G_i \left( \bm{\tau}^\varphi_i(\tilde{\bm{\tau}}^\star) \right) - \sum_{j} a_{ij} c_i
$$
for all governments $i$. This condition can be rewritten
$$
c_i \geq \max_{ \bm{a}_i } \underbrace{\frac{1}{ \sum_{j \neq i} a_{ij} } \left( \sum_{\varphi_i \in \Phi} \text{Pr}(\varphi_i; \bm{a}_i) G_i \left( \bm{\tau}^\varphi_i(\tilde{\bm{\tau}}^\star) \right) - G_i(\tilde{\bm{\tau}}^\star) \right)}_{A(\tilde{\bm{\tau}}^\star; \bm{a}_i)}
$$
Note that $G_i$ is a continuous function mapping $[1, \bm{\tau}]^{N \times N}$ (compact) to $\mathbb{R}$. $A(\tilde{\bm{\tau}}; \bm{a}_i)$ is a linear combination of $G_i$s and is therefore itself continuous and also maps $[1, \bm{\tau}]^{N \times N}$ to $\mathbb{R}$. Then, by Weierstrass' Theorem, $\max_{ \bm{a}_i } A(\tilde{\bm{\tau}}^\star; \bm{a}_i)$ is finite. Let $c_i^{\bm{a}} = \max_{ \bm{a}_i } A(\tilde{\bm{\tau}}^\star; \bm{a}_i)$ and 
$$
c^{\bm{a}} = \left\{ c_i^{\bm{a}} \right\}_{i \in \left\{1, ..., N \right\} }
$$
Now, condition \ref{eq:BRtau} and peace requires 
$$
G_j^{\bm{\tau}}(\tilde{\bm{\tau}}_j^\star; \tilde{\bm{\tau}}_{-j}^\star) \geq \sum_{\varphi^j \in \Phi} \text{Pr} \left( \varphi^j; \bm{a}^\star(\tilde{\bm{\tau}}_j^\prime) \right) G_j \left( \bm{\tau}^{\varphi^j}(\tilde{\bm{\tau}}^\prime) \right) - \sum_i a_{ij}^\star(\tilde{\bm{\tau}}_j^\prime; \tilde{\bm{\tau}}_j^\star) c_j
$$
for all alternative policies $\tilde{\bm{\tau}}_j^\prime$ with $a_{ij}^\star(\tilde{\bm{\tau}}_j^\prime; \tilde{\bm{\tau}}_j^\star) = 1$ for some $i \neq j$. Alternatively,
$$
c_j \geq \max_{\tilde{\bm{\tau}}_j^\prime} \underbrace{\frac{1}{ \sum_{i \neq j} a_{ij}^\star(\tilde{\bm{\tau}}_j^\prime; \tilde{\bm{\tau}}_j^\star) } \left( \sum_{\varphi^j \in \Phi} \text{Pr} \left( \varphi^j; \bm{a}^\star(\tilde{\bm{\tau}}_j^\prime) \right) G_j \left( \bm{\tau}^{\varphi^j}(\tilde{\bm{\tau}}^\prime) - G_j^{\bm{\tau}}(\tilde{\bm{\tau}}_j^\star; \tilde{\bm{\tau}}_{-j}^\star) \right) \right)}_{B(\tilde{\bm{\tau}}_j^\prime; \bm{\tau}^\star)}
$$
By the same argument made above, $\max_{\tilde{\bm{\tau}}_j^\prime} B(\tilde{\bm{\tau}}_j^\prime; \bm{\tau}^\star)$ is finite. Let $c_j^{\bm{\tau}} = \max_{\tilde{\bm{\tau}}_j^\prime} B(\tilde{\bm{\tau}}_j^\prime; \bm{\tau}^\star)$ and $$
c^{\bm{\tau}} = \left\{ c_j^{\bm{\tau}} \right\}_{ j \in \left\{ 1, ..., N \right\} }
$$
Finally, let $c^\star = \max \left\{ c^{\bm{\tau}}, c^{\bm{a}} \right\}$. Since each element of this set is finite, $c^\star$ is itself finite. It is then immediate that all $c \geq c^\star$ satisfy the conditions for a peaceful subgame equilibrium of $\Gamma^{\bm{\tau}}(\bm{c})$ given in Definition `r DGammac`. $\blacksquare$