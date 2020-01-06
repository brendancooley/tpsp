The estimation algorithm iteratively grid searches over $\bm{b}$ and $\hat{c}$ to minimize $\ell_{\bm{\tau}}$ and chooses $\bm{alpha}$ and $\gamma$ to minimizes $\ell_{\epsilon}$ via ordinary least squares. First, as a normalization, I set $\sigma_{\epsilon}^2 = 1$. Then, I populate a set $\mathcal{C} = \left\{ \hat{c}_{\ell} \right\}$ of trial values for $\hat{c}$. Then, for each $\hat{c}_{\ell} \in \mathcal{C}$:

1. Initialize $k=1$ and set starting values, $\bm{\theta}_k$. Let $\Delta(\bm{\theta}_{m,k}, \bm{\theta}_{m, k-1})$ be a loss function that measures the distance between any pair of parameter estimates.
2. While $\Delta(\bm{\theta}_{m,k}, \bm{\theta}_{m, k-1}) < \delta$ (where $\delta$ is some convergence threshold):
    - Search over a grid of values for $\bm{b}$ and set
$$
\bm{b}_k = \argmin_{\bm{b}} \ell_{\bm{\tau}}(\bm{\theta}_{m, k})
$$
    - Draw many $\bm{\epsilon}$ and calculate $\E \left[ Y_{ji} \mid \epsilon_{ji} < \epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_{m,k}) \right]$ by calculating best responses $\hat{\tilde{\bm{\tau}}}_i^\star(\bm{\epsilon})$. 
    - Calculate initial $\epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_{m, k})$
    - Choose $\gamma_k$ and $\bm{\alpha}_k$ to minimize $\ell_{\epsilon}(\bm{\theta}_{m, k})$ via least squares, recompute $\epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_{m, k})$, and iterate until convergence. Update values of $\bm{\theta}_{m, k}$.
    - $k = k + 1$
3. Set $\bm{\theta}_m(\hat{c}_{\ell}) = \left( \bm{b}_k, \bm{\alpha}_k, \gamma_k, \hat{c}_{\ell}, \sigma_{\epsilon}^2 \right)$

This generates a sequence $\left\{ \bm{\theta}_m(\hat{c}_{\ell}) \right\}_{ \hat{c}_{\ell} \in \mathcal{C} }$ from which I choose estimates that satisfy
$$
\tilde{\bm{\theta}}_m \in \argmin_{ \hat{c}_{\ell} \in \mathcal{C} } \quad \ell_{\bm{\tau}}(\bm{\theta}_m(\hat{c}_{\ell}))
$$
