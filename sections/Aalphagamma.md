Government $i$'s war constraint vis a vis $j$ is slack in equilibrium when
$$
\hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) - \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{c} \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)^{-1} \right) \geq 0
$$
for some proposed $\hat{\tilde{\bm{\tau}}}$. The constraint is therefore slack so long as
\begin{align*}
\hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) - \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{c} \tilde{\chi}(\bm{Z}; \bm{\theta}_m)^{-1} \right) &\geq 0 \\
\tilde{\chi}(\bm{Z}; \bm{\theta}_m) &\leq \hat{c} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) \right)^{-1}
\end{align*}

Note that
$$
1 - \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m) = \frac{ m_i^\gamma }{ \rho_{ji}(\bm{W}; \bm{\theta}_m) m_j^\gamma + m_i^\gamma }
$$
which implies
$$
\frac{\tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)}{1 - \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)} =\rho_{ji}(W_{ji}; \bm{\alpha}) \left( \frac{ M_j }{ M_i } \right)^\gamma
$$.

Recall from Equation \ref{eq:rho} that
$$
\rho_{ji}(\bm{W}_{ji}; \bm{\alpha}) = e^{ -\bm{\alpha}^T \bm{W}_{ji} + \epsilon_{ji} }
$$.
We can therefore rewrite the slackness condition as
\begin{align*}
\tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m) &\leq \hat{c} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) \right)^{-1} \\
\frac{\tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)}{1 - \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)} &\leq \frac{\hat{c} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}) - \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) \right)^{-1}}{1 - \hat{c} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) \right)^{-1}} \\ 
\rho_{ji}(\bm{W}_{ji}; \bm{\alpha}) \left( \frac{ M_j }{ M_i } \right)^\gamma &\leq \frac{1}{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) \right) - 1} \\
- \bm{\alpha}^T \bm{W}_{ji} + \epsilon_{ji} + \gamma \left( \frac{ M_j }{ M_i } \right) &\leq \ln \left( \frac{1}{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) \right) - 1} \right) \\
\epsilon_{ij} &\leq \bm{\alpha}^T \bm{W}_{ji} - \gamma \left( \frac{ M_j }{ M_i } \right) + \ln \left( \frac{1}{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) \right) - 1} \right)
\end{align*}.
Let $\epsilon_{ji}^\star$ solve this with equality,^[If $\hat{c}^{-1} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) \right) < 1$ then $\epsilon_{ji}^\star$ does not exist and $i$'s war constraint vis-à-vis $j$ will never bind.]
\begin{equation} \label{eq:epsilon_star}
\epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_m) = \bm{\alpha}^T \bm{W}_{ji} - \gamma \left( \frac{ M_j }{ M_i } \right) + \ln \left( \frac{1}{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) \right) - 1} \right)
\end{equation}
With $\epsilon_{ij}$ distributed normal, the probability that the constraint is slack can be computed as 
$$
\text{Pr} \left( \epsilon_{ij} < \epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_m) \right) = \Phi \left( \frac{\epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_m)}{\sigma_{\epsilon}} \right)
$$
where $\Phi$ is the standard normal c.d.f.

Let $Y_{ji}(\hat{\tilde{\bm{\tau}}}^\star; \bm{\theta}_m) = \ln \left( \frac{1}{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) \right) - 1} \right)$. With the above quantities in hand, we can write
\begin{align*}
\E_{\hat{\tilde{\bm{\tau}}}^\star, \bm{\epsilon}} \left[ Y_{ji}(\hat{\tilde{\bm{\tau}}}^\star; \bm{\theta}_m) \right] =& \Phi \left( \frac{ \E \left[ \epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_m) \right] }{\sigma_{\epsilon}} \right) \E \left[ Y_{ji} \mid \epsilon_{ji} < \epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_m) \right] + \\
& \left( 1 - \Phi \left( \frac{ \E \left[ \epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_m) \right] }{\sigma_{\epsilon}} \right) \right) \E \left[ Y_{ji} \mid \epsilon_{ji} \geq \epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_m) \right] \\
=& \Phi \left( \frac{ \E \left[ \epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_m) \right] }{\sigma_{\epsilon}} \right) \E \left[ Y_{ji} \mid \epsilon_{ji} < \epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_m) \right] + \\
& \left( 1 - \Phi \left( \frac{ \E \left[ \epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_m) \right] }{\sigma_{\epsilon}} \right) \right) \times \\
& \left( \gamma \left( \frac{ M_j }{ M_i } \right) - \bm{\alpha}^T \bm{W}_{ji} + \E \left[ \epsilon_{ji} \mid  \epsilon_{ji} \geq \epsilon_{ji}^\star(\hat{\tilde{\bm{\tau}}}^\star, \bm{Z}; \bm{\theta}_m) \right] \right)
\end{align*}

$\E \left[ Y_{ji} \mid \epsilon_{ji} < \epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m) \right]$ can be approximated by simulating the integral
$$
\int_{-\infty}^{\epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m)} Y_{ji}(\hat{\tilde{\bm{\tau}}}^\star; \bm{\theta}_m) f(\epsilon) d \epsilon
$$
and $\E \left[ \epsilon_{ji} \mid  \epsilon_{ji} \geq \epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m) \right]$ is the mean of a truncated normal distribution. 

The sample analogue for $\E \left[ Y_{ji}(\hat{\tilde{\bm{\tau}}}^\star; \bm{\theta}_m) \right]$ is $\E \left[ Y_{ji}(\bm{1}; \bm{\theta}_m) \right]$. Then, 
\begin{equation*}
\begin{split}
\E \left[ Y_{ji}(\bm{1}; \bm{\theta}_m) \right] = \Phi \left( \frac{ \epsilon_{ji}^\star(\bm{1}, \bm{Z}; \bm{\theta}_m) }{ \sigma_{\epsilon} } \right) \E \left[ Y_{ji} \mid \epsilon_{ji} < \epsilon_{ji}^\star(\bm{1}, \bm{Z}; \bm{\theta}_m) \right] + \\
\left( 1 - \Phi \left( \frac{ \epsilon_{ji}^\star(\bm{1}, \bm{Z}; \bm{\theta}_m) }{ \sigma_{\epsilon} } \right) \right) \left( \gamma \left( \frac{ M_j }{ M_i } \right) - \bm{\alpha}^T \bm{W}_{ji} + \epsilon_{ij} \right)
\end{split}
\end{equation*}

We can now construct a loss function
\begin{equation} \label{eq:lossEpsilon}
\ell_{\epsilon}(\bm{\theta}_m) = \sum_i \sum_j \left( \E_{\hat{\tilde{\bm{\tau}}}^\star, \bm{\epsilon}} \left[ Y_{ji}(\hat{\tilde{\bm{\tau}}}^\star; \bm{\theta}_m) \right] - \E \left[ Y_{ji}(\bm{1}; \bm{\theta}_m) \right] \right)^2
\end{equation}

In practice, I minimize this loss function by iteratively recalculating the weights, $\Phi \left( \frac{ \epsilon_{ji}^\star(\bm{1}, \bm{Z}; \bm{\theta}_m) }{ \sigma_{\epsilon} } \right)$, and choosing $\gamma$ and $\bm{\alpha}$ via ordinary least squares.

In the case where the constraint holds almost surely then $\Phi \left( \frac{\epsilon_{ij}^\star(\bm{1}, \bm{Z}; \bm{\theta}_m)}{\sigma_{\epsilon}} \right) \approx 0$ and this resembles a tobit regression of $Y_{ji}(\bm{1}; \bm{\theta}_m)$ on the military capability ratio and dyadic geograpy. If $Y_{ji}(\bm{1}; \bm{\theta}_m)$ varies positively with the capability ratio, this indicates higher returns to military expenditure, corresponding to a larger $\gamma$.