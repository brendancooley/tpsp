Government $i$'s war constraint vis a vis $j$ is slack when
$$
\hat{G}_j(\hat{\tilde{\bm{\tau}}}; b_j) - \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{c} \chi_{ji}(\bm{Z}; \bm{\theta}_m)^{-1} \right) \geq 0
$$
for some proposed $\hat{\tilde{\bm{\tau}}}$. In the data, $\hat{\tilde{\bm{\tau}}}^\star = \bm{1} \implies \hat{G}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) = 1$. The constraint is therefore empirically slack so long as
\begin{align*}
1 - \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{c} \chi_{ji}(\bm{Z}; \bm{\theta}_m)^{-1} \right) &\geq 0 \\
\chi_{ji}(\bm{Z}; \bm{\theta}_m) &\leq \hat{c} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - 1 \right)^{-1}
\end{align*}

Note that
$$
1 - \chi_{ji}(\bm{Z}; \bm{\theta}_m) = \frac{ m_i^\gamma }{ \rho_{ji}(\bm{W}; \bm{\theta}_m) m_j^\gamma + m_i^\gamma }
$$
which implies
$$
\frac{\chi_{ji}(\bm{Z}; \bm{\theta}_m)}{1 - \chi_{ji}(\bm{Z}; \bm{\theta}_m)} =\rho_{ji}(W_{ji}; \bm{\alpha}) \left( \frac{ m_{ji} }{ m_{ii} } \right)^\gamma
$$.

Recall from Equation \ref{eq:rho} that
$$
\rho_{ji}(\bm{W}_{ji}; \bm{\alpha}) = e^{ -\bm{\alpha}^T \bm{W}_{ji} + \epsilon_{ji} }
$$.
We can therefore rewrite the empirical slackness condition as
\begin{align*}
\chi_{ji}(\bm{Z}; \bm{\theta}_m) &\leq \hat{c} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - 1 \right)^{-1} \\
\frac{\chi_{ji}(\bm{Z}; \bm{\theta}_m)}{1 - \chi_{ji}(\bm{Z}; \bm{\theta}_m)} &\leq \frac{\hat{c} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}) - 1 \right)^{-1}}{1 - \hat{c} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - 1 \right)^{-1}} \\ 
\rho_{ji}(\bm{W}_{ji}; \bm{\alpha}) \left( \frac{ m_{ji} }{ m_{ii} } \right)^\gamma &\leq \frac{1}{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - 1 \right) - 1} \\
- \bm{\alpha}^T \bm{W}_{ji} + \epsilon_{ji} + \gamma \left( \frac{ m_{ji} }{ m_{ii} } \right) &\leq \ln \left( \frac{1}{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - 1 \right) - 1} \right) \\
\epsilon_{ij} &\leq \bm{\alpha}^T \bm{W}_{ji} - \gamma \left( \frac{ m_{ji} }{ m_{ii} } \right) + \ln \left( \frac{1}{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - 1 \right) - 1} \right)
\end{align*}.
Let $\epsilon_{ji}^\star$ solve this with equality,
\begin{equation} \label{eq:epsilon_star}
\epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m) = \bm{\alpha}^T \bm{W}_{ji} - \gamma \left( \frac{ m_{ji} }{ m_{ii} } \right) + \ln \left( \frac{1}{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - 1 \right) - 1} \right)
\end{equation}
With $\epsilon_{ij}$ distributed normal, the probability that the constraint is slack can be computed as 
$$
\text{Pr} \left( \epsilon_{ij} < \epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m) \right) = \Phi \left( \frac{\epsilon_{ij}(\bm{Z}; \bm{\theta}_m)}{\sigma_{\epsilon}} \right)
$$
where $\Phi$ is the standard normal c.d.f.

Let $\tilde{Y}_{ji}(\hat{c}, b_j) = \ln \left( \frac{1}{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - 1 \right) - 1} \right)$. With the above quantities in hand, we can write
\begin{align*}
\E_{\epsilon} \left[ \tilde{Y}_{ji} \right] =& \Phi \left( \frac{\epsilon_{ij}^\star(\bm{Z}; \bm{\theta}_m)}{\sigma_{\epsilon}} \right) \E \left[ \tilde{Y}_{ji} | \epsilon_{ji} < \epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m) \right] + \left( 1 - \Phi \left( \frac{\epsilon_{ij}^\star(\bm{Z}; \bm{\theta}_m)}{\sigma_{\epsilon}} \right) \right) \E \left[ \tilde{Y}_{ji} | \epsilon_{ji} \geq \epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m) \right] \\
=& \Phi \left( \frac{\epsilon_{ij}^\star(\bm{Z}; \bm{\theta}_m)}{\sigma_{\epsilon}} \right) \E \left[ \tilde{Y}_{ji} | \epsilon_{ji} < \epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m) \right] + \\
& \left( 1 - \Phi \left( \frac{\epsilon_{ij}^\star(\bm{Z}; \bm{\theta}_m)}{\sigma_{\epsilon}} \right) \right) \left( \gamma \left( \frac{ m_{ji} }{ m_{ii} } \right) - \bm{\alpha}^T \bm{W}_{ji} + \E \left[ \epsilon_{ji} |  \epsilon_{ji} \geq \epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m) \right] \right)
\end{align*}

Replacing $\E_{\epsilon} \left[ \tilde{Y}_{ji} \right]$ and $\E \left[ \tilde{Y}_{ji} | \epsilon_{ji} < \epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m) \right]$ with their simulated sample analogues, we have
$$
\tilde{Y}_{ji}(\hat{c}, b_j) - \E \left[ \epsilon_{ji} |  \epsilon_{ji} \geq \epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m) \right]
$$