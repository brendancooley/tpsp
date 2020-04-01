Assumption: war results in imposition of $v_i = 1$, $\bm{\tau}_i = \bm{1}$ $\implies \hat{G}_i^\prime = 0$

War costs are specific to the directed dyad ($\hat{c}_{ji}$ is $j$'s relative cost for replacing $i$) and is a realization of a random varible from a known aggressor-specific distribution. The shape of the distribution depends on an aggressor-specific cost shifter, as well as the military balance and loss of strength gradient. These are held as private information to the aggressor. Government $j$ prefers not to attack $i$ so long as
\begin{align*}
\hat{G}_j(\hat{\bm{\tau}}_i^\prime; \hat{\bm{h}}_i^\prime) - \hat{c}_{ji} &\leq \hat{G}_j(\hat{\tilde{\bm{\tau}}}; \hat{\tilde{\bm{h}}}) \\
\hat{c}_{ji}^{-1} &\leq \left( \hat{G}_j(\hat{\bm{\tau}}_i^\prime; \hat{\bm{h}}_i^\prime) - \hat{G}_j(\hat{\tilde{\bm{\tau}}}; \hat{\tilde{\bm{h}}}) \right)^{-1}
\end{align*}

Let inverse war costs be distributed Frechét,
$$
\text{Pr}\left( \frac{1}{\hat{c}_{ji}} \leq \frac{1}{\hat{c}} \right) = F_j \left( \frac{1}{\hat{c}} \right) = \exp \left( -\frac{1}{C_j} \left( \frac{m_j}{m_i} \right)^{\gamma} Z_{ji}^{-\alpha} \hat{c}^{\eta} \right)
$$
where $C_j$ is an aggressor-specific cost shifter, $\frac{m_j}{m_i}$ is the relative military balance (elasticity: $\gamma$, $Z_{ji}$ is the geographic distance between $j$ and $i$ (elasticity: $\alpha$), and $\eta$ is a global shape parameter.^[Higher $\eta$ correspond to more concentrated cost draws.]

From this, we can calculate the probability that no government finds it profitable to attack $i$, which is given by
\begin{align*}
H_i(\hat{\tilde{\bm{\tau}}}; \bm{Z}, \bm{\theta}_m) &= \prod_{j \neq i} F_j \left( \left( \hat{G}_j(\hat{\bm{\tau}}_i^\prime; \hat{\bm{h}}_i^\prime) - \hat{G}_j(\hat{\tilde{\bm{\tau}}}; \hat{\tilde{\bm{h}}}) \right)^{-1} \right) \\
&= \exp \left( - \sum_{j \neq i} - \frac{1}{C_j} \left( \frac{m_j}{m_i} \right)^{\gamma} Z_{ji}^{-\alpha} \left( \hat{G}_j(\hat{\bm{\tau}}_i^\prime; \hat{\bm{h}}_i^\prime) - \hat{G}_j(\hat{\tilde{\bm{\tau}}}; \hat{\tilde{\bm{h}}}) \right)^{\eta} \right)
\end{align*}

A prospective policy-chooser then confronts the following objective function
$$
\hat{\tilde{G}}_i \left( \hat{\tilde{\bm{\tau}}}, \hat{\tilde{\bm{h}}} \right) = H_i(\hat{\tilde{\bm{\tau}}}, \hat{\tilde{\bm{h}}}; \bm{Z}, \bm{\theta}_m) \hat{G}_i \left( \hat{\tilde{\bm{\tau}}}, \hat{\tilde{\bm{h}}} \right)
$$
where implicitly $i$'s utility is zero with probability $1 - H_i(\hat{\tilde{\bm{\tau}}}, \hat{\tilde{\bm{h}}}; \bm{Z}, \bm{\theta}_m)$. We solve this best response by imposing the constraints
$$
\hat{\tilde{\bm{h}}} = \hat{\bm{h}}(\hat{\tilde{\bm{\tau}}})
$$
(equilibrium ge constraints) and
$$
\hat{\bm{h}}_i^\prime = \hat{\bm{h}}(\bm{1}_i; \hat{\tilde{\bm{\tau}}}_{-i})
$$
(conquest ge constraints)