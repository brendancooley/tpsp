Government $i$'s constraint vis a vis $j$ binds when
$$
\hat{G}_j(\hat{\tilde{\tau}}) - \left( \hat{G}_j(\hat{\tau}_i^{j\star}) - \hat{c} \chi_{ji}(\bm{m}; \bm{\alpha}) \right) = 0
$$
In the data, $\hat{\tilde{\tau}} = 1 \implies \hat{G}_j(\hat{\tilde{\tau}}) = 1$. Then,
\begin{align*}
1 - \left( \hat{G}_j(\hat{\tau}_i^{j\star}) - \hat{c} \left( \chi_{ji}(\bm{m}; \bm{\alpha}) \right)^{-1} \right) &= 0 \\
\left( \chi_{ji}(\bm{m}; \bm{\alpha}) \right)^{-1} &= \hat{c}^{-1} \left( \hat{G}_j(\hat{\tau}_i^{j\star}) - 1 \right) \\
\chi_{ji}(\bm{m}; \bm{\alpha}) &= \hat{c} \left( \hat{G}_j(\hat{\tau}_i^{j\star}) - 1 \right)^{-1}
\end{align*}

Note that 
\begin{align*}
1 - \chi_{ji}(\bm{m}; \bm{\alpha}) &= 1 - \frac{ \rho_{ji}(\bm{\alpha}) m_{ji} }{ \rho_{ji}(\bm{\alpha}) m_{ji} + m_{ii} } \\
&= \frac{ { \rho_{ji}(\bm{\alpha}) m_{ji} + m_{ii} } }{ { \rho_{ji}(\bm{\alpha}) m_{ji} + m_{ii} } } - \frac{ \rho_{ji}(\bm{\alpha}) m_{ji} }{ \rho_{ji}(\bm{\alpha}) m_{ji} + m_{ii} } \\
&= \frac{ m_{ii} }{ \rho_{ji}(\bm{\alpha}) m_{ji} + m_{ii} }
\end{align*}

Applying the logit transformation, 
$$
\frac{\chi_{ji}(\bm{\alpha}, \bm{W}_{ij}, \epsilon_{ij})}{1 - \chi_{ji}(\bm{\alpha}, \bm{W}_{ij}, \epsilon_{ij})} = \rho_{ji}(\bm{\alpha}) \frac{ m_{ji} }{ m_{ii} }
$$

Recall that 
$$
\rho_{ji}(\bm{\alpha}) = e^{ -\bm{\alpha}^T W_{ji} + \epsilon_{ji} }
$$

Returning to the constraint
\begin{align*}
\chi_{ji}(\bm{m}; \bm{\alpha}) &= \hat{c} \left( \hat{G}_j(\hat{\tau}_i^{j\star}) - 1 \right)^{-1} \\
\frac{ \chi_{ji}(\bm{m}; \bm{\alpha}) }{ 1 - \chi_{ji}(\bm{m}; \bm{\alpha}) } &= \frac{\hat{c} \left( \hat{G}_j(\hat{\tau}_i^{j\star}) - 1 \right)^{-1}}{1 - \hat{c} \left( \hat{G}_j(\hat{\tau}_i^{j\star}) - 1 \right)^{-1}} \\
\rho_{ji}(\bm{\alpha}) \frac{ m_{ji} }{ m_{ii} } &= \frac{ 1 }{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\tau}_i^{j\star}) - 1 \right) } \\
- \bm{\alpha}^T W_{ji} + \epsilon_{ji} + \ln \left( \frac{ m_{ji} }{ m_{ii} } \right) &= \ln \left( \frac{ 1 }{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\tau}_i^{j\star}) - 1 \right) } \right) \\
\epsilon_{ji}^\star &= \bm{\alpha}^T W_{ji} - \ln \left( \frac{ m_{ji} }{ m_{ii} } \right) + \ln \left( \frac{ 1 }{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\tau}_i^{j\star}) - 1 \right) } \right)
\end{align*}