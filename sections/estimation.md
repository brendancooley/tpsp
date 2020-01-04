Solving the economy in changes for a set of $\hat{\bm{\tau}}$ requires values for a vector of economic parameters $\bm{\theta}_h$ and data on trade flows, policy barriers, and and national accounts. I discuss how I calibrate the economy in Appendix `r AeconomyC`. With $\hat{h}(\hat{\bm{\tau}}; \bm{\theta}_h)$ calibrated, $\hat{G}_i(\hat{\bm{\tau}})$ can be calculated for any set of trade policies and the optimal policy change problem (\ref{eq:optTaujHat}) can be solved, yielding $\hat{\bm{\tau}}_i^{j \star}$ for all $i, j$. I can then focus attention on $\hat{\Gamma}^{\bm{\tau}}$. The equilibrium of this game depends on a vector of parameters $\bm{\theta}_m = \left\{ \bm{b}, \bm{\alpha}, \gamma, \hat{c}, \sigma_{\epsilon}^2 \right\}$, as well as the values of the shocks $\left\{ \xi_{ij} \right\}_{i,j \in \left\{ 1, \dots, N \right\}}$ and $\left\{ \epsilon_{ij} \right\}_{i,j \in \left\{ 1, \dots, N \right\}}$. Because I work with an equilibrium in changes, a prediction $\hat{\tilde{\tau}}_{ij}^\star = 1$ is consistent with the data -- the model predicts that in equilibrium, government $i$ would make no changes to its factual trade policy toward $j$.

Estimation leverages Assumptions `r Axi` and `r Aepsilon`. First, note that Assumption `r Axi` implies
\begin{equation*}
\begin{split}
\E \left[ \hat{\tilde{\bm{\tau}}}_i^\star(\bm{1}; \bm{\theta}_m) \right] = \argmax_{ \hat{\tilde{\bm{\tau}}}_i } & \quad \hat{G}_i(\hat{\tilde{\bm{\tau}}}_i; \bm{1}) \\
\text{subject to} & \quad \hat{G}_j(\hat{\tilde{\bm{\tau}}}) - \hat{G}_j(\hat{\bm{\tau}}_i^{j \star}) + \hat{c} \E_{\epsilon} \left[ \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)^{-1} \right] \geq 0 \quad \text{for all } j \neq i
\end{split}
\end{equation*}
where the affinity shocks drop out of the maximand. For any guess of the parameter vector $\bm{\theta}_m$, we can calculate $\E \left[ \hat{\tilde{\bm{\tau}}}_i^\star(\bm{1}; \bm{\theta}_m) \right]$ for each $i$. In practice, I compute this quantity by recasting \ref{eq:tauTildeStarHat} as a mathematical program with equilibrium constraints, which I discuss further in Appendix `r Ampec`. From here, we can construct the loss function
$$
\ell_{\bm{\tau}}(\bm{\theta}_m) = \sum_i \sum_j \left| \ln \left( \E \left[ \hat{\tilde{\tau}}_{ij}^\star(\bm{1}; \bm{\theta}_m) \right] \right) \right|
$$
where the theoretical prediction is implicitly divided by the empirical policy $\hat{\tilde{\tau}}_{ij}^\star = 1$. Absolute log loss is a natural loss function to use in this setting. If $\hat{\tilde{\tau}}_{ij}^\star = 1 / 2$ then $i$'s best response features a tariff on imports from $j$ that is half its facutal value. If $\hat{\tilde{\tau}}_{ij}^\star = 2$ then this tariff is twice it's factual value. The considered loss function penalizes each of these deviations equally.

The war constraints are informative about the value of the parameters $\bm{\alpha}$ and $\gamma$. Consider the case in which the $i$'s war constraint vis-à-vis $j$ holds with certainty. Then,
\begin{align*}
\hat{G}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) - \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{c} \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)^{-1} \right) &= 0 \\
1 - \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{c} \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)^{-1} \right) &= 0
\end{align*}
which can be rearranged as
\begin{equation} \label{eq:constraintRegression}
\ln \left( \frac{ 1 }{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\tau}_i^{j\star}) - 1 \right) - 1 } \right) = \gamma \ln \left( \frac{ M_j }{ M_i } \right) - \bm{\alpha}^T W_{ji} + \epsilon_{ji}
\end{equation}
. The left side is a measure of the difference between $j$'s factual utility difference and it's regime change value ($\hat{G}_j(\hat{\tau}_i^{j\star}) - 1$). The extent to which this correlates with $j$'s military advantage over $i$ ($M_j / M_i$) is informative about the returns to military power, $\gamma$. The extent to which this correlates with dyadic geography is informative about the power projection parameters $\bm{\alpha}$.

In Appendix `r Aalphagamma` I show that if the regime change value is sufficiently high, there exists an $\epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m)$ such that for all $\epsilon_{ji} \geq \epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m)$, $i$'s constraint vis-à-vis $j$ holds. Given $\sigma_{\epsilon}$, I can then construct a stochastic variant of \ref{eq:constraintRegression} which can be used as a second moment condition. 


epsilon tilde?

Variation in policies unexplained by preference parameters $b_i$ is informative about the vector of power projection parameters $\bm{\alpha}$ and $\gamma$. Note that the Legrangian corresponding to the governments' constrained policy problem (\ref{eq:tauTildeStarHat}) function is a weighted average of the governments own utility and others' utility, where the weights are given by the Legrange multipliers. Constraints on policy choice are more likely to bind when a threatening government $j$ a) has a larger military allocation ($m_{ji}$ high) and b) when power projection costs are lower ($\rho_{ij}$ high). Therefore, the extent to which $i$'s policy choices favor government $j$ helps pin down power projection parameters. 


...

Let $\mathcal{L}_i^{\hat{\bm{x}}}(\hat{\bm{x}}_i; \bm{m}, \bm{\lambda}_i^{\bm{\chi}}, \bm{\lambda}_i^{\bm{w}} \bm{\theta}_m)$ denote the associated Lagrangian, where $\bm{\lambda}_i^{\bm{\chi}}$ are multipliers associated with the war constraints and $\bm{\lambda}_i^{\bm{w}}$ are multipliers associated with the economic equilibrium constraints. 

Examining this problem allows us to characterize how other governments' utilities change as a function of their military strategies. The Karush-Kuhn-Tucker (KKT) optimality conditions of this problem require
$$
\lambda_{ij}^{\bm{\chi}} \left( \hat{G}_j(\hat{\bm{w}}; \bm{\theta}_m) - \hat{G}_j\left( \hat{\bm{\tau}}_i^{j \star}(\bm{\theta}_m) \right) + \hat{c} \left( \chi_{ji}(\bm{m}, \bm{\theta}_m) \right)^{-1} \right) = 0
$$
or that
$$
\hat{G}_j(\hat{\bm{w}}; \bm{\theta}_m) = \hat{G}_j \left( \hat{\bm{\tau}}_i^{j \star}(\bm{\theta}_m) \right) - \hat{c} \left( \chi_{ji}(\bm{m}, \bm{\theta}_m) \right)^{-1} \quad \text{if } \lambda_{ij}^{\bm{\chi}} > 0
$$
Differentiating this condition with respect to $m_{j, i \neq j}$ ($j$'s military effort dedicated toward war with $i$) gives
$$
\frac{\partial \hat{G}_j}{\partial m_{j, i \neq j}} = \begin{cases}
\frac{\hat{c}}{ \left( \chi_{ji}(\bm{m}, \bm{\theta}_m) \right)^2} \frac{\partial \chi_{ji}(\bm{m}, \bm{\theta}_m)}{\partial m_{j, i \neq j}} & \text{if } \lambda_{ij}^{\bm{\chi}} > 0 \\
0 & \text{otherwise}
\end{cases}
$$
From here, optimal military strategies can be characterized from \ref{eq:mStarHat}. Changes in $i$'s utility with respect to effort it expends on defense can be calculated as
$$
\frac{\partial \mathcal{L}_i^x(\bm{x}_i; \bm{m}, \bm{\lambda}_i^{\bm{\chi}}, \bm{\lambda}_i^{\bm{w}} \bm{\theta}_m)}{ \partial m_{ii}} = - \sum_j \lambda_{ij}^{\bm{\chi}} \frac{\hat{c}}{ \left( \chi_{ji}(\bm{m}, \bm{\theta}_m) \right)^2} \frac{\partial \chi_{ji}(\bm{m}, \bm{\theta}_m)}{\partial m_{ii}}
$$

Equilibrium in $\hat{\Gamma}^{\bm{m}}$ then requires the following
\begin{align}
\nabla_{\hat{\bm{x}}_i} \mathcal{L}_i^{\hat{\bm{x}}}(\hat{\bm{x}}_i; \bm{m}, \bm{\lambda}_i^{\bm{\chi}}, \bm{\lambda}_i^{\hat{\bm{w}}}, \bm{\theta}_m) &= \bm{0} \quad \text{for all } i \label{eq:g1} \\
\lambda_{ij}^{\bm{\chi}} \left( \hat{G}_j(\hat{\bm{w}}; \bm{\theta}_m) - \hat{G}_j \left( \hat{\bm{\tau}}_i^{j \star}(\bm{\theta}_m) \right) + \hat{c} \left( \chi_{ji}(\bm{m}, \bm{\theta}_m) \right)^{-1} \right) &= 0 \quad \text{for all } i, j \\
\hat{\bm{w}} - \hat{h}(\hat{\tilde{\bm{\tau}}}) &= \bm{0} \\
\nabla_{\bm{m}} \mathcal{L}_i^{\bm{m}}(\bm{m}; \bm{\lambda}^{\bm{m}}, \bm{\theta}_m) &= \bm{0} \quad \text{for all } i \\
\lambda_i^{\bm{m}} \left( M_i - \sum_k m_{jk} \right) &= 0 \quad \text{for all } i \label{eq:glast}
\end{align}
Let $g(\hat{\bm{x}}, \bm{\theta}_m, \bm{m}, \bm{\lambda}^{\bm{\chi}}, \bm{\lambda}^{\bm{w}}, \bm{\lambda}^{\bm{m}})$ be a function storing left hand side values of equations \ref{eq:g1}-\ref{eq:glast}.

For any guess of the parameter vector $\bm{\theta}_m$, a matrix of structural residuals can be calculated as
\begin{equation} \label{eq:moment}
\bm{\epsilon}_{N \times N}(\bm{\theta}_m) = \hat{\tilde{\bm{\tau}}}^\star(\bm{m}^\star; \bm{\theta}_m) - \bm{1}_{N \times N}
\end{equation}
In the data, $\hat{\tilde{\bm{\tau}}} = \bm{1}_{N \times N}$ which implies $\hat{\bm{w}} = \bm{1}_{1 \times N}$ and that $\hat{\bm{x}}_i$ is a vector of ones for all governments $i$. Then, $\bm{\theta}_m$ can be estimated by solving the following
\begin{equation} \label{eq:estimator}
\begin{split}
\min_{\bm{\theta}_m, \bm{m}, \bm{\lambda}^{\bm{\chi}}, \bm{\lambda}^{\bm{w}}, \bm{\lambda}^{\bm{m}}} & \quad \sum_i \sum_j \epsilon_{ij}(\bm{\theta}_m)^2 \\
\text{subject to} & \quad g(\bm{1}, \bm{\theta}_m, \bm{m}, \bm{\lambda}^{\bm{\chi}}, \bm{\lambda}^{\bm{w}}, \bm{\lambda}^{\bm{m}}) = \bm{0}
\end{split}
\end{equation}
Note that this also delivers estimates of equilibrium-consistent military strategies, $\bm{m}^\star$.







