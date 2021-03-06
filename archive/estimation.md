Solving the economy in changes for a set of $\hat{\bm{\tau}}$ requires values for a vector of economic parameters $\bm{\theta}_h$ and data on trade flows, policy barriers, and and national accounts. I discuss how I calibrate the economy in Appendix `r AeconomyC`. With $\hat{h}(\hat{\bm{\tau}}; \bm{\theta}_h)$ calibrated, $\hat{G}_i(\hat{\bm{\tau}})$ can be calculated for any set of trade policies and the optimal policy change problem (\ref{eq:optTaujHat}) can be solved, yielding $\hat{\bm{\tau}}_i^{j \star}$ for all $i, j$. I can then focus attention on $\hat{\Gamma}^{\bm{\tau}}$. The equilibrium of this game depends on a vector of parameters $\bm{\theta}_m = \left\{ \bm{b}, \bm{\alpha}, \gamma, \hat{c}, \sigma_{\epsilon}^2 \right\}$, as well as the values of the shocks $\left\{ \xi_{ij} \right\}_{i,j \in \left\{ 1, \dots, N \right\}}$ and $\left\{ \epsilon_{ij} \right\}_{i,j \in \left\{ 1, \dots, N \right\}}$. Because I work with an equilibrium in changes, a prediction $\hat{\tilde{\tau}}_{ij}^\star = 1$ is consistent with the data -- the model predicts that in equilibrium, government $i$ would make no changes to its factual trade policy toward $j$.

Estimation leverages Assumptions `r Axi` and `r Aepsilon`. First, note that these assumptions imply
\begin{equation*}
\begin{split}
\E_{\xi, \epsilon} \left[ \hat{\tilde{\bm{\tau}}}_i^\star(\bm{1}; \bm{\theta}_m) \right] = \argmax_{ \hat{\tilde{\bm{\tau}}}_i } & \quad \hat{G}_i(\hat{\tilde{\bm{\tau}}}_i; \bm{1}) \\
\text{subject to} & \quad \hat{G}_j(\hat{\tilde{\bm{\tau}}}) - \hat{G}_j(\hat{\bm{\tau}}_i^{j \star}) + \hat{c} \E_{\epsilon} \left[ \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)^{-1} \right] \geq 0 \quad \text{for all } j \neq i
\end{split}
\end{equation*}
where the affinity shocks drop out of the maximand. For any guess of the parameter vector $\bm{\theta}_m$, we can calculate $\E \left[ \hat{\tilde{\bm{\tau}}}_i^\star(\bm{1}; \bm{\theta}_m) \right]$ for each $i$. In practice, I compute this quantity by recasting \ref{eq:tauTildeStarHat} as a mathematical program with equilibrium constraints [@Su2012], which I discuss further in Appendix `r Ampec`. From here, we can construct the loss function
$$
\ell_{\bm{\tau}}(\bm{\theta}_m) = \sum_i \sum_j \left| \ln \left( \E \left[ \hat{\tilde{\tau}}_{ij}^\star(\bm{1}; \bm{\theta}_m) \right] \right) \right|
$$
where the theoretical prediction is implicitly divided by the empirical policy $\hat{\tilde{\tau}}_{ij}^\star = 1$. Absolute log loss is a natural loss function to use in this setting. If $\hat{\tilde{\tau}}_{ij}^\star = 1 / 2$ then $i$'s best response features a tariff on imports from $j$ that is half its facutal value. If $\hat{\tilde{\tau}}_{ij}^\star = 2$ then this tariff is twice it's factual value. The considered loss function penalizes each of these deviations equally.

The war constraints are informative about the value of the parameters $\bm{\alpha}$ and $\gamma$. Consider the case in which the $i$'s war constraint vis-à-vis $j$ holds with certainty. Then,
$$
\hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) - \left( \hat{G}_j(\hat{\bm{\tau}}_i^{j\star}; b_j) - \hat{c} \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)^{-1} \right) = 0
$$
which can be rearranged as
\begin{equation} \label{eq:constraintRegression}
\ln \left( \frac{ 1 }{ \hat{c}^{-1} \left( \hat{G}_j(\hat{\tau}_i^{j\star}) - 1 \right) - \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j) } \right) = \gamma \ln \left( \frac{ M_j }{ M_i } \right) - \bm{\alpha}^T W_{ji} + \epsilon_{ji}
\end{equation}
. The left side is a measure of the difference between $j$'s equilibrium utility difference and it's regime change value ($\hat{G}_j(\hat{\tau}_i^{j\star}) - \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}^\star; b_j)$). The extent to which this correlates with $j$'s military advantage over $i$ ($M_j / M_i$) is informative about the returns to military power, $\gamma$. The extent to which this correlates with dyadic geography is informative about the power projection parameters $\bm{\alpha}$.

In Appendix `r Aalphagamma` I show that if the regime change value is sufficiently high, there exists an $\epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m)$ such that for all $\epsilon_{ji} \geq \epsilon_{ji}^\star(\bm{Z}; \bm{\theta}_m)$, $i$'s constraint vis-à-vis $j$ holds. I can then construct a stochastic variant of \ref{eq:constraintRegression} which can be used as a second moment condition. This condition generates a second loss function, $\ell_{\epsilon}(\bm{\theta}_m)$ (Equation \ref{eq:lossEpsilon}).

I estimate the model by minimizing the sum of these two loss functions. Formally, I seek estimates $\hat{\bm{\theta}}_m$ such that
\begin{equation} \label{eq:thetaLoss}
\tilde{\bm{\theta}}_m \in \argmin_{\bm{\theta}_m} \quad \ell_{\bm{\tau}}(\bm{\theta}_m) + \ell_{\epsilon}(\bm{\theta}_m)
\end{equation}

In practice, I minimize these loss functions using an iterative procedure. In the first stage, I hold $\gamma$ and $\bm{\alpha}$ fixed and miminize $\ell_{\bm{\tau}}(\bm{\theta}_m)$ via adaptive grid search over $\bm{b}$, $\hat{c}$, and $\sigma_{\epsilon}$. Then, I hold these values fixed and choose $\gamma$ and $\bm{\alpha}$ to minimize $\ell_{\epsilon}(\bm{\theta}_m)$ using an ordinary least squares variant. I then return to the first stage and iterate until the loss functions stabilize. I discuss this procedure in more detail in Appendix `r AestimationAlg`.

This algorithm codifies intuition about identifying variation. Holding the military coercive environment fixed, the openness of trade policy is informative about governments underlying preferences, $\bm{b}$. The overall responsiveness of trade policy to the military coercive environment is governed by $\hat{c}$. When $\hat{c}$ is low, coercive threats are more credible, and policy is more responsive to threats. Conversely, when, $\hat{c}$ is high, few governments are willing to risk war and policies are more reflective of underlying preferences. For fixed $\gamma$ and $\bm{\alpha}$, $\tilde{\bm{b}}$, $\tilde{\hat{c}}$ are chosen to rationalize observed trade policies, given these incentives. 

Fixing governments preferences and the relative costs of war, the probability that constraints are binding on governments' trade policy choices can be calculated. The responsiveness of the constraints that bind with high probability to the military capability ratio ($M_j / M_i$) and dyadic geography $\bm{W}_{ji}$ are informative about $\gamma$ and $\bm{\alpha}$ respectively. 

This procedure produces point estimates $\tilde{\bm{\theta}}_m$. I then construct uncertainty intervals through nonparametric bootstrap, sampling policies and constraints and re-estimating $\tilde{\bm{\theta}}_m$. 
