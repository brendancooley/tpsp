Solving the economy in changes for a set of $\hat{\bm{\tau}}$ requires values for a vector of economic parameters $\bm{\theta}_h$ and data on trade flows, policy barriers, and and national accounts. I discuss how I calibrate the economy in Appendix `r AeconomyC`. With $\hat{h}(\hat{\bm{\tau}}; \bm{\theta}_h)$ calibrated, $\hat{G}_i(\hat{\bm{\tau}})$ can be calculated for any set of trade policies and the optimal policy change problem (\ref{eq:optTaujHat}) can be solved, yielding $\hat{\bm{\tau}}_i^{j \star}$ for all $i, j$. I can then focus attention on $\hat{\Gamma}^{\bm{m}}$. The equilibrium of this game depends on a vector of parameters $\bm{\theta}_m = \left\{ \bm{b}, \bm{\alpha}, \hat{c} \right\}$. While military allocations $\bm{m}^\star$ are unobserved, total military capacity ($M_i$) for each government is observable. Because I work with an equilibrium in changes, a prediction $\hat{\tilde{\tau}}_{ij} = 1$ is consistent with the data -- the model predicts that in equilibrium, government $i$ would make no changes to its factual trade policy toward $j$.

The ability of military allocations to distort choices depends on the power projection function $\rho_{ij}$. I adopt a simple logistic functional form for this function where
$$
\rho_{ij}(\bm{W}; \bm{\alpha}) = \frac{ 1 }{ 1 + e^{- \alpha_0 - \sum_k \alpha_k W_{ij, k} } }
$$
Here, $W_{ij, k}$ stores the $k$th dyadic geographic feature of the $ij$ dyad, such as minimum distance, and $\alpha_k$ is the effect of this feature on power projection capacity. If military power degrades with distance, the associated $\alpha_k$ would take a negative sign. 

Variation in policies unexplained by preference parameters $b_i$ is informative about the vector of power projection parameters $\bm{\alpha}$ and $\gamma$. Note that the Legrangian corresponding to the governments' constrained policy problem (\ref{eq:tauTildeStarHat}) function is a weighted average of the governments own utility and others' utility, where the weights are given by the Legrange multipliers. Constraints on policy choice are more likely to bind when a threatening government $j$ a) has a larger military allocation ($m_{ji}$ high) and b) when power projection costs are lower ($\rho_{ij}$ high). Therefore, the extent to which $i$'s policy choices favor government $j$ helps pin down power projection parameters. 

Solving each stage of the game is computationally expensive, however. For a given parameter vector, computing the equilibrium of $\hat{\Gamma}^{\bm{m}}$ requires computing the Nash Equilibrium of the military allocation game $\bm{m}^\star(\bm{\theta}_m)$ and computing the Nash Equilibrium of the constrained policy announcement game $\hat{\tilde{\bm{\tau}}}^\star(\bm{m}^\star; \bm{\theta}_m)$. In turn, computing these requires solving the equilibrium of the economy $\hat{h}(\hat{\bm{\tau}}; \bm{\theta}_h)$ for many trial $\bm{\tau}$. To ease computation, I recast governments' constrained policy problem (\ref{eq:tauTildeStarHat}) and the parameter estimation problem as mathematical programs with equilibrium constraints (MPECs) [@Su2012].

Consider first each government's policy problem. Converting \ref{eq:tauTildeStarHat} to an MPEC requires allowing the government to choose policies $\hat{\tilde{\bm{\tau}}}$ and wages $\hat{\bm{w}}$ while satisfying other governments' war constraints and enforcing that wages are consistent with international economic equilibrium $\hat{h}$.^[@Ossa2014 and @Ossa2016 also study optimal trade policy using this methodology.] Let $\hat{\bm{x}}_i = \left\{ \hat{\tilde{\bm{\tau}}}_i, \hat{\bm{w}} \right\}$ store $i$'s choice variables in this problem. Then, noting explicitly dependencies on $\bm{\theta}_m$, \ref{eq:tauTildeStarHat} can be rewritten^[I supress the $\bm{a}$ arguments in $\chi_{ji}$, where it is implied that no wars occur in equilibrium. $\chi_{ji}(\bm{m}, \bm{\theta}_m)$ is then the probability $j$ is successful in a war against $i$ when no other wars occur.] 
\begin{equation} \label{eq:tauTildeHatMPEC}
\begin{split}
\max_{\hat{\bm{x}}_i} & \quad \hat{G}_i(\hat{\bm{w}}; \bm{\theta}_m) \\
\text{subject to} & \quad \hat{G}_j(\hat{\bm{w}}; \bm{\theta}_m) - \hat{G}_j \left( \hat{\bm{\tau}}_i^{j \star}(\bm{\theta}_m) \right) + \hat{c} \left( \chi_{ji}(\bm{m}, \bm{\theta}_m) \right)^{-1} \geq 0 \quad \text{for all } j \neq i \\
& \quad \hat{\bm{w}} = \hat{h}(\hat{\tilde{\bm{\tau}}})
\end{split}
\end{equation}
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







