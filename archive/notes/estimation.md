## Econometric Specification

Governments solve the following problem
\begin{equation} \label{eq:tauTildeStarHat}
\begin{split}
\max_{ \hat{\tilde{\bm{\tau}}}_i } & \quad \hat{G}_i(\hat{\tilde{\bm{\tau}}}_i; \hat{\tilde{\bm{\tau}}}_{-i}, b_i) + \sum_{j \neq i} \eta_{ij} \hat{G}_j(\hat{\tilde{\bm{\tau}}}; b_j) \\
\text{subject to} & \quad \hat{G}_j(\hat{\tilde{\bm{\tau}}}; b_j) - \hat{G}_j(\hat{\bm{\tau}}_i^{j \star}; b_j) + \hat{c} \left( \chi_{ji}(\bm{m}^\star, \bm{\alpha}) \right)^{-1} \geq 0 \quad \text{for all } j \neq i
\end{split}
\end{equation}
where $\eta_{ij}$ is a mean-zero random variable representing a stochastic affinity for government $j$ by government $i$. Let 
$$
\mathcal{L}_i^{\hat{\bm{\tau}}}(\hat{\tilde{\bm{\tau}}}_i, \bm{m}; \bm{\lambda}^{\chi})
$$
denote the Lagrangian associated with this problem and $\lambda_{ij}^\chi$ denote the multiplier associated with the $i-j$th war constraint. 

The contest function is
$$
\chi_{ji}(\bm{m}^\star, \bm{\alpha}) = \frac{\rho(\bm{X}_{ji}; \bm{\alpha}) m_{ji}^\star}{\rho(\bm{X}_{ji}; \bm{\alpha}) m_{ji}^\star + m_{ii}^\star}
$$
where
$$
\rho(\bm{X}_{ji}; \bm{\alpha})) = e^{-\bm{X}_{ji}^T \bm{\alpha} + \epsilon_{ij}}
$$
and
$$
\epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)
$$
is an unobserved but public shock to power projection capacity.

## Estimation Algorithm

Now recall our parameter vector of interest $\bm{\theta}_m = \left\{ \bm{b}, \bm{\alpha}, \hat{c} \right\}$ and fix starting values of $\bm{\alpha}$ and $\hat{c}$ at $\bm{\alpha}_0$ and $\hat{c}_0$. Additionally, set starting values for $\bm{m}_0^\star$ at $\bm{m}_i / M_i$ for all $i$. Let 
$$
\hat{\tilde{\bm{\tau}}}_i(b_i; \bm{m}^\star, \bm{\alpha}, \hat{c}) = \argmax_{\bm{\tau}_i} \mathcal{L}_i^{\hat{\bm{\tau}}}(\hat{\tilde{\bm{\tau}}}_i, \bm{m}; \bm{\lambda}^{\chi}) - \sum_{j \neq i} \eta_{ij} \hat{G}_j(\hat{\tilde{\bm{\tau}}}; b_j)
$$

We can get an initial estimate of the preference parameters $\bm{b}_1$ by finding a vector of $b_{i}$ such that
$$
b_{1i} = \argmin_{b_i} \ln \left[ \hat{\tilde{\bm{\tau}}}_i(b_i; \bm{m}^\star, \bm{\alpha}, \hat{c}) - 1 \right]
$$
Here, we minimize the log distance between the data moment and the model-implied expected moment.^[Is this equivalent to minimizing errors $\bm{\eta}$?] Since $\hat{\tilde{\bm{\tau}}}^\star = \bm{1}$ in the data and constitutes a Nash equilibrium, we do not need to iteratively compute best responses to find equilibria, we only want to find preference parameters that make that data as close as possible to a best response in the deterministic model (with no $\bm{\eta}$). I can find these $\bm{b}$ reasonably quickly through grid search. 

**Problem here:** Lagrange multipliers on constraint at $\bm{b}_1$ are not the actual multipliers. Is this also a problem for measurement error in power projection? Also tricky interdependencies between multipliers and power projection parameters...if we're too restrictive on these none of the multipliers will turn on and estimates will be affected by this. Which constraints do we want to use?]

**Solution:** We can draw on Proposition 1 here...we don't need lambdas, only need $\bm{m}^\star$. Only use moments for which $m_{ji} > 0$. Followup: what happens when lambdas and $m$s disagree? Think we have to use $m$ because constraints also bind probabilistically. On second round of updates this problem goes away.^[This is potentially an issue...Incorrect distribution of military force leads us to infer it is not effective which leads equilibrium levels to adjust downward (maybe) and feed cycle inferring force isn't effective. Maybe because making force less effective also means I can put less on defense...these effects might cancel out.]

**Next Problem:** We *will* need the lambdas for calculating $m_{ii}^\star$.

**Solution (Maybe):** we can work with unbiased estimate from solving step one without errors.

*Assuming affinity shocks and power projection shocks are realized after military strategies but before policy announcements helps a lot. Then we can use the expectation of the multiplier to inform about allocations in the first stage. We do need an analytical expression for the probability the constraint binds as a function of effort, however.*

Now, with an initial estimate of $\bm{b}$ we can proceed to estimate the remainder of the parameter vector $\bm{\theta}_m$. 