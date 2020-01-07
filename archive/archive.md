\begin{center}
\begin{table}[ht]
\caption{Data Requirements and Estimation Targets \label{tab:data}}
\centering
\begin{tabular}{ccccc}
\multicolumn{2}{c@{\quad}}{Observables} 
&& 
\multicolumn{2}{c}{Unobservables} \\\cline{1-2}\cline{4-5} 
$\bm{M}$ & Military Endowments & & $\bm{\theta}_m$ & Parameters \\
$\bm{W}$ & Dyadic Geography & & $\bm{m}$ & Military Strategies \\
$\bm{\tau}$ & Policy Barriers to Trade & & &  \\
$\hat{h}$ & Calibrated Economy & & & 
\end{tabular}
\end{table}
\end{center}

This algorithm codifies intuition about identifying variation. Holding the military coercive environment fixed, the openness of trade policy is informative about governments underlying preferences, $\bm{b}$. The overall responsiveness of trade policy to the military coercive environment is governed by $\hat{c}$. When $\hat{c}$ is low, coercive threats are more credible, and policy is more responsive to threats. Conversely, when, $\hat{c}$ is high, few governments are willing to risk war and policies are more reflective of underlying preferences. For fixed $\gamma$ and $\bm{\alpha}$, $\tilde{\bm{b}}$, $\tilde{\hat{c}}$ are chosen to rationalize observed trade policies, given these incentives. 

This generates a sequence $\left\{ \bm{\theta}_m(\hat{c}_{\ell}) \right\}_{ \hat{c}_{\ell} \in \mathcal{C} }$ from which I choose estimates that satisfy
$$
\tilde{\bm{\theta}}_m \in \argmin_{ \hat{c}_{\ell} \in \mathcal{C} } \quad \ell_{\bm{\tau}}(\bm{\theta}_m(\hat{c}_{\ell}))
$$

First, suppose all governments are welfare-maximizing ($b_i=0$ for all $i$). It is straightforward to calculate their gains from moving to a world of free trade ($\tau_{ij} = 1$ for all $i, j$).^[In "hats", we have $\hat{\bm{\tau}}^{\text{ft}} = \bm{\tau}^{-1}$.] These come from two sources. First, prices fall and consumers benefit when the levels of factual barriers are significantly greater than welfare-optimal levels. Second, increasing market access through reducing trade barriers abroad increases real wages at home. Figure \ref{fig:Vhat} displays the empirical magnitudes of these gains. 

```{r Vhat, echo=FALSE, warning=FALSE, message=FALSE, fig.cap = paste0("Consumer welfare gains from moving to global free trade, $\\hat{V}_i(h(\\hat{\\bm{\\tau}}^{\\text{ft}}))$, for each country. Recall that changes are expressed in multiples from utility in the factual equilibrium, where this value is normalized to one $\\hat{V}_i(h(\\bm{1})) = 1$. \\label{fig:Vhat}"), fig.height=5, dpi=300, fig.pos="t"}

source("../02_figs/Ghatft.R")
GhatftFig + theme(aspect.ratio=1)

```

All governments gain substantially from moving to free trade. Those that face the largest barriers to foreign market access, such as Turkey and Russia, gain the most. This provides suggestive evidence that governments are not welfare-maximizers ($b_i > 0$). 

I can also solve the optimal policy change problem (\ref{eq:optTaujHat}) and calculate the welfare changes associated with successful wars. . I calculate counterfactual government utilities for each possible war under two assumptions on governments' motivations. First, I consider the case in which governments maximize consumer welfare ($b_i = 0$), followed by the case in which governments maximize rents ($b_i = 1$). The results are shown in Figure \ref{fig:rcv}.