There are $N$ governments, indexed $i \in \left\{ 1, ..., N \right\}$. Governments choose trade policies $\bm{\tau}_i = \left( \tau_{i1}, ..., \tau_{iN} \right) \in [1, \bar{\tau}]^N$ which affect their welfare indirectly through changes in the international economy.^[$\bar{\tau}$ is an arbitarily large but finite value sufficient to shut down trade between any pair of countries.] An entry of the trade policy vector, $\tau_{ij}$ is the cost country $i$ imposes on imports from $j$.^[Costs enter in an "iceberg" fashion, and I normalize $\tau_{ii} = 1$. Then, if the price of a good in country $j$ is $p_{jj}$, its cost (less freight) in country $i$ is $\tau_{ij} p_{jj}$. The ad valorem tariff equivalent of the trade policy is $t_{ij} = \tau_{ij} - 1$. I employ structural estimates of these costs from @Cooley2019b to estimate the model, which are described in more detail in Appendix `r Aeconomy`.] The economy, detailed in Appendix `r Aeconomy`, can be succinctly characterized by a function $h: \bm{\tau} \rightarrow \mathbb{R}_{++}^N$ mapping trade policies to wages in each country, denoted $\bm{w} = \left( w_1, ..., w_N \right)$. These in turn determine trade flows between pairs of countries and price levels around the world.^[The economy is a variant of the workhorse model of @Eaton2002.]

Throughout, I will use $\bm{\theta}_m$ to denote the vector of all parameters to be estimated and $\bm{Z}$ to denote the vector of all data observed by the researcher. $\bm{\theta}_h$ denotes parameters associated with the economy, $h$, which will be calibrated. I will explicate the elements of these vectors in the proceeding sections and the Appendix.

Government welfare depends on these general equilibrium responses to trade policy choices. Governments value the welfare of a representative consumer that resides within each country. The consumer's welfare in turn depends on net revenues accrued through the government's trade policy distortions, which are redistributed to the consumer. Revenues and induced can be computed given knowledge of the general equilibrium function $h(\bm{\tau})$. Each government's welfare, is equivalent to the consumer's indirect utility, $V_i \left( h(\bm{\tau}); v_i \right)$ where $v_i$ is the revenue threshold parameter. This value of this function depends on the consumer's net income and is characterized fully in the Appendix. The consumer's net income can be written as a function of the governments' policy choices
$$
\tilde{Y}_i(h_i(\bm{\tau}))  = h_i(\bm{\tau}) * L_i + r_i(h(\bm{\tau}); v_i) . 
$$
$L_i$ is the country's labor endowment, $r_i(h(\bm{\tau}); v_i)$ is trade policy revenues, and $h_i(\bm{\tau})$ are equilibrium wages in $i$. $v_i \in [1, \infty)$ is a structural parameter that modulates the government's ability to extract trade policy rents from society. 

Adjusted revenues are given by
\begin{equation} \label{eq:r}
r_i(h(\bm{\tau}), v_i) = \sum_j (\tau_{ij} - v_i) X_{ij}(h(\bm{\tau}))
\end{equation}
and $X_{ij}(h(\bm{\tau}))$ are country $i$'s imports from country $j$.^[This object does not correspond empirically to governments' factual tariff revenues, as $\tau_{ij}$ incorporates a larger set of trade policy distortions than tariffs alone. Yet, non-tariff barriers to trade also generate rents that do not accrue directly to the government's accounts (see, for example, @Anderson1992 for the case of quotas). This revenue function is designed to capture this broader set of rents.] When $v_i$ is close to one, small policy distortions are sufficient to generate revenue for the government. Conversely when $v_i$ is high, the government must erect large barriers to trade before revenues begin entering government coffers and returning to the pockets of the consumer. Because consumers' consumption possibilities depend on revenue generation, increasing $v_i$ induces governments' to become more protectionist. This formulation provides substantial flexibility in rationalizing various levels of protectionism, while avoiding assuming specific political economic motivations for its genesis. From the perspective of the consumers, rents extracted from imports are valued equally, regardless of their source. Ex ante, governments are not discriminatory in their trade policy preferences preferences. Optimal policies for government $i$ maximize $V_i \left( h(\bm{\tau}); v_i \right)$.

These optimal policies impose externalities on other governments. By controlling the degree of market access afforded to foreign producers, trade policies affect the wages of foreign workers and the welfare of the governments that represent them. They also partially determine trade flows, which affect other governments' ability to collect rents. In this sense, protectionism is "beggar they neighbor." Governments' joint policy proposals are denoted $\tilde{\bm{\tau}}$.

Wars are fought in order to impose more favorable trade policies abroad. After observing policy proposals, governments decide whether or not to launch wars against one another. Wars are offensive and *directed*. If a government decides to launch a war it pays a dyad-specific cost and imposes more favorable trade policies on the target. These war costs are modeled as realizations of a random variable from a known family of distributions and are held as private information to the prospective attacker. The shape of these distributions is affected by the governments' relative power resources, denoted $\frac{M_j}{M_i}$, as well as the geographic distance between them, $Z_{ij}$. These costs are distributed with c.d.f. $F_{ji}$ which is described in more detail below. I normalize the cost of defending against aggression to zero.

If $i$ is not attacked by any other government its announced policies are implemented. Otherwise, free trade is imposed, setting $\bm{\tau}_i = \left\{ 1, \dots, 1 \right\} = \bm{1}_i$. Substituting these policies into $j$'s utility function gives $G_j(\bm{1}_i; \tilde{\bm{\tau}}_{-i})$ as $j$'s *conquest value* vis-à-vis $i$. Note that I prohibit governments from imposing discriminatory policies on conquered states. Substantively, this assumption reflects the difficulty in enforcing sub-optimal policies on prospective client states, relative to reorienting their political institutions to favor free trade. This also ensures that the benefits of regime change wars are public. However, it does not guarantee non-discrimination in times of peace. Governments that pose most credible threat of conquest can extract larger policy concessions from their targets in the form of directed trade liberalization. 

Government $j$ therefore prefers not to attack $i$ so long as
\begin{align*}
V_j \left( \bm{1}_i; \tilde{\bm{\tau}}_{-i} \right) - c_{ji} &\leq V_j \left( \tilde{\bm{\tau}} \right) \\
c_{ji}^{-1} &\leq \left( V_j \left( \bm{1}_i; \tilde{\bm{\tau}}_{-i} \right) - V_j \left( \tilde{\bm{\tau}} \right) \right)^{-1}
\end{align*}
or if the benefits from imposing free trade on $i$ are outweighed by the costs, holding other governments policies fixed. The probability that no government finds it profitable to attack $i$ can then be calculated as
$$
H_i \left( \tilde{\bm{\tau}}; \bm{Z}, \bm{\theta}_m \right) = \prod_{j \neq i} F_{ji} \left( \left( V_j \left( \bm{1}_i; \tilde{\bm{\tau}}_{-i} \right) - V_j \left( \tilde{\bm{\tau}} \right) \right)^{-1} \right)
$$
I am agnostic as to the process by which the coordination problem is resolved in the case in which multiple prospective attackers find it profitable to attack $i$. I assume simply that $i$ is attacked with certainty when it is profitable for any government to do so. This event occurs with probability $H_i(\tilde{\bm{\tau}}; \bm{Z}, \bm{\theta}_m)$. 

Because of strategic interdependencies between trade policies, optimal policy proposals are difficult to formulate. Governments face a complex problem of forming beliefs over the probabilities that they and each of their counterparts will face attack and the joint policies that will result in each contingency. For simplicity, I assume governments solve the simpler problem of maximizing their own utility, assuming no other government faces attack. I denote this objective function with $G_i(\tilde{\bm{\tau}})$ which can be written
\begin{equation} \label{eq:G}
G_i(\tilde{\bm{\tau}}) = H_i(\tilde{\bm{\tau}}; \bm{Z}, \bm{\theta}_m) V_i(\tilde{\bm{\tau}}) + \left( 1 - H_i(\tilde{\bm{\tau}}; \bm{Z}, \bm{\theta}_m) \right) V_i(\bm{1}_i; \tilde{\bm{\tau}}_{-i})
\end{equation}
where $V_i(\bm{1}_i; \tilde{\bm{\tau}}_{-i})$ denotes $i$'s utility when free trade is imposed upon it. This objective function makes clear the tradeoff $i$ faces when making policy proposals. Policies closer to $i$'s ideal point deliver higher utility conditional on peace, but raise the risk of war. Lowering barriers to trade on threatening countries increases $H_i(\tilde{\bm{\tau}}; \bm{Z}, \bm{\theta}_m)$, the probability $i$ avoids war, at the cost of larger deviations from policy optimality. 

Policy proposals are made simultaneously. Let $\tilde{\bm{\tau}}_i^\star(\tilde{\bm{\tau}}_{-i})$ denote a solution to this problem and $\tilde{\bm{\tau}}^\star$ a Nash equilibrium of this policy announcement game. 

## Policy Equilibrium in Changes

The equilibrium of the international economy depends on a vector of structural parameters and constants $\bm{\theta}_h$ defined in Appendix `r Aeconomy`. Computing the economic equilibrium $h(\bm{\tau}; \bm{\theta}_h)$ requires knowing these values. Researchers have the advantage of observing data related to the equilibrium mapping for one particular $\bm{\tau}$, the factual trade policies. 

The estimation problem can be therefore partially ameliorated by computing the equilibrium in *changes*, relative to a factual baseline. Consider a counterfactual trade policy $\tau_{ij}^\prime$ and its factual analogue $\tau_{ij}$. The counterfactual policy can be written in terms of a proportionate change from the factual policy with $\tau_{ij}^\prime = \hat{\tau}_{ij} \tau_{ij}$ where $\hat{\tau}_{ij} = 1$ when $\tau_{ij}^\prime = \tau_{ij}$. By rearranging the equilibrium conditions, I can solve the economy in changes, replacing $h(\bm{\tau}; \bm{\theta}_h) = \bm{w}$ with $\hat{h}(\hat{\bm{\tau}}; \bm{\theta}_h) = \hat{\bm{w}}$. Counterfactual wages can the be computed as $\bm{w}^\prime = \bm{w} \odot \hat{\bm{w}}$.

This method is detailed in Appendix `r Aeconomy`. Because structural parameters and unobserved constants do not change across equilibria, parameters that enter multiplicatively drop out of the equations that define this "hat" equilibrium. This allows me to avoid estimating these parameters, while enforcing that the estimated equilibrium is consistent with their values. The methodology, introduced by @Dekle2007, is explicated further in @Costinot2015 and used to study trade policy changes in @Ossa2014 and @Ossa2016.

It is straightforward to extend this methodology to the game studied here. Consider a modification to the policy-setting game described above in which governments propose changes to factual trade policies, denoted $\hat{\tilde{\bm{\tau}}}$. Note that this modification is entirely cosmetic -- the corresponding equilibrium in levels can be computed by multiplying factual policies by the "hat" equilibrium values ($\tau_{ij}^\prime = \hat{\tau}_{ij} \tau_{ij}$). I can then replace the equilibrium conditions above with their analogues in changes. 

Let $\hat{V}_j(\hat{\tilde{\bm{\tau}}})$ denote changes in $j$'s consumer welfare under proposed policy changes. Prospective attackers' peace conditions can be written in changes as
$$
\hat{c}_{ji}^{-1} \leq \left( \hat{V}_j \left( \bm{1}_i; \hat{\tilde{\bm{\tau}}}_{-i} \right) - \hat{V}_j \left( \hat{\tilde{\bm{\tau}}} \right) \right)^{-1}
$$
where
$$
\hat{c}_{ji} = \frac{c_{ji}}{V_j \left( \bm{\tau} \right)}
$$
measures the share of $j$'s utility lost to wage a war with $i$. I assume the inverse relative cost of war $j$ incurs when attacking $i$ is distributed Frechét with 
$$
\text{Pr}\left( \frac{1}{\hat{c}_{ji}} \leq \frac{1}{\hat{c}} \right) = \hat{F}_{ji} \left( \frac{1}{\hat{c}} \right) = \exp \left( -\frac{1}{\hat{C}} \left( \frac{M_j}{M_i} \right)^{\gamma} Z_{ji}^{-\alpha} \hat{c}^{\eta} \right) .
$$
The parameters $\alpha$ and $\gamma$ govern the extent to which military advantage and geographic proximity are converted into cost advantages. If $\gamma$ is greater than zero, then military advantage reduces the costs of war. Similarly, if $\alpha$ is greater than zero, then war costs increase with geographic distance, consistent with a loss of strength gradient. $\hat{C}$ and $\eta$ are global shape parameters that shift the cost distribution for all potential attackers.

Each government's objective function (\ref{eq:G}) in changes is
\begin{equation} \label{eq:Ghat}
\hat{G}_i(\hat{\tilde{\bm{\tau}}}) = \hat{H}_i(\hat{\tilde{\bm{\tau}}}; \bm{Z}, \bm{\theta}_m) \hat{V}_i(\hat{\tilde{\bm{\tau}}}) + \left( 1 - \hat{H}_i(\hat{\tilde{\bm{\tau}}}; \bm{Z}, \bm{\theta}_m) \right) \hat{V}_i(\bm{1}_i; \hat{\tilde{\bm{\tau}}}_{-i})
\end{equation}
where
$$
\hat{H}_i \left( \hat{\tilde{\bm{\tau}}}; \bm{Z}, \bm{\theta}_m \right) = \prod_{j \neq i} \hat{F}_{ji} \left( \left( \hat{V}_j \left( \bm{1}_i; \tilde{\bm{\tau}}_{-i} \right) - \hat{V}_j \left( \hat{\tilde{\bm{\tau}}} \right) \right)^{-1} \right) .
$$
With Frechét-distributed relative costs this equation has a closed functional form, with 
$$
\hat{H}_i \left( \hat{\tilde{\bm{\tau}}}; \bm{Z}, \bm{\theta}_m \right) = \exp \left( - \sum_{j \neq i} - \frac{1}{\hat{C}} \left( \frac{M_j}{M_i} \right)^{\gamma} Z_{ji}^{-\alpha} \left( \hat{V}_j \left( \bm{1}_i; \tilde{\bm{\tau}}_{-i} \right) - \hat{V}_j \left( \hat{\tilde{\bm{\tau}}} \right) \right)^{\eta} \right) .
$$

Let $\hat{\tilde{\bm{\tau}}}_i^\star(\hat{\tilde{\bm{\tau}}}_{-i})$ denote a solution to policy change proposal problem and $\hat{\tilde{\bm{\tau}}}^\star$ a Nash equilibrium of this policy change announcement game. 