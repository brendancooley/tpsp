There are $N$ governments, indexed $i \in \left\{ 1, ..., N \right\}$. Governments choose trade policies $\bm{\tau}_i = \left( \tau_{i1}, ..., \tau_{iN} \right) \in [1, \bar{\tau}]^N$ which affect their welfare indirectly through changes in the international economy.^[$\bar{\tau}$ is an arbitarily large but finite value sufficient to shut down trade between any pair of countries.] An entry of the trade policy vector, $\tau_{ij}$ is the cost country $i$ imposes on imports from $j$.^[Costs enter in an "iceberg" fashion, and I normalize $\tau_{ii} = 1$. Then, if the price of a good in country $j$ is $p_{jj}$, its cost (less freight) in country $i$ is $\tau_{ij} p_{jj}$. The ad valorem tariff equivalent of the trade policy is $t_{ij} = \tau_{ij} - 1$. I employ structural estimates of these costs from @Cooley2019b to estimate the model, which are described in more detail in Appendix `r Aeconomy`.] The economy, detailed in Appendix `r Aeconomy`, can be succinctly characterized by a function $h: \bm{\tau} \rightarrow \mathbb{R}_{++}^N$ mapping trade policies to wages in each country, denoted $\bm{w} = \left( w_1, ..., w_N \right)$. These in turn determine trade flows between pairs of countries and price levels around the world.^[The economy is a variant of the workhorse model of @Eaton2002.]

Throughout, I will use $\bm{\theta}_m$ to denote the vector of all parameters to be estimated and $\bm{Z}$ to denote the vector of all data observed by the researcher. $\bm{\theta}_h$ denotes parameters associated with the economy, $h$, which will be calibrated. I will explicate the elements of these vectors in the proceeding sections and the Appendix.

Government welfare depends on these general equilibrium responses to trade policy choices. Governments value the welfare of a representative consumer that resides within each country and rents accrued through trade policy distortions (tariff revenues). These can be computed given knowledge of the general equilibrium function $h(\bm{\tau})$. I assume these elements are complements in the production of government utility, formally,
\begin{equation} \label{eq:G}
G_i(\bm{\tau}; b_i) = V_i \left( h(\bm{\tau}) \right) r_i \left( h(\bm{\tau}; v_i) \right)
\end{equation}
where $V_i(h(\bm{\tau}))$ is consumer welfare, $r_i(h(\bm{\tau}), v_i)$ are tariff revenues, and $v_i \in [1, \infty)$ is a structural parameter that modulates the government's ability to extract trade policy rents from society. Revenues are given by
\begin{equation} \label{eq:r}
r_i(h(\bm{\tau}), v_i) = \sum_j (\tau_{ij} - v_i) X_{ij}(h(\bm{\tau}))
\end{equation}
and $X_{ij}(h(\bm{\tau}))$ are country $i$'s imports from country $j$.^[This object does not correspond empirically to governments' factual tariff revenues, as $\tau_{ij}$ incorporates a larger set of trade policy distortions than tariffs alone. Yet, non-tariff barriers to trade also generate rents that do not accrue directly to the government's accounts (see, for example, @Anderson1992 for the case of quotas). This revenue function is designed to capture this broader set of rents.] When $v_i$ is close to one, small policy distortions are sufficient to generate revenue for the government. Conversely when $v_i$ is high, the government must erect large barriers to trade before revenues begin entering government coffers. Because some revenue is necessary in order to produce government welfare, increasing $v_i$ induces governments' to become more protectionist. This formulation provides substantial flexibility in rationalizing various levels of protectionism, while avoiding assuming specific political economic motivations for its genesis. From the perspective of the governments, rents extracted imports are valued equally, regardless of their source. Ex ante, governments are not discriminatory in their preferences.

Characterizing the theoretical properties of this objective function is challenging due to the network effects inherent in general equilibrium trade models. Suppose $i$ increases barriers to trade with $j$. This affects wages and prices in $j$ and its competitiveness in the world economy. These changes affect its trade with all other countries $k$, which affects their welfare, and so on [@Allen2019]. In order to make progress, I assume that $G_i$ is quasiconcave in $i$'s own policies, which ensures the existence of an unconstrained optimal policy vector and a pure-strategy Nash equilibrium to the simultaneous policy-setting game.^[This property holds in my numerical simulations, but may not hold more generally. The existence of a Nash equilibrium follows from the fixed point theorem of Debreu, Glicksberg, and Fan [@Fudenberg1992 p. 34].]

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
AG <- Atick
Atick <- Atick + 1

AGText <- knit_child("../results/AG.md")
```

**Assumption `r AG`**: `r AGText`

These optimal policies impose externalities on other governments. By controlling the degree of market access afforded to foreign producers, trade policies affect the wages of foreign workers and the welfare of the governments that represent them. They also partially determine trade flows, which affect other governments' ability to collect rents. In this sense, protectionism is "beggar they neighbor." Governments' joint policy proposals are denoted $\tilde{\bm{\tau}}$.

After trade policies announcements are made, governments decide whether or not they would like to wage war against other governments. Wars are fought in order to impose more favorable trade policies abroad. Each government is endowed with military capacity $M_i$ which can be used in wars with other governments. Wars are offensive and *directed*. Formally, let $\bm{a}_i = \left( a_{i1}, ..., a_{iN} \right)$ denote $i$'s war entry choices, where $a_{ij} \in \left\{ 0 , 1 \right\}$ denotes whether or not $i$ choose to attack $j$.^[Note that this formulation leaves open the possibility that two governments launch directed wars against one another, $a_{ij} = a_{ji} = 1$.] $a_{ii} = 1$ for all $i$ by assumption -- governments always choose to defend themselves.

If $i$ is successful in defending itself against all attackers, its announced policies are implemented. Government $i$ suffers a cost $c_i$ for each war it must fight, accruing total war costs $\sum_{j \neq i} a_{ij} c_i$. Each attacker $j$ also pays $c_j$. When a government $i$ wins a war against $j$, it earns the right to impose free trade on the losing government, setting $\bm{\tau}_j = \left\{ 1, \dots, 1 \right\} = \bm{1}_j$. Substituting these policies into $i$'s utility function gives $G_i(\bm{1}_j; \tilde{\bm{\tau}}_{-j})$ as $i$'s *conquest value* vis-à-vis $j$. Note that I prohibit governments from imposing discriminatory policies on conquered states. Substantively, this assumption reflects the difficulty in enforcing sub-optimal policies on prospective client states, relative to reorienting their political institutions to favor free trade.^[It also substantially eases computational burdens by avoiding re-calculating optimal imposed policies for different values of preference parameters.] This also ensures that the benefits of regime change wars are public. However, it does not guarantee non-discrimination in times of peace. Governments that pose most credible threat of conquest can extract larger policy concessions from their targets in the form of directed trade liberalization. 

Governments' ability to prosecute wars against one another depend on dyadic geographic factors $\bm{W}$, such as geographic distance. For every unit of force $i$ dedicates toward attacking $j$, $\rho_{ji}(\bm{W}_{ji}; \bm{\alpha}) > 0$ units arrive. I normalize $\rho_{jj} = 1$ -- defensive operations do not result in any loss of strength. $\bm{\alpha}$ is a vector of structural parameters governing this relationship to be estimated. I adopt a simple exponential functional form for this function where
\begin{equation} \label{eq:rho}
\rho_{ji}(\bm{W}_{ji}; \bm{\alpha}) = e^{ -\bm{\alpha}^T \bm{W}_{ji} } .
\end{equation}
$\bm{W}_{ji}$ is a vector of dyadic geographic features such as centroid-centroid distance, and $\bm{\alpha}$ parameterizes the effect of these features on power projection capacity.

War outcomes are determined by a contest function
\begin{equation} \label{eq:chi}
\chi_{ij}(\bm{a}) = \frac{ a_{ij} \rho_{ij}(\bm{W}; \bm{\alpha}) M_i }{ \sum_k a_{kj} \rho_{kj}(\bm{W}; \bm{\alpha}) M_k } .
\end{equation}
$\chi_{ij} \in [0, 1]$ is the probability that $i$ is successful in an offensive war against $j$. Note that wars are multilateral and winner-take-all. Many countries can choose to attack $j$, but only one government can win. 

For the moment, fix $a_{jk} = 0$ for all $k \neq i$, $j \neq k$. $i$ is the only government that faces the possibility of attack. Then, all other policy proposal vectors $\tilde{\bm{\tau}}_{-i}$ are implemented with certainty and $i$'s utility as a function of war entry decisions is 
$$
\tilde{G}_i^{\bm{a}}(\bm{a}) = \chi_{ii}(\bm{a}) G_i(\tilde{\bm{\tau}}) + \sum_{j \neq i} \left( \chi_{ji}(\bm{a}) G_i(\bm{1}_i; \tilde{\bm{\tau}}_{-i}) - a_{ji} c_i \right)
$$

Attackers consider the effect of their war entry on the anticipated policy outcome. Now consider an attacker $j$'s war entry decision vis-à-vis a defender $i$, assuming no other country launches a war. Let 
$$
\tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m) = \frac{ \rho_{ji}(\bm{W}; \bm{\alpha}) M_j }{ \rho_{ji}(\bm{W}; \bm{\alpha}) M_j + M_i }
$$
denote the probability that $j$ is successful in this contingency.

Government $j$ prefers not to attack $i$ so long as
\begin{equation} \label{eq:AwarConstraint}
G_j(\tilde{\bm{\tau}}) \geq \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m) G_j(\bm{1}_i; \tilde{\bm{\tau}}_{-j}) + \left( 1 - \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m) \right) G_j(\tilde{\bm{\tau}}) - c_j .
\end{equation}

Let $\bm{a}^\star : \tilde{\bm{\tau}} \rightarrow \left\{ 0, 1 \right\}_{N - 1 \times N - 1}$ denote equilibrium war entry decisions as a function of announced policies and $a_{ij}^\star(\tilde{\bm{\tau}})$ denote $i$'s equilibrium decision of whether or not to attack $j$. Governments choose whether or not to enter wars simultaneously. When peace prevails, $a_{ij}^\star(\tilde{\bm{\tau}}) = 0$ for all $i \neq j$. 

To recap, governments make policy announcements, $\tilde{\bm{\tau}}$, and then launch wars $\bm{a}$.  At each stage, actions are taken simultaneously. The solution concept is subgame perfect equilibrium. In Appendix `r AwarEntry`, I show that there exist peaceful equilibria to the game as long as war costs are large enough (Proposition `r Pc`).^[This result mirrors @Fearon1995's proof of the existence of a bargaining range in a unidimensional model. Here, because the governments' objective functions are not necessarily concave, war costs may need to be larger than zero in order to guarantee peace.] I assume the values of $\left\{ c_i \right\}_{ i \in \left\{ 1, \dots, N \right\} }$ are large enough to deliver peace and restrict attention to these equilibria. I can then analyze the subgame consisting policy announcements only, while ensuring that inequality \ref{eq:AwarConstraint} holds for every attacker $k$ and potential target $i$.

Optimal trade policies proposals for $i$ in this case solve
\begin{equation} \label{eq:tauTildeStar}
\begin{split}
\max_{ \tilde{\bm{\tau}}_i } & \quad G_i(\tilde{\bm{\tau}}_i; \tilde{\bm{\tau}}_{-i}) \\
\text{subject to} & \quad G_j(\tilde{\bm{\tau}}) - G_j(\bm{1}_i; \tilde{\bm{\tau}}_{-i}); \tilde{\bm{\tau}}_{-i}) + c \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)^{-1} \geq 0 \quad \text{for all } j \neq i
\end{split}
\end{equation}
where the constraints can be derived by rearranging \ref{eq:AwarConstraint}. Formulated in this manner, it becomes clear that military allocations affect trade policy through their effect on the $i$'s war constraints. As $M_j$ increases, $\tilde{\chi}_{ji}$ increases as well, tightening the constraint on $i$'s policy choice. Let $\tilde{\bm{\tau}}_i^\star(\tilde{\bm{\tau}}_{-i})$ denote a solution to this problem and $\tilde{\bm{\tau}}^\star$ a Nash equilibrium of the constrained policy announcement game. I refer to this game as $\Gamma^{\bm{\tau}}$.

## Policy Equilibrium in Changes

The equilibrium of the international economy depends on a vector of structural parameters and constants $\bm{\theta}_h$ defined in Appendix `r Aeconomy`. Computing the equilibrium $h(\bm{\tau}; \bm{\theta}_h)$ requires knowing these values. Researchers have the advantage of observing data related to the equilibrium mapping for one particular $\bm{\tau}$, the factual trade policies. 

The estimation problem can be therefore partially ameliorated by computing the equilibrium in *changes*, relative to a factual baseline. Consider a counterfactual trade policy $\tau_{ij}^\prime$ and its factual analogue $\tau_{ij}$. The counterfactual policy can be written in terms of a proportionate change from the factual policy with $\tau_{ij}^\prime = \hat{\tau}_{ij} \tau_{ij}$ where $\hat{\tau}_{ij} = 1$ when $\tau_{ij}^\prime = \tau_{ij}$. By rearranging the equilibrium conditions, I can solve the economy in changes, replacing $h(\bm{\tau}; \bm{\theta}_h) = \bm{w}$ with $\hat{h}(\hat{\bm{\tau}}; \bm{\theta}_h) = \hat{\bm{w}}$. Counterfactual wages can the be computed as $\bm{w}^\prime = \bm{w} \odot \hat{\bm{w}}$.

This method is detailed in Appendix `r Aeconomy`. Because structural parameters and unobserved constants do not change across equilibria, parameters that enter multiplicatively drop out of the equations that define this "hat" equilibrium. This allows me to avoid estimating these parameters, while enforcing that the estimated equilibrium is consistent with their values. The methodology, introduced by @Dekle2007, is explicated further in @Costinot2015 and used to study trade policy changes in @Ossa2014 and @Ossa2016.

It is straightforward to extend this methodology to the game studied here. Consider a modification to the policy-setting subgame in which governments propose changes to factual trade policies $\hat{\tilde{\bm{\tau}}}$ and call this game $\Gamma^{\hat{\bm{\tau}}}$. Note that this modification is entirely cosmetic -- the corresponding equilibrium of $\Gamma^{\hat{\bm{\tau}}}$ in levels can be computed by multiplying factual policies by the "hat" equilibrium values ($\tau_{ij}^\prime = \hat{\tau}_{ij} \tau_{ij}$). I can then replace the equilibrium conditions of $\Gamma^{\bm{\tau}}$ with their analogues in changes. Each government's welfare (\ref{eq:G}) in changes is
\begin{equation} \label{eq:Ghat}
\hat{G}_i(\hat{\bm{\tau}}) = \hat{V}_i \left( \hat{h}(\hat{\bm{\tau}}) \right) \hat{r}_i \left( \hat{h}(\hat{\bm{\tau}}), v_i \right) .
\end{equation}

If $i$ loses a war, free trade is imposed, equivalent to $\hat{\bm{\tau}} = bm{\tau}_i^{-1}$ where $\bm{\tau}_i$ are observed policies. By dividing the governments' war constraints (\ref{eq:AwarConstraint}) by their factual welfare $G_j(\bm{\tau})$, their constrained policy announcement problem can be rewritten as the solution to
\begin{equation} \label{eq:tauTildeStarHat}
\begin{split}
\max_{ \hat{\tilde{\bm{\tau}}}_i } & \quad \hat{G}_i(\hat{\tilde{\bm{\tau}}}_i; \hat{\tilde{\bm{\tau}}}_{-i}) \\
\text{subject to} & \quad \hat{G}_j(\hat{\tilde{\bm{\tau}}}) - \hat{G}_j(\bm{\tau}_i^{-1}) + \hat{c} \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)^{-1} \geq 0 \quad \text{for all } j \neq i
\end{split}
\end{equation}
where
$$
\hat{c} = \frac{c_i}{G_j(\bm{\tau}; b_j)}
$$
is the *share* of factual utility each government pays if a war occurs. Assumption `r Achat` requires that governments pay the same share of their factual utility in any war.^[While not innocuous, this assumption is more tenable than assumption constant absolute costs. It formalizes the idea that larger countries (that collect more rents and have high real incomes than their smaller counterparts) also pay more in military operations. It avoids the complications inherent in the more realistic but less tractable assumption that war costs depend on power ($\tilde{\chi}$).]

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
Achat <- Atick
Atick <- Atick + 1

AchatText <- knit_child("../results/Achat.md")
```

**Assumption `r Achat` (Constant Relative War Costs):** `r AchatText`

$\hat{\tilde{\bm{\tau}}}_i^\star(\hat{\tilde{\bm{\tau}}}_{-i})$ denotes $i$'s best response function and $\hat{\tilde{\bm{\tau}}}^\star$ denotes an equilibrium in changes. 