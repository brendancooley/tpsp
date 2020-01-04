There are $N$ governments, indexed $i \in \left\{ 1, ..., N \right\}$. Governments choose trade policies $\bm{\tau}_i = \left\{ \tau_{i1}, ..., \tau_{iN} \right\} \in [1, \bar{\tau}]^N$ which affect their welfare indirectly through changes in the international economy.^[$\bar{\tau}$ is an arbitarily large but finite value sufficient to shut down trade between any pair of countries.] An entry of the trade policy vector, $\tau_{ij}$ is the cost country $i$ imposes on imports from $j$.^[Costs enter in an "iceberg" fashion, and I normalize $\tau_{ii} = 1$. Then, if the price of a good in country $j$ is $p_{jj}$, its cost (less freight) in country $i$ is $\tau_{ij} p_{jj}$. The ad valorem tariff equivalent of the trade policy is $t_{ij} = \tau_{ij} - 1$. I employ structural estimates of these costs from @Cooley2019b to estimate the model, which are described in more detail in Appendix `r Aeconomy`.] The economy, detailed in Appendix `r Aeconomy`, can be succinctly characterized by a function $h: \bm{\tau} \rightarrow \mathbb{R}_{++}^N$ mapping trade policies to wages in each country, denoted $\bm{w} = \left\{ w_1, ..., w_N \right\}$. These in turn determine trade flows between pairs of countries and prices around the world.^[The economy is a variant of the workhorse model of @Eaton2002.]

Throughout, I will use $\bm{\theta}_m$ to denote the vector of all parameters to be estimated and $\bm{Z}$ to denote the vector of all data observed by the researcher. $\bm{\theta}_h$ denotes parameters associated with the economy, $h$, which will be calibrated. I will explicate the elements of these vectors in the proceeding sections and the Appendix.

Government welfare depends on these general equilibrium responses to trade policy choices. Governments value the welfare of a representative consumer that resides within each country and rents accrued through trade policy distortions (tariff revenues). These can be computed given knowledge of the general equilibrium function $h(\bm{\tau})$. Formally, governments' *direct* utility is 
\begin{equation} \label{eq:G}
G_i(\bm{\tau}; b_i) = V_i \left( h(\bm{\tau}) \right)^{1 - b_i} r_i \left(h(\bm{\tau}) \right)^{b_i}
\end{equation}
where $V_i(h(\bm{\tau}))$ is consumer welfare, $r_i(h(\bm{\tau}))$ are tariff revenues, and $b_i$ is a structural parameter that governs the government's relative preference for these. When the government values the welfare of the consumer ($b_i = 0$), it prefers to set "optimal tariffs" in the sense of @Johnson1953. Tariffs higher than these hurt consumers but raise more revenue for the government. When the government values rents ($b_i = 1$), it prefers trade policies that maximize revenue where
\begin{equation} \label{eq:r}
r_i(h(\bm{\tau})) = \sum_j (\tau_{ij} - 1) X_{ij}(h(\bm{\tau}))
\end{equation}
and $X_{ij}(h(\bm{\tau}))$ are country $i$'s imports from country $j$.^[This object does not correspond empirically to governments' factual tariff revenues, as $\tau_{ij}$ incorporates a larger set of trade policy distortions than tariffs alone. Yet, non-tariff barriers to trade also generate rents that do not accrue directly to the government's accounts (see, for example, @Anderson1992 for the case of quotas). This revenue function is designed to capture this broader set of rents.]

Characterizing the theoretical properties of this objective function is challenging due to the network effects inherent in general equilibrium trade models. Suppose $i$ increases barriers to trade with $j$. This affects wages and prices in $j$ and its competitiveness in the world economy. These changes affect its trade with all other countries $k$, which affects their welfare, and so on [@Allen2019]. In order to make progress, I assume that $G_i$ is quasiconcave in $i$'s own policies, which ensures the existence of an unconstrained optimal policy vector and a pure-strategy Nash equilibrium to the simultaneous policy-setting game.^[This property holds in my numerical applications, but may not hold more generally. The existence of a Nash equilibrium follows from the fixed point theorem of Debreu, Glicksberg, and Fan [@Fudenberg1992 p. 34].]

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
AG <- Atick
Atick <- Atick + 1

AGText <- knit_child("../results/AG.md")
```

**Assumption `r AG`**: `r AGText`

These optimal policies impose externalities on other governments. By controlling the degree of market access afforded to foreign producers, trade policies affect the wages of foreign workers and the welfare of the governments that represent them. They also partially determine trade flows, which affect other governments' ability to collect rents. In this sense, protectionism is "beggar they neighbor." Governments' joint policy proposals are denoted $\tilde{\bm{\tau}}$.

Governments may be induced to value the welfare of other governments for cultural, diplomatic, or other non-military reasons. These incentives are summarized in *affinity shocks*, $\xi_{ij}$, realized before policies are proposed. Governments' optimal policies maximize their utility net of these affinity shocks, denoted
$$
\tilde{G}_i(\bm{\tau}) = G_i(\bm{\tau}) + \sum_{j \neq i} \xi_{ij} G_j(\bm{\tau})
$$.

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
Axi <- Atick
Atick <- Atick + 1

AxiText <- knit_child("../results/Axi.md")
```

**Assumption `r Axi`**: `r AxiText`

After trade policies announcements are made, governments decide whether or not they would like to wage war against other governments. Wars are fought in order to impose more favorable trade policies abroad. Each government is endowed with military capacity $M_i$ which can be used in wars with other governments. Wars are offensive and *directed*. Formally, let $\bm{a}_i = \left\{ a_{i1}, ..., a_{iN} \right\}$ denote $i$'s war entry choices, where $a_{ij} \in \left\{ 0 , 1 \right\}$ denotes whether or not $i$ choose to attack $j$.^[Note that this formulation leaves open the possibility that two governments launch directed wars against one another, $a_{ij} = a_{ji} = 1$.] $a_{ii} = 1$ for all $i$ by assumption -- governments always choose to defend themselves.

If $i$ is successful in defending itself against all attackers, its announced policies are implemented. Government $i$ suffers a cost $c_i$ for each war it must fight, accruing total war costs $\sum_{j \neq i} a_{ij} c_i$. Each attacker $j$ also pays $c_j$. When a government $i$ wins a war against $j$, it earns the right to dictate $j$'s trade policy. Optimal policies for a victorious government $j$ are denoted $\bm{\tau}_j^{i \star}$ and solve^[Note that these maximize $G_i$, rather than $\tilde{G}_i$. I assume that in the event of a war, affinity shocks are re-drawn, so governments receive $\E[\tilde{G}_i(\bm{\tau})] = G_i(\bm{\tau})$.]
\begin{equation} \label{eq:optTauj}
\begin{split}
\max_{\bm{\tau}_j} & \quad G_i(\bm{\tau}_j; \tilde{\bm{\tau}}_{-j}) \\
\text{subject to} & \quad \tau_{jj} = 1
\end{split}
\end{equation}

Governments' ability to prosecute wars against one another depend on dyadic geographic factors $\bm{W}$, such as geographic distance. For every unit of force $i$ allocates toward attacking $j$, $\rho_{ji}(W_{ji}; \bm{\alpha}) > 0$ units arrive. I normalize $\rho_{jj} = 1$ -- defensive operations do not result in any loss of strength. $\bm{\alpha}$ is a vector of structural parameters governing this relationship to be estimated. I adopt a simple exponential functional form for this function where
\begin{equation} \label{eq:rho}
\rho_{ji}(W_{ji}; \bm{\alpha}) = e^{ -\bm{\alpha}^T \bm{W}_{ji} + \epsilon_{ji} }
\end{equation}
. $\bm{W}_{ij}$ is a vector of dyadic geographic features such as centroid-centroid distance, and $\bm{\alpha}$ parameterizes the effect of these features on power projection capacity. $\epsilon_{ij}$ is a *war shock* that captures determinants of power projection capacity not included in $\bm{W}_{ji}$.

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
Aepsilon <- Atick
Atick <- Atick + 1

AepsilonText <- knit_child("../results/Aepsilon.md")
```

**Assumption `r Aepsilon`**: `r AepsilonText`

War outcomes are determined by a contest function 
\begin{equation} \label{eq:chi}
\chi_{ij}(\bm{a}) = \frac{ a_{ij} \rho_{ij}(\bm{W}; \bm{\alpha}) M_i^{\gamma} }{ \sum_k a_{kj} \rho_{kj}(\bm{W}; \bm{\alpha}) M_k^{\gamma} }
\end{equation}.
$\chi_{ij} \in [0, 1]$ is the probability that $i$ is successful in an offensive war against $j$. $\gamma$ is a structural parameter that governs the returns to military strength. When $\gamma = 1$, $\chi_{ij}(\bm{a}, \bm{m})$ is a standard additive contest function, where strengths are discounted by $\rho_{ij}(\bm{W}; \bm{\alpha})$. When $\gamma = 0$, military strength does not influence contest outcomes [@Jia2013]. Note that wars are multilateral and winner-take-all. Many countries can choose to attack $j$, but only one government can win the right to dictate $j$'s policy. 

For the moment, fix $a_{jk} = 0$ for all $k \neq i$, $j \neq k$. $i$ is the only government that faces the possibility of attack. Then, all other policy proposal vectors $\tilde{\bm{\tau}}_{-i}$ are implemented with certainty and $i$'s utility as a function of war entry decisions is 
$$
\tilde{G}_i^{\bm{a}}(\bm{a}) = \chi_{ii}(\bm{a}) \tilde{G}_i(\tilde{\bm{\tau}}) + \sum_{j \neq i} \left( \chi_{ji}(\bm{a}) G_i(\bm{\tau}_i^{j \star}; \tilde{\bm{\tau}}_{-i}) - a_{ji} c_i \right)
$$

Attackers consider the effect of their war entry on the anticipated policy outcome. Now consider an attacker $j$'s war entry decision vis-à-vis a defender $i$, assuming no other country launches a war. Let 
$$
\tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m) = \frac{ \rho_{ji}(\bm{W}; \bm{\alpha}) M_j^{\gamma} }{ \rho_{ji}(\bm{W}; \bm{\alpha}) M_j^{\gamma} + M_i^{\gamma} }
$$
denote the probability that $j$ is successful in this contingency.

Government $j$ prefers not to attack $i$ so long as
\begin{equation} \label{eq:AwarConstraint}
\tilde{G}_j(\tilde{\bm{\tau}}) \geq \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m) G_j(\bm{\tau}_i^{j \star}; \tilde{\bm{\tau}}_{-j}) + \left( 1 - \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m) \right) \tilde{G}_j(\tilde{\bm{\tau}}) - c_j
\end{equation}. 

Let $\bm{a}^\star : \tilde{\bm{\tau}} \rightarrow \left\{ 0, 1 \right\}_{N - 1 \times N - 1}$ denote equilibrium war entry decisions as a function of announced policies and $a_{ij}^\star(\tilde{\bm{\tau}})$ denote $i$'s equilibrium decision of whether or not to attack $j$. Governments choose whether or not to enter wars simultaneously. When peace prevails, $a_{ij}^\star(\tilde{\bm{\tau}}) = 0$ for all $i \neq j$. 

To recap, governments make policy announcements, $\tilde{\bm{\tau}}$, and then launch wars $\bm{a}$.  At each stage, actions are taken simultaneously. The solution concept is subgame perfect equilibrium. In Appendix `r AwarEntry`, I show that there exist peaceful equilibria of the subgame consisting of policy announcements and war entry decisions as long as war costs are large enough (Proposition `r Pc`).^[This result mirrors @Fearon1995's proof of the existence of a bargaining range in a unidimensional model. Here, because the governments' objective functions are not necessarily concave, war costs may need to be larger than zero in order to guarantee peace.] I assume the values of $\left\{ c_i \right\}_{ i \in \left\{ 1, \dots, N \right\} }$ are large enough to deliver peace and restrict attention to these equilibria. I can then analyze the subgame consisting policy announcements only, while ensuring that inequality \ref{eq:AwarConstraint} holds for every attacker $k$ and potential target $i$.

Optimal trade policies proposals for $i$ in this case solve
\begin{equation} \label{eq:tauTildeStar}
\begin{split}
\max_{ \tilde{\bm{\tau}}_i } & \quad \tilde{G}_i(\tilde{\bm{\tau}}_i; \tilde{\bm{\tau}}_{-i}) \\
\text{subject to} & \quad \tilde{G}_j(\tilde{\bm{\tau}}) - G_j(\bm{\tau}_i^{j \star}; \tilde{\bm{\tau}}_{-i}) + c \tilde{\chi}_{ji}(\bm{Z}; \bm{\theta}_m)^{-1} \geq 0 \quad \text{for all } j \neq i
\end{split}
\end{equation}
where the constraints can be derived by rearranging \ref{eq:AwarConstraint}. Formulated in this manner, it becomes clear that military allocations affect trade policy through their effect on the $i$'s war constraints. As $M_j$ increases, $\chi_{ji}$ increases as well (for $\gamma > 0$), tightening the constraint on $i$'s policy choice. Let $\tilde{\bm{\tau}}_i^\star(\tilde{\bm{\tau}}_{-i})$ denote a solution to this problem and $\tilde{\bm{\tau}}^\star$ a Nash equilibrium of the constrained policy announcement game.

## Policy Equilibrium in Changes

The equilibrium of the international economy depends on a vector of structural parameters and constants $\bm{\theta}_h$ defined in Appendix `r Aeconomy`. Computing the equilibrium $h(\bm{\tau}; \bm{\theta}_h)$ requires knowing these values. Researchers have the advantage of observing data related to the equilibrium mapping for one particular $\bm{\tau}$, the factual trade policies. 

The estimation problem can be therefore partially ameliorated by computing the equilibrium in *changes*, relative to a factual baseline. Consider a counterfactual trade policy $\tau_{ij}^\prime$ and its factual analogue $\tau_{ij}$. The counterfactual policy can be written in terms of a proportionate change from the factual policy with $\tau_{ij}^\prime = \hat{\tau}_{ij} \tau_{ij}$ where $\hat{\tau}_{ij} = 1$ when $\tau_{ij}^\prime = \tau_{ij}$. By rearranging the equilibrium conditions, I can solve the economy in changes, replacing $h(\bm{\tau}; \bm{\theta}_h) = \bm{w}$ with $\hat{h}(\hat{\bm{\tau}}; \bm{\theta}_h) = \hat{\bm{w}}$. Counterfactual wages can the be computed as $\bm{w}^\prime = \bm{w} \odot \hat{\bm{w}}$.

This method is detailed in Appendix `r Aeconomy`. Because structural parameters and unobserved constants do not change across equilibria, variables that enter multiplicatively drop out of the equations that define this "hat" equilibrium. This allows me to avoid estimating these variables, while enforcing that the estimated equilibrium is consistent with their values. The methodology, introduced by @Dekle2007, is explicated further in @Costinot2015 and used to study trade policy changes in @Ossa2014 and @Ossa2016.

It is straightforward to extend this methodology to the game studied here. Consider a modification to the policy-setting subgamein which governments propose changes to factual trade policies $\hat{\tilde{\bm{\tau}}}$ and call this game $\Gamma^{\hat{\bm{\tau}}}$. Note that this modification is entirely cosmetic -- the corresponding equilibrium of $\Gamma^{\hat{\bm{\tau}}}$ in levels can be computed by multiplying factual policies by the "hat" equilibrium values ($\tau_{ij}^\prime = \hat{\tau}_{ij} \tau_{ij}$). I can then replace the equilibrium conditions of $\Gamma^{\bm{\tau}}$ with their analogues in changes. Each government's welfare (\ref{eq:G}) in changes is
\begin{equation} \label{eq:Ghat}
\hat{G}_i(\hat{\bm{\tau}}) = \hat{V}_i \left( \hat{h}(\hat{\bm{\tau}}) \right)^{1 - b_i} \hat{r}_i \left(\hat{h}(\hat{\bm{\tau}}) \right)^{b_i}
\end{equation}
and its utility function is
\begin{equation}
\hat{\tilde{G}}_i(\hat{\bm{\tau}}) = \hat{G}_i(\hat{\bm{\tau}}; b_i) + \sum_{j \neq i} \xi_{ij} \frac{G_j(\bm{\tau})}{G_i(\bm{\tau})} \hat{G}_i(\hat{\bm{\tau}}; b_j) 
\end{equation}
.

Optimal policy changes for governments successful in wars (\ref{eq:optTauj}) are denoted $\hat{\bm{\tau}}_i^{j \star}$ and satisfy
\begin{equation} \label{eq:optTaujHat}
\begin{split}
\max_{\hat{\bm{\tau}}_j} & \quad \hat{G}_i(\hat{\bm{\tau}}_j; \hat{\tilde{\bm{\tau}}}_{-j}) \\
\text{subject to} & \quad \hat{\tau}_{jj} = 1
\end{split}
\end{equation}
By dividing the governments' war constraints (\ref{eq:AwarConstraint}) by their factual welfare $G_j(\bm{\tau})$, their constrained policy announcement problem can be rewritten as the solution to
\begin{equation} \label{eq:tauTildeStarHat}
\begin{split}
\max_{ \hat{\tilde{\bm{\tau}}}_i } & \quad \hat{\tilde{G}}_i(\hat{\tilde{\bm{\tau}}}_i; \hat{\tilde{\bm{\tau}}}_{-i}) \\
\text{subject to} & \quad \hat{\tilde{G}}_j(\hat{\tilde{\bm{\tau}}}) - \hat{G}_j(\hat{\bm{\tau}}_i^{j \star}) + \hat{c} \left( \chi_{ji}(1; \bm{0}_{-j, -i}, \bm{m}) \right)^{-1} \geq 0 \quad \text{for all } j \neq i
\end{split}
\end{equation}
where
$$
\hat{c} = \frac{c_i}{G_i(\bm{\tau}; b_i)}
$$
is the *share* of factual utility each government pays if a war occurs. Assumption `r Achat` requires that governments pay the same share of their factual utility in any war.^[While not innocuous, this assumption is more tenable than assumption constant absolute costs. It formalizes the idea that larger countries (that collect more rents and have high real incomes than their smaller counterparts) also pay more in military operations. It avoids the complications inherent in the more realistic but less tractable assumption that war costs depend on power ($\chi$).]

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
Achat <- Atick
Atick <- Atick + 1

AchatText <- knit_child("../results/Achat.md")
```

**Assumption `r Achat` (Constant Relative War Costs):** `r AchatText`