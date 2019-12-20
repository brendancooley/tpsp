There are $N$ governments, indexed $i \in \left\{ 1, ..., N \right\}$. Governments choose trade policies $\bm{\tau}_i = \left\{ \tau_{i1}, ..., \tau_{iN} \right\} \in [1, \bar{\tau}]^N$ which affect their welfare indirectly through changes in the international economy.^[$\bar{\tau}$ is an arbitarily large but finite value sufficient to shut down trade between any pair of countries.] An entry of the trade policy vector, $\tau_{ij}$ is the cost country $i$ imposes on imports from $j$.^[Costs enter in an "iceberg" fashion, and I normalize $\tau_{ii} = 1$. Then, if the price of a good in country $j$ is $p_{jj}$, its cost (less freight) in country $i$ is $\tau_{ij} p_{jj}$. The ad valorem tariff equivalent of the trade policy is $t_{ij} = \tau_{ij} - 1$. I employ structural estimates of these costs from @Cooley2019b to estimate the model, which are described in more detail in Appendix `r Aeconomy`.] The economy, detailed in Appendix `r Aeconomy`, can be succinctly characterized by a function $h: \bm{\tau} \rightarrow \mathbb{R}_{++}^N$ mapping trade policies to wages in each country, denoted $\bm{w} = \left\{ w_1, ..., w_N \right\}$. These in turn determine trade flows between pairs of countries and prices around the world.^[The economy is a variant of the workhorse model of @Eaton2002.]

Government welfare depends on these general equilibrium responses to trade policy choices. Governments value the welfare of a representative consumer that resides within each country and rents accrued through trade policy distortions (tariff revenues). These can be computed given knowledge of the general equilibrium function $h(\bm{\tau})$. Formally, government utility is 
\begin{equation} \label{eq:G}
G_i(\bm{\tau}; b_i) = V_i \left( h(\bm{\tau}) \right)^{1 - b_i} r_i \left(h(\bm{\tau}) \right)^{b_i}
\end{equation}
where $V_i(h(\bm{\tau}))$ is consumer's indirect utility, $r_i(h(\bm{\tau}))$ are tariff revenues, and $b_i$ is a structural parameter that governs the government's relative preference for these. When the government values the welfare of the consumer ($b_i = 0$), it prefers to set "optimal tariffs" in the sense of @Johnson1953. Tariffs higher than these hurt consumers but raise more revenue for the government. When the government values rents ($b_i = 1$), it prefers trade policies that maximize revenue where
\begin{equation} \label{eq:r}
r_i(h(\bm{\tau})) = \sum_j (\tau_{ij} - 1) X_{ij}(h(\bm{\tau}))
\end{equation}
and $X_{ij}(h(\bm{\tau}))$ are country $i$'s imports from country $j$.^[This object does not correspond empirically to governments' factual tariff revenues, as $\tau_{ij}$ incorporates a larger set of trade policy distortions than tariffs alone. Yet, non-tariff barriers to trade also generate rents that do not accrue directly to the government's accounts (see, for example, @Anderson1992 for the case of quotas). This revenue function is designed to capture this broader set of rents.]

Characterizing the theoretical properties of this objective function is challenging due to the network effects inherent in general equilibrium trade models. Suppose $i$ increases barriers to trade with $j$. This affects wages and prices in $j$ and its competitiveness in the world economy. These changes affect its trade with all other countries $k$, which affects their welfare, and so on [@Allen2019]. In order to make progress, I assume that $G_i$ is quasiconcave in $i$'s own policies, which ensures the existence of an interior, unconstrained optimal policy vector and a pure-strategy Nash equilibrium to the simultaneous policy-setting game.^[This property holds in my numerical applications, but may not hold more generally. The existence of a Nash equilibrium follows from the fixed point theorem of Debreu, Glicksberg, and Fan [@Fudenberg1992 p. 34].]

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
AG <- Atick
Atick <- Atick + 1

AGText <- knit_child("../results/AG.md")
```

**Assumption `r AG`**: `r AGText`

These optimal policies impose externalities on other governments. By controlling the degree of market access afforded to foreign producers, trade policies affect the wages of foreign workers and the welfare of the governments that represent them. They also partially determine trade flows, which affect other governments' ability to collect rents. In this sense, protectionism is "beggar they neighbor." Governments policy proposals are denoted $\tilde{\bm{\tau}}$.

Governments' military strategies seek to alter other governments' incentives in order to ameliorate these externalities. Each government is endowed with military capacity $M_i$ which can be allocated toward potential wars with other governments. A given government $i$ can only employ this military force in a war against another government $j$ if it has first allocated effort to this task. A military allocation is a vector $\bm{m}_i = \left\{ m_{i1}, ..., m_{iN} \right\}$ where each entry $m_{ij}$ represents the amount of force $i$ dedicates to a potential war against $j$. $m_{ii}$ is the amount of force $i$ reserves for self-defense. In total, these allocations must not exceed $i$'s military endowment, $\sum_j m_{ij} \leq M_i$. The matrix $\bm{m}$ stores all governments' military allocations, stacked row-wise. Each column of this matrix $\bm{m}^i$ stores the effort levels others' have invested in threatening $i$. Governments pay an arbitrarily small cost $\epsilon^{m}$ to allocate force against other governments.^[This ensures that governments only allocate force when doing so secures policy concessions.]

Governments simultaneously set military allocations *before* trade policy announcements (which also occur simultaneously). After military allocations are set and trade policies announcements are made, governments decide whether or not they would like to wage war against other governments. Wars are fought in order to impose more favorable trade policies abroad. Wars are offensive and *directed*. Formally, let $\bm{a}_i = \left\{ a_{i1}, ..., a_{iN} \right\}$ denote $i$'s war entry choices, where $a_{ij} \in \left\{ 0 , 1 \right\}$ denotes whether or not $i$ choose to attack $j$.^[Note that this formulation leaves open the possibility that two governments launch directed wars against one another, $a_{ij} = a_{ji} = 1$.] $a_{ii} = 1$ for all $i$ by assumption -- governments always choose to defend themselves.

If $i$ is successful in defending itself against all attackers, its announced policies are implemented. Government $i$ suffers a cost $c_i$ for each war it must fight, accruing total war costs $\sum_{j \neq i} a_{ij} c_i$. Each attacker $j$ also pays $c_j$. When a government $i$ wins a war against $j$, it earns the right to dictate $j$'s trade policy. Optimal policies for a victorious government $j$ are denoted $\bm{\tau}_j^{i \star}$ and solve
\begin{equation} \label{eq:optTauj}
\begin{split}
\max_{\bm{\tau}_j} & \quad G_i(\bm{\tau}_j; \tilde{\bm{\tau}}_{-j}) \\
\text{subject to} & \quad \tau_{jj} = 1
\end{split}
\end{equation}

Governments' ability to prosecute wars against one another depend on dyadic geographic factors $\bm{W}$, such as geographic distance. For every unit of force $i$ allocates toward attacking $j$, $\rho_{ij}(\bm{W}; \bm{\alpha}) \in [0,1]$ units arrive. I normalize $\rho_{jj} = 1$ -- defensive operations do not result in any loss of strength. $\bm{\alpha}$ is a vector of structural parameters governing this relationship to be estimated. War outcomes are determined by a contest function 
\begin{equation} \label{eq:chi}
\chi_{ij}(\bm{a}, \bm{m}) = \frac{ a_{ij} \rho_{ij}(\bm{W}; \bm{\alpha}) m_{ij} }{ \sum_k a_{kj} \rho_{kj}(\bm{W}; \bm{\alpha}) m_{kj} }
\end{equation}.
$\chi_{ij}$ is the probability that $i$ is successful in an offensive war against $j$. 

For the moment, fix $a_{jk} = 0$ for all $k \neq i$, $j \neq k$. $i$ is the only government that faces the possibility of attack. Then, all other policy proposal vectors $\tilde{\bm{\tau}}_{-i}$ are implemented with certainty and $i$'s utility as a function of war entry decisions is 
$$
G_i^{\bm{a}}(\bm{a}) = \chi_{ii}(\bm{a}) G_i(\tilde{\bm{\tau}}) + \sum_{j \neq i} \left( \chi_{ji}(\bm{a}) G_i(\bm{\tau}_i^{j \star}; \tilde{\bm{\tau}}_{-i}) - a_{ji} c_i \right)
$$

Attackers consider the effect of their war entry on the anticipated policy outcome. Now consider an attacker $j$'s war entry decision vis-à-vis a defender $i$, assuming no other country launches a war. Government $j$ prefers not to attack $i$ so long as
\begin{equation} \label{eq:AwarConstraint}
G_j(\tilde{\bm{\tau}}) \geq \chi_{ji}(1; \bm{0}_{-j, -i}) G_j(\bm{\tau}_i^{j \star}; \tilde{\bm{\tau}}_{-j}) + \left( 1 - \chi_{ji}(1; \bm{0}_{-j, -i}) \right) G_j(\tilde{\bm{\tau}}) - c_j
\end{equation}
where $\chi_{ji}(1; \bm{0}_{-j, -i})$ is the probability $j$ is successful when it attacks $i$, enforcing peace on other potential war entrants. Let $\bm{a}^\star : \tilde{\bm{\tau}} \rightarrow \left\{ 0, 1 \right\}_{N - 1 \times N - 1}$ denote equilibrium war entry decisions as a function of announced policies and $a_{ij}^\star(\tilde{\bm{\tau}})$ denote an element of this set. Governments choose whether or not to enter wars simultaneously. When peace prevails, $a_{ij}^\star(\tilde{\bm{\tau}}) = 0$ for all $i \neq j$. 

To recap, governments allocate military forces $\bm{m}$, make policy announcements, $\tilde{\bm{\tau}}$, and then launch wars $\bm{a}$.  At each stage, actions are taken simultaneously. The solution concept is subgame perfect equilibrium. In Appendix `r AwarEntry`, I show that there exist peaceful equilibria of the subgame consisting of policy announcements and war entry decisions ($\Gamma^{\bm{\tau}}$) as long as war costs are large enough (Proposition `r Pc`).^[This result mirrors @Fearon1995's proof of the existence of a bargaining range in a unidimensional model. Here, because the governments' objective functions are not necessarily concave, war costs may need to be larger in order to guarantee peace.] I assume $c_i$ is large enough and restrict attention to these equilibria. I can then analyze the subgame consisting of military allocations and policy announcements only, while ensuring that inequality \ref{eq:AwarConstraint} holds for every attacker $k$ and potential target $i$. Call this game $\Gamma^{\bm{m}}$. Then, $\chi_{ji}(1; \bm{0}_{-j, -i})$ can be written $\chi_{ji}(\bm{m})$ representing the probability $j$ wins a war against $i$ when no other country engages in war against $i$.

Given military allocations $\bm{m}$, optimal trade policies for $i$ then solve
\begin{equation} \label{eq:tauTildeStar}
\begin{split}
\max_{ \tilde{\bm{\tau}}_i } & \quad G_i(\tilde{\bm{\tau}}_i; \tilde{\bm{\tau}}_{-i}) \\
\text{subject to} & \quad G_j(\tilde{\bm{\tau}}) - G_j(\bm{\tau}_i^{j \star}) + c \left( \chi_{ji}(\bm{m}) \right)^{-1} \geq 0 \quad \text{for all } j \neq i
\end{split}
\end{equation}
where the constraints can be derived by rearranging \ref{eq:AwarConstraint}. Let $\mathcal{L}_i^{\bm{\tau}}(\tilde{\bm{\tau}}_i, \bm{m}; \bm{\lambda}^{\chi})$ denote the Lagrangian associated with this problem, where $\lambda_{ij}^{\chi}$ corresponds to the $j$th Lagrange multiplier in $i$'s Legrangian. Formulated in this manner, it becomes clear that military allocations affect trade policy through their effect on the $i$'s war constraints. As $m_{ji}$ increases, $\chi_{ji}$ increases as well, tightening the constraint on $i$'s policy choice. Let $\tilde{\bm{\tau}}_i^\star(\bm{m})$ denote a solution to this problem and $\tilde{\bm{\tau}}^\star(\bm{m})$ a Nash equilibrium of the constrained policy announcement game.

Optimal military allocations then solve
\begin{equation} \label{eq:mStar}
\begin{split}
\max_{\bm{m}_i} & \quad G_i \left(\tilde{\bm{\tau}}^\star(\bm{m}_i; \bm{m}_{-i}) \right) - \sum_{j \neq i} m_{ij} \epsilon^m \\
\text{subject to} & \quad \sum_j m_{ij} \leq M_i
\end{split}
\end{equation}

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
Deq <- Dtick
Dtick <- Dtick + 1

DeqText <- knit_child("../results/Deq.md")
```

**Definition `r Deq`:** `r DeqText`

Proposition `r Pm` states that constraints in problem \ref{eq:tauTildeStar} will hold with equality in equilibrium whenever $m_{ji}^\star > 0$. The logic behind this result is simple. Consider a defender $i$'s war constraint vis a vis a threatening government $k$. If $i$'s war constraint vis a vis $k$ does not bind, then its policy choice is unaffected by $k$'s threats. Since $k$ pays a small cost $\epsilon^m$ to allocate military force to threaten $i$, it can profitably reallocate this force to defense. $k$'s military allocation is therefore inconsistent with equilibrium.

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
Pm <- Ptick
Ptick <- Ptick + 1

PmText <- knit_child("../results/Pm.md")
```

**Proposition `r Pm`:** `r PmText`

**Proof:** See Appendix `r Aproofs`.

## Policy Equilibrium in Changes

The equilibrium of the international economy depends on a vector of structural parameters and constants $\bm{\theta}_h$ defined in Appendix `r Aeconomy`. Computing the equilibrium $h(\bm{\tau}; \bm{\theta}_h)$ requires knowing these values. Researchers have the advantage of observing data related to the equilibrium mapping for one particular $\bm{\tau}$, the factual trade policies. 

The estimation problem can be therefore partially ameliorated by computing the equilibrium in *changes*, relative to a factual baseline. Consider a counterfactual trade policy $\tau_{ij}^\prime$ and its factual analogue $\tau_{ij}$. The counterfactual policy can be written in terms of a proportionate change from the factual policy with $\tau_{ij}^\prime = \hat{\tau}_{ij} \tau_{ij}$ where $\hat{\tau}_{ij} = 1$ when $\tau_{ij}^\prime = \tau_{ij}$. By rearranging the equilibrium conditions, I can solve the economy in changes, replacing $h(\bm{\tau}, \bm{D}; \bm{\theta}_h) = \bm{w}$ with $\hat{h}(\hat{\bm{\tau}}, \hat{\bm{D}}; \bm{\theta}_h) = \hat{\bm{w}}$. Counterfactual wages can the be computed as $\bm{w}^\prime = \bm{w} \odot \hat{\bm{w}}$.

This method is detailed in Appendix `r Aeconomy`. Because structural parameters and unobserved constants do not change across equilibria, variables that enter multiplicatively drop out of the equations that define this "hat" equilibrium. This allows me to avoid estimating these variables, while enforcing that the estimated equilibrium is consistent with their values. The methodology, introduced by @Dekle2007, is explicated further in @Costinot2015 and used to study trade policy changes in @Ossa2014 and @Ossa2016.

It is straightforward to extend this methodology to the game ($\Gamma^{\bm{m}}$) studied here. Consider a modification to the policy-setting subgame ($\Gamma^{\bm{\tau}}$) in which governments propose changes to factual trade policies $\hat{\tilde{\bm{\tau}}}$ and call this game $\Gamma^{\hat{\bm{\tau}}}$. Note that this modification is entirely cosmetic -- the corresponding equilibrium of $\Gamma^{\hat{\bm{\tau}}}$ in levels can be computed by multiplying factual policies by the "hat" equilibrium values ($\tau_{ij}^\prime = \hat{\tau}_{ij} \tau_{ij}$). I can then replace the equilibrium conditions of $\Gamma^{\bm{\tau}}$ with their analogues in changes. The governments' objective functions (\ref{eq:G}) in changes are
\begin{equation} \label{eq:Ghat}
\hat{G}_i(\hat{\bm{\tau}}; b_i) = \hat{V}_i \left( \hat{h}(\hat{\bm{\tau}}) \right)^{1 - b_i} \hat{r}_i \left(\hat{h}(\hat{\bm{\tau}}) \right)^{b_i}
\end{equation}
Optimal policy changes for governments successful in wars (\ref{eq:optTauj}) are denoted $\hat{\bm{\tau}}_i^{j \star}$ and satisfy
\begin{equation} \label{eq:optTaujHat}
\begin{split}
\max_{\hat{\bm{\tau}}_j} & \quad \hat{G}_i(\hat{\bm{\tau}}_j; \hat{\tilde{\bm{\tau}}}_{-j}) \\
\text{subject to} & \quad \hat{\tau}_{jj} = 1
\end{split}
\end{equation}
By dividing the governments' war constraints (\ref{eq:AwarConstraint}) by their factual utility $G(\bm{\tau}; b_i)$, their constrained policy announcement problem can be rewritten as the solution to
\begin{equation} \label{eq:tauTildeStarHat}
\begin{split}
\max_{ \hat{\tilde{\bm{\tau}}}_i } & \quad \hat{G}_i(\hat{\tilde{\bm{\tau}}}_i; \hat{\tilde{\bm{\tau}}}_{-i}) \\
\text{subject to} & \quad \hat{G}_j(\hat{\tilde{\bm{\tau}}}) - \hat{G}_j(\hat{\bm{\tau}}_i^{j \star}) + \hat{c} \left( \chi_{ji}(1; \bm{0}_{-j, -i}, \bm{m}) \right)^{-1} \geq 0 \quad \text{for all } j \neq i
\end{split}
\end{equation}
where
$$
\hat{c} = \frac{c_i}{G_i(\bm{\tau}; b_i)}
$$
is the *share* of factual utility each government pays if a war occurs. Let $\mathcal{L}_i^{\hat{\bm{\tau}}}(\hat{\tilde{\bm{\tau}}}_i, \bm{m}; \bm{\lambda}^{\chi})$ denote the Lagrangian associated with this problem. Assumption `r Achat` requires that governments pay the same share of their factual utility in any war.^[While not innocuous, this assumption is more tenable than assumption constant absolute costs. It formalizes the idea that larger countries (that collect more rents and have high real incomes than their smaller counterparts) also pay more in military operations. It avoids the complications inherent in the more realistic but less tractable assumption that war costs depend on power ($\chi$).]

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
Achat <- Atick
Atick <- Atick + 1

AchatText <- knit_child("../results/Achat.md")
```

**Assumption `r Achat` (Constant Relative War Costs):** `r AchatText`

Military allocations (\ref{eq:mStar}) are then designed to induce favorable *changes* in trade policy abroad, solving
\begin{equation} \label{eq:mStarHat}
\begin{split}
\max_{\bm{m}_i} & \quad \hat{G}_i \left(\hat{\tilde{\bm{\tau}}}^\star(\bm{m}_i; \bm{m}_{-i}) \right) - \sum_{j \neq i} m_{ij} \epsilon^m \\
\text{subject to} & \quad \sum_j m_{ij} \leq M_i
\end{split}
\end{equation}
Let $\mathcal{L}_i^{\bm{m}}(\bm{m}; \bm{\lambda}^{\bm{m}})$ denote the Lagrangian associated with this problem and the modified military allocation game $\hat{\Gamma}^{\bm{m}}$. 

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
DeqHat <- Dtick
Dtick <- Dtick + 1

DeqHatText <- knit_child("../results/DeqHat.md")
```

**Definition `r DeqHat`:** `r DeqHatText`