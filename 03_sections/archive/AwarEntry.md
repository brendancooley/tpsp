Consider the policy announcement and war subgame of the model articulated above in which military allocations $\bm{m}$ are fixed. Governments first simultaneously make policy announcements $\tilde{\bm{\tau}}$. Observing these announcements, governments then make war entry decisions $\bm{a}$. These depend on war costs $c_i$. Let $\bm{c} = \left\{ c_1, ..., c_N \right\}$ Let the set of all such games be denoted $\Gamma^{\bm{\tau}}(\bm{c})$.

Government $i$'s best response function to a trade policy announcement can be written $a_i^\star(\tilde{\bm{\tau}}; \bm{a}_{-j}) \in \left\{ 0, 1 \right\}^{N - 1}$. Let $\varphi_i \in \left\{ 0, 1 \right\}_{N - 1} = \Phi$ denote the set of possible war outcomes for a given attacker $i$, where $\varphi_{ij} = 1$ if $i$ is successful in prosecuting a war against $j$. Fix $a_{k, j} = 0$ for all $k \neq i$ $j \neq k$ -- $i$ is the only government that can attack others. Then, policies can be written as a function of war outcomes as follows
$$
\bm{\tau}^{\varphi_i}(\tilde{\bm{\tau}}) = \tilde{\bm{\tau}}_i \cup \left\{ \varphi_{ij} \bm{\tau}_j^{i \star} + (1 - \varphi_{ij}) \tilde{\bm{\tau}_j} \right\}_{j \neq i}
$$
The probability of outcome $\varphi_i$ is 
$$
\text{Pr}(\varphi_i; \bm{a}_i) = \prod_{j \neq i} \left( \varphi_j \chi_{ij}(\bm{a}_i) + (1 - \varphi_j) (1 - \chi_{ij}(\bm{a}_i) \right)
$$
Then, enforcing peace elsewhere, $i$'s utility for a given war entry vector can be written
$$
G_i^{\bm{a}}(\bm{a}_i) = \sum_{\varphi_i \in \Phi} \text{Pr} \left( \varphi_i; \bm{a}_i \right) G_i \left( \bm{\tau}^\varphi_i(\tilde{\bm{\tau}}) \right) - \sum_{j} a_{ij} c_i
$$
Government $i$'s best response condition when peace is enforced elsewhere can then be written
\begin{equation} \label{eq:BRa}
a_i^\star(\tilde{\bm{\tau}}; \bm{0}_{-j}) \in \argmax_{\bm{a}_i} G_i^{\bm{a}}(\bm{a}_i)
\end{equation}

Now let $\varphi^j \in \left\{ 0, 1 \right\}_{N - 1} = \Phi$ with $\sum \varphi^{ji} \leq 1$ denote a war outcome for a defending government $j$.^[The constraint reflects the fact only one country can be successful in a war against $j$.] Policy outcomes are 
$$
\bm{\tau}^{\varphi^j}(\tilde{\bm{\tau}}) = \left\{ \tilde{\bm{\tau}}_i \right\}_{i \neq j} \cup \left\{ \varphi^{ji} \bm{\tau}_j^{i \star} + \left( 1 - \sum_k \varphi^{jk} \right) \tilde{\bm{\tau}}_j \right\}_i
$$
Now, assume $j$ is the only country that faces the possibility of attack -- $a_{k, i} = 0$ for all $i \neq j$, $k \neq i$. Then, $j$'s utility for a given offer can be written
$$
G_j^{\bm{\tau}}(\tilde{\bm{\tau}}_j; \tilde{\bm{\tau}}_{-j}) = \sum_{\varphi^j \in \Phi} \text{Pr} \left( \varphi^j; \bm{a}^\star(\tilde{\bm{\tau}}_j) \right) G_j \left( \bm{\tau}^{\varphi^j}(\tilde{\bm{\tau}}) \right) - \sum_i \bm{a}^\star(\tilde{\bm{\tau}}_j) c_j
$$
and its best response condition is
\begin{equation} \label{eq:BRtau}
\tilde{\bm{\tau}}_j^\star(\tilde{\bm{\tau}}_{-j}) \in \argmax_{\tilde{\bm{\tau}}_j} G_j^{\bm{\tau}}(\tilde{\bm{\tau}}_j; \tilde{\bm{\tau}}_{-j})
\end{equation}

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
DGammac <- paste0(LETTERS[a], Dtick)
Dtick <- Dtick + 1

DGammacText <- knit_child("../results/DGammac.md")
```

**Definition `r DGammac`:** `r DGammacText`

```{r, echo=FALSE, warning=FALSE, message=FALSE, results='hide'}
Pc <- paste0(LETTERS[a], Ptick)
Ptick <- Ptick + 1

PcText <- knit_child("../results/Pc.md")
PcProof <- knit_child("../proofs/Pc.md")
```

**Proposition `r Pc`:** `r PcText`

**Proof:** `r PcProof`

