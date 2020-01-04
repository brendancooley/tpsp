Solving each stage of the game is computationally expensive, however. For a given parameter vector, computing the equilibrium of $\hat{\Gamma}^{\bm{\tau}}$ computing the Nash Equilibrium of the constrained policy announcement game $\hat{\tilde{\bm{\tau}}}^\star(\bm{\theta}_m)$. In turn, computing these requires solving the equilibrium of the economy $\hat{h}(\hat{\bm{\tau}}; \bm{\theta}_h)$ for many trial $\bm{\tau}$. To ease computation, I recast governments' constrained policy problem (\ref{eq:tauTildeStarHat}) and the parameter estimation problem as mathematical programs with equilibrium constraints (MPECs) [@Su2012].

Converting \ref{eq:tauTildeStarHat} to an MPEC requires allowing the government to choose policies $\hat{\tilde{\bm{\tau}}}$ and wages $\hat{\bm{w}}$ while satisfying other governments' war constraints and enforcing that wages are consistent with international economic equilibrium $\hat{h}$.^[@Ossa2014 and @Ossa2016 also study optimal trade policy using this methodology.] Let $\hat{\bm{x}}_i = \left\{ \hat{\tilde{\bm{\tau}}}_i, \hat{\bm{w}} \right\}$ store $i$'s choice variables in this problem. Then, noting explicitly dependencies on $\bm{\theta}_m$, \ref{eq:tauTildeStarHat} can be rewritten
\begin{equation} \label{eq:tauTildeHatMPEC}
\begin{split}
\max_{\hat{\bm{x}}_i} & \quad \hat{G}_i(\hat{\bm{w}}; \bm{\theta}_m) \\
\text{subject to} & \quad \hat{G}_j(\hat{\bm{w}}; \bm{\theta}_m) - \hat{G}_j \left( \hat{\bm{\tau}}_i^{j \star}(\bm{\theta}_m) \right) + \hat{c} \left( \chi_{ji}(\bm{m}, \bm{\theta}_m) \right)^{-1} \geq 0 \quad \text{for all } j \neq i \\
& \quad \hat{\bm{w}} = \hat{h}(\hat{\tilde{\bm{\tau}}})
\end{split}
\end{equation}