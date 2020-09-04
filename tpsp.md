---
title: | 
	| Trade Policy in the Shadow of Power
	| \tiny \hfill
    | \Large Quantifying Military Coercion in the International System
author:
	- name: Brendan Cooley
      affiliation: Ph.D. Candidate, Department of Politics, Princeton University
date: 28 July 2020
abstract: In international relations, how does latent military coercion affect governments’ policy choices? Because militarily powerful governments can credibly threaten to impose their policy preferences by force, weaker governments may adjust their policy choices to avoid costly conflict. This setting raises an inference problem -- do observed policies reflect the preferences of the governments that adopted them or the military constraints of the anarchic international system? Here, I investigate the role of this “shadow of power” in determining trade policy. Specifically, I build a model of trade policy choice under threat that allows me to measure empirically governments’ underlying trade policy preferences, the returns to military advantage, and the extent to which power projection capacity degrades across space. I then estimate the parameters of the model using data on governments' observed trade policies in 2011. I find that geographic distance is not an impediment to the projection of force but that there are increasing returns to military advantage in the technology of coercion. Through counterfactual experiments, I quantify the effect of military constraints on the international economy and governments' welfare. These and other exercises simultaneously shed light on how military power affects international economic exchange, and how patterns of trade and protectionism affect the governments' propensity to engage in military conflict. 
thanks: Thanks to Geneveive Bates, Allison Carnegie, Jim Fearon, Haosen Ge, Mike Gibilisco, Kyle Haynes, Helen Milner, Sayumi Miyano, Steve Monroe, In Young Park, Jim Qian, Kris Ramsay, and Joe Ruggiero for comments and discussions on many earlier versions of this project. Thanks also to audiences at the American Political Science Association's 2019 annual meeting and the 2020 conference on Formal Models of International Relations.
# jelcodes: JEL codes go here

bibliography: bib/library.bib
biblio-style: apsr

papersize: letter
documentclass: article
geometry: margin=1.25in
link-citations: true

output:
	fig_caption: yes
    citation_package: natbib

---



# Introduction




Military power holds a central position in international relations (IR) theory. Governments exist in a state of anarchy — there is no world authority tasked with preventing the use of violence in settling policies disputes between them. As a result, powerful governments can employ force against others to secure more favorable policy outcomes. This does not necessarily imply that international relations are uniquely violent, however. Threatened governments can adjust their policy choices to accommodate the interests of the powerful, avoiding costly conflict [@Brito1985; @Fearon1995]. This setting raises an inference problem — do observed policies reflect the preferences of the governments that adopted them, or the military constraints of the anarchic international system?

In this paper, I propose and implement a method to assess the effect of military power on trade policy choices. Trade is a natural issue area in which to undertake such an investigation. For a variety of reasons, governments' endeavor to protect their home market to some extent. Governments also seek access to foreign markets [@Grossman2016]. These preferences put governments into conflict with one another -- each would like to erect some barriers to imports while dismantling barriers to trade abroad. Given dictatorial power, governments would protect their home market and enforce openness elsewhere. Moreover, aggregate policy-induced trade frictions are large [@Cooley2019b] and have large effects on the distribution and level of welfare within and across countries [@Autor2013; @Costinot2015; @Goldberg2016]. These effects may be particularly salient for politically influential groups [@Grossman1994; @Osgood2017]. Governments therefore have incentives to use force to shape trade policy abroad to their liking. Historically, they have been willing to fight wars to realize such goals [@Findlay2007].

Assessing the effect of military power on trade policy requires imagining what policy choices governments would have made in the absence of coercion. Uncoerced policies can be taken as a measure of the government's ideal point, representing its true underlying preferences. When coercion is possible, however, weaker governments must consider the effect of their policy choices on the powerful. If a particular policy choice harms a threatening government enough, it can choose to impose an alternative policy by force. Recognizing this threat, weaker governments adjust their policies to avoid war. In an anarchic world, policies are jointly determined by both power and preferences.  

I proceed in three steps to untangle power and preferences as determinants of trade policies. First, I model a coercive international political economy in which governments propose trade policies, observe others proposals, and choose whether or not to fight wars to win the right to modify these. The model's equilibrium depends on a vector of parameters governing governments' preferences for protectionism and costs of war, which in turn depend on the military strengths of potential belligerents and the geographic distance between them. I then estimate these parameters by minimizing the distance between the model's predictions and observed policies in the year 2011. Finally, I answer the question posed here: how does military coercion affect trade policy? With estimates for the model's parameters in hand, this question can be answered by a simple counterfactual experiment — eliminate governments' military capacity, and recalculate the model's equilibrium. The difference between counterfactual equilibrium policies and the factual policies represents the net effect of military coercion on trade policy.

I find that there are increasing returns to military advantage in international trade policy bargaining. Governments that are militarily powerful are estimated to enjoy lower costs of war, which they exploit to coerce policy concessions from their trade partners. In this sense, military might is a force for trade liberalization, inducing reductions in barriers to trade that governments would be unwilling to undertake in the absence of coercion. These reductions in barriers to trade stimulate international economic exchange. Counterfactually eliminating militaries reduces the value of global trade to 61.4 percent of its model-estimated value. I estimate that the effectiveness of military coercion does not degrade across space -- in fact, governments are estimated to enjoy lower average costs of war against geographically distant adversaries. This may reflect the peculiarities of the technology of war in the era under study, in which geographic distance represents a minimal impediment to the projection of power. 

In the model, governments choose trade policies to maximize a country-specific social welfare function. Each government's trade policy is a set of taxes, one for each importing country, imposed on imports. Notably, trade policies can be discriminatory, affecting certain source countries disproportionately. A model of the international economy connects trade policy choices to social welfare.^[The model of the international economy is a variant of the workhorse model of @Eaton2002. @Costinot2015 study a broader class of structural gravity models that connect trade frictions (such as trade policy) to trade and welfare outcomes.] After observing trade policies, governments may choose to fight wars against other governments in order to impose free trade. The threat of war constrains threatened governments and affects their trade policy choices. The dyadic costs of war, held as private information to potential attackers, depend on observable features of the attacking and defending countries, including the potential belligerents' relative military strengths' and the geographic distance between them. Governments' ideal policies depend on a country-specific parameter governing the ease with which policy distortions are converted into revenues. I show that these preference parameters and the elasticities that convert military strength and geographic distance into war costs can be estimated given data on directed policy barriers to trade -- or the magnitude of policy distortion each government imposes on imposes on imports from each other country.^[I use data on aggregate directed trade policy distortions from @Cooley2019b, a companion paper to this study. These data are discussed in more detail below.]

Within-country variation in trade policy identifies the model. Consider the ideal set of trade policies of a government whose preferences are known. The policies that achieve this objective can be readily calculated given knowledge of parameters governing international economic relations. Policies adopted toward imports from countries that pose no military threat will reflect this objective. Conversely, the imports of threatening countries will encounter lower barriers to trade, in order to satisfy the threatener's war constraint. This favoritism is informative about the effectiveness of military threats. The level of barriers toward non-threatening countries is informative about the government's preferences. Differential responses to the same level of threat from different geographic locations identifies parameters governing the effectiveness of power projection across space.

The identified model enables two classes of counterfactuals. First, it allows me to quantify the "shadow of power" by comparing factual policies to those that would prevail if governments' counterfactually possessed zero military capability. These policies can then be fed into the model of the international economy to calculate the effect on trade flows, prices, and wages around the world. Would different trade blocs emerge in a coercion-free world? Which governments would benefit the most? In the model, consumers benefit from the liberalizing effect of foreign military coercion [@Antras2011; @Cooley2019a]. How large are these benefits? Whose citizens benefit the most from international power politics? How would relative changes to U.S. and Chinese military strength affect the functioning of the international economy?

I also examine how domestic political economic changes (changes to government preferences) affect the salience of military coercion. Governments that value the welfare of consumers prefer to adopt lower barriers to trade. The returns to coercing these governments are smaller, because their ideal policies impose relatively small externalities on potential threatening governments. Military coercion plays a smaller role in influencing trade policy when governments are relatively liberal. Domestic political institutions are believed to affect trade policy preferences [@Rodrik1995; @Milner1999; @Milner2005]. The model facilitates exploration of how domestic political change affects the quality of international relations and governments' propensity to threaten, display, and use military force against one another. 

# Literature


Conflicts of interest and the specter of coercive diplomacy emerge in the model due to governments' protectionist preferences. Trade theory reserves a role for small trade policy distortions for governments that seek to maximize aggregate societal wealth [@Johnson1953; @Broda2008]. Empirically, governments implement larger trade distortions than predicted in theory, however. This regularity motivated the study of the political economics of trade policy. While nearly free trade may be good for a society as a whole, owners of specific factors of production may prefer protectionism. If these groups have better access to the policymaking process, trade policy may be more protectionist than is optimal for society [@Mayer1984; @Rogowski1987; @Grossman1994]. A family of studies uses these theoretical insights to estimate governments' sensitivity to narrow versus diffuse interests [@Goldberg1999; @Mitra2006; @Gawande2009; @Gawande2012; @Ossa2014; @Gawande2015]. Because these models incorporate no theory of international military coercion, these estimates reflect the assumption that policy choices reflect the outcome of non-cooperative policy choice or non-militarized bargaining. Fiscal pressures might also drive protectionism. Some governments are constrained in their ability to raise revenue through taxes on domestic economic activities. Tariffs and other trade distortions may substitute as a revenue-raising strategy in these cases [@Rodrik2008; @Queralt2015]. 

I take no stance on the domestic political origins of protectionist preferences. I induce these by varying the ease with which governments can collect revenues from trade distortions. Each government is characterized by a revenue threshold parameter. Trade distortions above the level of this threshold generate revenue while distortions below this level require subsidies. Governments with higher threshold parameters therefore prefer higher barriers to trade, all else equal. This simple formulation induces heterogeneity in the governments' ideal levels of protectionism and the magnitude of the externalities they would impose on other governments when choosing individually optimal policies.

These externalities motivate the lobbying efforts of domestic special interests and structure international negotiations over trade policy. In contemporary political economic accounts, large and productive firms pressure their own governments to secure market access abroad in order to increase profit opportunities [@Ossa2012; @Osgood2016; @Kim2017]. By contrast, in my model, lower barriers to trade abroad increase wages at home (helping consumers) and stimulate trade (increasing revenue). Modeling government preferences in this manner captures market access incentives tractably while avoiding ascribing a particular domestic political process to their origin.

Because of these preferences for foreign trade liberalization, governments have incentives to influence others' policy choices. Analyzing governments' foreign policy in the 17th and 18th centuries, @Viner1948 concludes "important sources of national wealth...were available...only to countries with the ability to acquire or retain them by means of the possession and readiness to use military strength." Powerful governments established colonies and threatened independent governments in order to shape policy abroad to their liking [@Gallagher1953]. While formal empires died quickly after World War II, softer forms of influence remained. @Lake2013 terms the resulting order a "hierarchy" in which weaker countries exchanged sovereignty for international political order, provided by a hegemonic United States. @Berger2013 show that this hierarchy has not always been benevolent — U.S. political influence was used to open markets abroad, a form of "commercial imperialism." An earlier literature ascribed international economic openness to the presence of such a hegemon [@Krasner1976; @Gilpin1981; @Kindleberger1986]. In conceptualizing openness as a public good, these theories made stark predictions about the distribution of military power and the international economy. In reality, the benefits of changes to trade policy are quite excludable. The model developed here reflects this reality by allowing governments to adopt discriminatory trade policies. Power can therefore be exercised to secure benefits not shared by other governments. The resulting international economic orders defy characterization as "open" or "closed." In a stylized version of the model developed here, I show that latent coercive threats can be used to open foreign markets. Militarily weak countries adopt lower barriers to trade than their powerful counterparts, all else equal [@Cooley2019a]. @Antras2011 consider a similar model in which governments influence elections abroad. Again, this influence has a liberalizing effect on the foreign government's trade policy.

Nevertheless, debate persists about the efficacy of military power in achieving economic benefits [@Mastanduno2009; @Drezner2013; @Bove2014; @Stokes2017]. These studies all confront the inference problem discussed here — does economic policy reflect governments' underlying preferences or the shadow of foreign military power? When redistribution is an alternative to war and bargaining is frictionless, war is not necessary to achieve coercive effects [@Brito1985; @Fearon1995; @Art1996]. I assume that the effectiveness of military coercion depends on governments' relative military advantage and the geographic distance between an attacking and defending country. Existing studies examine wars and militarized disputes to estimate the relationship between military spending and geographic distance on coercive capacity.^[On the relationship between military expenditure and military power, see @Kadera2004, @Beckley2010, @Beckley2018, @Carroll2019, and @Markowitz2020. On the loss of strength gradient, see @Boulding1962, @BuenodeMesquita1980, @Diehl1985, @Lemke1995, @Gartzke2011, and @Markowitz2013.] The number of disputes used to fit these models is relatively small and the nature of military technology changes over time. This dynamic technology and strategic selection into wars and disputes may confound estimates of these relationships. While I study a small sample of countries in a single year in this paper, I expand the universe of cases that can be used to estimate coercive capability by examining the responsiveness of policy to foreign threats.

Several studies have examined trade policy bargaining theoretically and empirically. @Grossman1995 extend the protection for sale model to a two-country bargaining setting. @Maggi1999 and @Bagwell1999 focus on the effect of the institutional context in which trade policy negotiations take place, relative to an un-institutionalized baseline. @Ossa2014 and @Bagwell2018a quantify these theories in structural models. Of course, the continued functioning of international institutions requires either a) that complying with the rules of the institution be incentive compatible for each member state, given others' strategies or b) that an external authority punish deviations from the institutions' rules sufficiently to induce compliance [@Powell1994]. Given the absence of such an external authority and the stark international distributional implications of alternative trade policy regimes, it is natural to consider how the ability to employ military force colors trade policy bargaining. 

Trade and trade policy are often theorized as tools governments can leverage to achieve political objectives [@Hirschman1945; @Gowa1993; @Martin2012; @Seitz2015].  Yet, affecting trade policy and concomitant prices, wages, and trade flows is also a central government objective in international relations. Moreover, the political objectives that ostensibly motivate governments in these "trade as means" models are loosely defined (e.g. "security") and themselves means to achieving other ends. Studying trade policy as a strategic end allows the analyst to leverage a family of empirical methods in international economics to construct counterfactual trade regimes and analyze their welfare implications [@Eaton2002; @Head2014; @Costinot2015; @Ossa2016]. Government objectives can be defined flexibly as a function of general equilibrium outputs (prices, wages, revenues).

A handful of other theoretical studies examine how power affects exchange in market environments [@Skaperdas2001; @Piccione2007; @Garfinkel2011; @Carroll2018]. Where property rights are assumed in classical models of the economy, these authors consider exchange and violence as coequal means to acquire goods from others. I instead direct attention to coercive bargaining over endogenous trade frictions (trade policy). These in turn affect the distribution of goods and welfare in the international economy.

# Data and Calibration of Economy




I estimate the model on a set of 9 governments in the year 2011.^[Focusing on a small set of governments is necessary for computational tractability. However, the largest countries (by GDP) are the most attractive targets for coercion, as changes to their trade policies return the largest welfare gains.] These governments are listed in Table \ref{tab:ccodes}. I aggregate all European Union governments into a single entity and collapse all countries not included in the analysis into a "Rest of World" (RoW) aggregate.^[Such an aggregation is necessary in order to calculate fully general equilibrium effects of counterfactual trade policies. However, I prohibit other countries from invading RoW and likewise prohibit RoW from invading others. This ensures that estimates of military parameters depend almost entirely on interactions between countries within my sample.] Non-RoW countries make up 72 percent of world GDP.

\begin{table}

\caption{\label{tab:ccodes}In-Sample Countries \label{tab:ccodes}}
\centering
\begin{tabular}[t]{ll}
\toprule
iso3 & Country Name\\
\midrule
\cellcolor{gray!6}{AUS} & \cellcolor{gray!6}{Australia}\\
CAN & Canada\\
\cellcolor{gray!6}{CHN} & \cellcolor{gray!6}{China}\\
EU & European Union\\
\cellcolor{gray!6}{JPN} & \cellcolor{gray!6}{Japan}\\
\addlinespace
KOR & South Korea\\
\cellcolor{gray!6}{RoW} & \cellcolor{gray!6}{Rest of World}\\
RUS & Russia\\
\cellcolor{gray!6}{USA} & \cellcolor{gray!6}{United States}\\
\bottomrule
\end{tabular}
\end{table}

Estimating the model and conducting the subsequent counterfactual exercises requires knowledge of governments' trade policies, disaggregated at the directed dyadic level. While detailed data on a particular policy instrument (tariffs) are available to researchers, these are but one barrier governments can use to influence the flow of trade. In a companion paper [@Cooley2019b], I show how to measure aggregate directed trade policy distortions given data on national accounts (gross consumption, gross production, and gross domestic product), price levels, trade flows, and freight costs. This method produces a matrix of trade barriers, in which the $i$, $j$th entry is the magnitude of policy barriers to trade an importing country $i$ imposes on goods from an exporting country $j$. In 2011, the estimated barriers were large, equivalent to an 81 percent import tariff on average.^[These results and the calibration choices that produce this value are discussed in more detail in Appendix B.] They also reveal substantial trade policy discrimination, with a subset of developed exporters facing far more favorable market access conditions than their less-developed peer countries. 

I take these estimated trade policies as the equilibrium output of the model developed here. I assume these policies are measured with error and construct an estimator that minimizes the magnitude of the resulting error vector. I sample from bootstrapped iterations of the trade policy estimation routine and re-compute parameter estimates many times in order to construct confidence intervals around my point estimates. 

Estimating the magnitude of these trade policies and tracing their impact on government welfare requires specifying a model of the international economy. This model, which follows closely that of @Eaton2002, can be represented succinctly as a mapping $h(\bm{\tau}, \bm{Z}_h; \bm{\theta}_h) = \bm{w}$ where $\bm{\tau}$ is a vector of trade policies, $\bm{Z}_h$ is a vector of economic data (including information on national accounts, price levels, trade flows, and freight costs), and $\bm{\theta}_h$ is a vector of parameters to be calibrated to match empirical analogues or taken from extant literature. $\bm{w}$ is a vector of wage levels, one for every country, from which equilibrium trade flows and price levels can be calculated. Government welfare is modeled below as a function of the outputs of this economy. I employ the same model of the international economy used to estimate trade policies in @Cooley2019b to calculate the welfare effects of trade policies in this study. The economy, the data required to calibrate it, and parameter calibration are discussed in more detail in Appendix B.

In the coercive political economy developed below, governments' relative directed war costs are modeled as a function of the military capability ratio between the attacker and defender, the geographic distance between the belligerents, and the gross domestic product of the attacking country. I store these observable features in the vector $\bm{Z}_m$. To measure military capability ratios, I employ [SIPRI](https://www.sipri.org/)'s data on military expenditure to measure governments' military capacity. These values are displayed in Figure \ref{fig:milex}. I use data from @Weidmann2010 to calculate centroid-centroid geographic distance between all countries in my sample. Data on gross domestic production comes from the [World Input-Output Database (WIOD)](http://www.wiod.org/home) [@Timmer2015]. 

![Military expenditure for in-sample governments. Values for ROW and EU are obtained by summing expenditure of all member countries. \label{fig:milex}](figure/milex-1.pdf)

## Reduced Form Evidence on Coercion and Trade Policy

To assist in the interpretation of the data, consider a simple bilateral coercive bargaining setting. Governments 1 and 2 bargain over a pie of size 1. Let $x \in [0, 1]$ denote the share of the pie awarded to government 1 (with the remainder, $1-x$, going to government 2). In the trade setting studied here, $x=1$ might correspond to government 1 implementing optimal tariffs and government 2 liberalizing fully. Each government's valuation of the pie is given by an increasing, weakly concave utility function $u_i(x)$. The value of each government's outside option is given by a war function, $w_i(M_i / M_j)$, which depends on their relative military capabilities, $\frac{M_i}{M_j}$. Assume $w_i$ is increasing in this military capability ratio -- that is, more powerful governments enjoy better outside options.

For simplicity, assume the pie is divided via the Nash Bargaining Solution, satisfying
\begin{equation}
\begin{split}
x^\star \in \argmax_x & \quad \left( u_1(x) - w_1(M_1 / M_2) \right) \left( u_2(x) - w_2(M_2 / M_1) \right) \\
\text{subject to} & \quad u_1(x) \geq w_1(M_1 / M_2) \\
& \quad u_2(x) \geq w_2(M_2 / M_1) .
\end{split}
\end{equation}
Taking first order conditions, it is straightforward to show that the allocation to government 1, $x^\star$, satisfies
$$
u_1(x^\star; M_1, M_2) = \frac{u_1^\prime(x^\star)}{u_2^\prime(1 - x^\star)} \left( u_2(1 - x^\star) - w_2(M_2 / M_1) \right) + w_1(M_1 / M_2) . 
$$
Differentiating this equation with respect to government 1's military capacity, $M_1$, we see that $u_1(x^\star; M_1, M_2)$ is increasing in $M_1$,
$$
\frac{\partial u_1(x^\star; M_1, M_2) }{\partial M_1} = \underbrace{- \frac{u_1^\prime(x^\star)}{u_2^\prime(1 - x^\star)} \frac{\partial w_2(M_2 / M_1)}{\partial M_1}}_{>0} + \underbrace{\frac{\partial w_1(M_1 / M_2)}{ \partial M_1}}_{>0} > 0 .
$$
In other words, the distance between government 1's equilibrium utility and the utility it receives at its ideal point is decreasing in its relative military advantage.

Suppose that governments endeavor to maximize the welfare of the representative consumer.^[I will relax this assumption in the structural model developed below.] With the economy, $h$, calibrated, I can calculate the change in utility each representative consumer would experience when each other government adopts free trade, relative to their utility at the baseline set of policies. Taking this as an empirical measure of the ratio $u_1(x^\star; M_1, M_2) / u_1(1)$, the model implies this quantity will be increasing in $M_1$, country 1's military capacity. I will refer to this quantity as government 1's inverse *conquest value* vis-à-vis government 2.

![Correlation between military capability ratios and inverse conquest values, all pairs of in-sample countries. \label{fig:rcvm}](figure/rcvm-1.pdf)

Figure \ref{fig:rcvm} plots the empirical relationship between military capability ratios and inverse conquest values. Each potential "attacking" country's military capability ratio vis-à-vis every "defending" country is plotted on the x-axis. On the y-axis is the attacking inverse country's value for conquering each defending country. Consistent with the predictions of this simple model, government's inverse conquest values correlate positively with their relative military power. Table \ref{fig:rcvm_reg_tex} and Figure \ref{fig:rcvm_reg_dw} display the results of a series of linear models that estimate the conditional correlations between the inverse conquest value and the military capability ratio, distance between the countries, and country-specific constants. 

\begin{table}

\caption{\label{tab:rcvm}Inverse Conquest Values and Military Capability Ratios \label{fig:rcvm_reg_tex}}
\centering
\resizebox{\linewidth}{!}{
\begin{tabular}[t]{lllll}
\toprule
  & Base & Base (Attacker FE) & Loss of Strength & Loss of Strength (Attacker FE)\\
\midrule
Log Mil Capability Ratio & 0.016*** & 0.033*** & 0.026 & 0.045\\
 & (0.004) & (0.004) & (0.052) & (0.039)\\
Log Distance &  &  & 0.003 & 0.002\\
 &  &  & (0.010) & (0.008)\\
(Log Mil Capability Ratio) X (Log Distance) &  &  & -0.001 & -0.001\\
 &  &  & (0.006) & (0.004)\\
\midrule
Num.Obs. & 56 & 56 & 56 & 56\\
R2 & 0.247 & 0.676 & 0.249 & 0.677\\
R2 Adj. & 0.233 & 0.621 & 0.205 & 0.605\\
F & 17.720 & 12.251 & 5.739 & 9.421\\
Attacker FE? &  & ✓ &  & ✓\\
\bottomrule
\multicolumn{5}{l}{\textsuperscript{} * p < 0.1, ** p < 0.05, *** p < 0.01}\\
\end{tabular}}
\end{table}

![Conditional correlations between inverse conquest values and military capability ratios, geographic distance, and country-specific constants. \label{fig:rcvm_reg_dw}](figure/rcv_reg_dw-1.pdf)

The first model confirms the statistical significance of the correlation shown in Figure \ref{fig:rcvm}. The second model estimates this correlation within potential "attacking" countries. Here, inverse conquest values continue to rise as military advantage rises. The final two models interact the military capability ratio with a measure of distance between the attacker and defender. The estimated correlation between military capability is not attenuated, but does lose statistical significance in these specifications. Distance and the interaction of distance with military capability does not covary with the inverse conquest values, whether or not country-specific factors are included. These raw correlations are informative about the role of coercion in the formation of trade policy but suggestive at best. Trade policy bargaining is a multilateral endeavor in which third party externalities loom large. Moreover, governments may vary in their preferences for protectionism, changing their ideal policies and their valuations for conquering others. The model developed below accounts explicitly for these features of trade policy bargaining, delivering interpretable estimates of the effects of military capability and geographic distance on trade policy outcomes.







# Model




There are $N$ governments, indexed $i \in \left\{ 1, ..., N \right\}$. Governments choose trade policies $\bm{\tau}_i = \left( \tau_{i1}, ..., \tau_{iN} \right) \in [1, \bar{\tau}]^N$ which affect their welfare indirectly through changes in the international economy.^[$\bar{\tau}$ is an arbitrarily large but finite value sufficient to shut down trade between any pair of countries.] An entry of the trade policy vector, $\tau_{ij}$ is the tax country $i$ imposes on imports from $j$.^[Costs enter in an "iceberg" fashion, and I normalize $\tau_{ii} = 1$. Then, if the price of a good in country $j$ is $p_{jj}$, its cost (less freight) in country $i$ is $\tau_{ij} p_{jj}$. The ad valorem tariff equivalent of the trade policy is $t_{ij} = \tau_{ij} - 1$. I employ structural estimates of these costs from @Cooley2019b to estimate the model, which are described in more detail in Appendix A.] The economy, detailed in Appendix A, can be succinctly characterized by a function $h: \bm{\tau} \rightarrow \mathbb{R}_{++}^N$ mapping trade policies to wages in each country, denoted $\bm{w} = \left( w_1, ..., w_N \right)$. These in turn determine trade flows between pairs of countries and price levels around the world.^[The economy is a variant of the workhorse model of @Eaton2002.]

Throughout, I will use $\bm{\theta}_m$ to denote the vector of all non-economic parameters to be estimated and $\bm{Z}_m$ to denote the vector of all non-economic data observed by the researcher. $\bm{\theta}_h$ denotes parameters associated with the economy, $h$, which will be calibrated. $\bm{Z}_h$ denotes data associated with the economy. I will explicate the elements of these vectors in the proceeding sections and the Appendix.

Government welfare depends on the economic consequences of trade policy choices. Governments value the welfare of a representative consumer that resides within each country. The consumer's welfare in turn depends on net revenues accrued through the government's trade policy distortions, which are redistributed to the consumer. Revenues and induced welfare can be computed given knowledge of the general equilibrium function $h(\bm{\tau})$. Each government's welfare is given by $V_i \left( h(\bm{\tau}); v_i \right)$ where $v_i$ is a revenue threshold parameter. This value of this function depends on the consumer's net income and is characterized fully in the Appendix. The consumer's net income can be written as a function of the governments' policy choices
$$
\tilde{Y}_i(h_i(\bm{\tau}))  = h_i(\bm{\tau}) L_i + r_i(h(\bm{\tau}); v_i) . 
$$
$L_i$ is the country's labor endowment, $r_i(h(\bm{\tau}); v_i)$ is trade policy revenues, and $h_i(\bm{\tau})$ are equilibrium wages in $i$. $v_i \in [1, \infty)$ is a structural parameter that modulates the government's ability to extract trade policy rents. 

Adjusted revenues are given by
\begin{equation} \label{eq:r}
r_i(h(\bm{\tau}), v_i) = \sum_j (\tau_{ij} - v_i) X_{ij}(h(\bm{\tau}))
\end{equation}
and $X_{ij}(h(\bm{\tau}))$ are country $i$'s imports from country $j$.^[This object does not correspond empirically to governments' factual tariff revenues, as $\tau_{ij}$ incorporates a larger set of trade policy distortions than tariffs alone. Yet, non-tariff barriers to trade also generate rents that do not accrue directly to the government's accounts (see, for example, @Anderson1992 for the case of quotas). This revenue function is designed to capture this broader set of rents.] When $v_i$ is close to one, small policy distortions are sufficient to generate revenue for the government. Conversely when $v_i$ is high, the government must erect large barriers to trade before revenues begin entering government coffers and returning to the pockets of the consumer. Because consumers' consumption possibilities depend on revenue generation, increasing $v_i$ induces governments' to become more protectionist. This formulation provides substantial flexibility in rationalizing various levels of protectionism, while avoiding assuming specific political economic motivations for its genesis. From the perspective of the consumers, rents extracted from imports are valued equally, regardless of their source. Ex ante, governments are not discriminatory in their trade policy preferences. Optimal policies for government $i$ maximize $V_i \left( h(\bm{\tau}_i; \bm{\tau}_{-i}); v_i \right)$.

These optimal policies impose externalities on other governments. By controlling the degree of market access afforded to foreign producers, trade policies affect the wages of foreign workers and the welfare of the governments that represent them. They also partially determine trade flows, which affect other governments' ability to collect rents. In this sense, protectionism is "beggar thy neighbor." Governments' joint policy proposals are denoted $\tilde{\bm{\tau}}$.

Wars are fought in order to impose free trade abroad. After observing policy proposals, governments decide whether or not to launch wars against one another. Wars are offensive and *directed*. If government $j$ decides to launch a war against $i$ it pays a dyad-specific cost, $c_{ji}$, and imposes free trade on the target. These war costs are modeled as realizations of a random variable from a known family of distributions and are held as private information to the prospective attacker. The shape of these distributions is affected by the governments' relative power resources, denoted $\frac{M_j}{M_i}$, as well as the geographic distance between them, $W_{ji}$. These inverse value of these costs are distributed with c.d.f. $F_{ji}$ which is described in more detail below. I normalize the cost of defending against aggression to zero.

If $i$ is not attacked by any other government its announced policies are implemented. Otherwise, free trade is imposed, setting $\bm{\tau}_i = \left( 1, \dots, 1 \right) = \bm{1}_i$. Substituting these policies into $j$'s utility function gives $V_j(\bm{1}_i; \tilde{\bm{\tau}}_{-i})$ as $j$'s *conquest value* vis-à-vis $i$. Note that I prohibit governments from imposing discriminatory policies on conquered states. Substantively, this assumption reflects the difficulty in enforcing sub-optimal policies on prospective client states, relative to reorienting their political institutions to favor free trade. This also ensures that the benefits of conquest are public. However, it does not guarantee non-discrimination in times of peace. Governments that pose the most credible threat of conquest can extract larger policy concessions from their targets in the form of directed trade liberalization. 

Government $j$ therefore prefers not to attack $i$ so long as
\begin{align*}
V_j \left( \bm{1}_i; \tilde{\bm{\tau}}_{-i} \right) - c_{ji} &\leq V_j \left( \tilde{\bm{\tau}} \right) \\
c_{ji}^{-1} &\leq \left( V_j \left( \bm{1}_i; \tilde{\bm{\tau}}_{-i} \right) - V_j \left( \tilde{\bm{\tau}} \right) \right)^{-1}
\end{align*}

or if the benefits from imposing free trade on $i$ are outweighed by the costs, holding other governments' policies fixed. The probability that no government finds it profitable to attack $i$ can then be calculated as
$$
H_i \left( \tilde{\bm{\tau}}; \bm{Z}_m, \bm{\theta}_m \right) = \prod_{j \neq i} F_{ji} \left( \left( V_j \left( \bm{1}_i; \tilde{\bm{\tau}}_{-i} \right) - V_j \left( \tilde{\bm{\tau}} \right) \right)^{-1} \right)
$$
I am agnostic as to the process by which the coordination problem is resolved in the case in which multiple prospective attackers find it profitable to attack $i$. I assume simply that $i$ is attacked with certainty when it is profitable for any government to do so. This event occurs with probability $1 - H_i(\tilde{\bm{\tau}}; \bm{Z}_m, \bm{\theta}_m)$. 

Because of strategic interdependencies between trade policies, optimal policy proposals are difficult to formulate. Governments face a complex problem of forming beliefs over the probabilities that they and each of their counterparts will face attack and the joint policies that will result in each contingency. For simplicity, I assume governments solve the simpler problem of maximizing their own utility, assuming no other government faces attack. I denote this objective function with $G_i(\tilde{\bm{\tau}})$ which can be written
\begin{equation} \label{eq:G}
G_i(\tilde{\bm{\tau}}) = H_i(\tilde{\bm{\tau}}; \bm{Z}_m, \bm{\theta}_m) V_i(\tilde{\bm{\tau}}) + \left( 1 - H_i(\tilde{\bm{\tau}}; \bm{Z}_m, \bm{\theta}_m) \right) V_i(\bm{1}_i; \tilde{\bm{\tau}}_{-i})
\end{equation}
where $V_i(\bm{1}_i; \tilde{\bm{\tau}}_{-i})$ denotes $i$'s utility when free trade is imposed upon it. This objective function makes clear the tradeoff $i$ faces when making policy proposals. Policies closer to $i$'s ideal point deliver higher utility conditional on peace, but raise the risk of war. Lowering barriers to trade on threatening countries increases $H_i(\tilde{\bm{\tau}}; \bm{Z}, \bm{\theta}_m)$, the probability $i$ avoids war, at the cost of larger deviations from policy optimality. 

Policy proposals are made simultaneously. Let $\tilde{\bm{\tau}}_i^\star(\tilde{\bm{\tau}}_{-i})$ denote a solution to this problem and $\tilde{\bm{\tau}}^\star$ a Nash equilibrium of this policy announcement game. 

## Policy Equilibrium in Changes

The equilibrium of the international economy depends on a vector of structural parameters and constants $\bm{\theta}_h$ defined in Appendix A. Computing the economic equilibrium $h(\bm{\tau}; \bm{\theta}_h)$ requires knowing these values. Researchers have the advantage of observing data related to the equilibrium mapping for one particular $\bm{\tau}$, the factual trade policies. 

The estimation problem can be therefore partially ameliorated by computing the equilibrium in *changes*, relative to a factual baseline. Consider a counterfactual trade policy $\tau_{ij}^\prime$ and its factual analogue $\tau_{ij}$. The counterfactual policy can be written in terms of a proportionate change from the factual policy with $\tau_{ij}^\prime = \hat{\tau}_{ij} \tau_{ij}$ where $\hat{\tau}_{ij} = 1$ when $\tau_{ij}^\prime = \tau_{ij}$. By rearranging the equilibrium conditions, I can solve the economy in changes, replacing $h(\bm{\tau}; \bm{\theta}_h) = \bm{w}$ with $\hat{h}(\hat{\bm{\tau}}; \bm{\theta}_h) = \hat{\bm{w}}$. Counterfactual wages can then be computed as $\bm{w}^\prime = \bm{w} \odot \hat{\bm{w}}$.

This method is detailed in Appendix A. Because structural parameters and unobserved constants do not change across equilibria, parameters that enter multiplicatively drop out of the equations that define this "hat" equilibrium. This allows me to avoid estimating these parameters, while enforcing that the estimated equilibrium is consistent with their values. The methodology, introduced by @Dekle2007, is explicated further in @Costinot2015 and used to study trade policy in @Ossa2014 and @Ossa2016.

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
\begin{equation} \label{eq:inv_costs}
\text{Pr}\left( \frac{1}{\hat{c}_{ji}} \leq \frac{1}{\hat{c}} \right) = \hat{F}_{ji} \left( \frac{1}{\hat{c}} \right) = \exp \left( -\frac{1}{\hat{C}} \left( \frac{M_j}{M_i} \right)^{\gamma} W_{ji}^{-\alpha_1} Y_j^{\alpha_2} \hat{c}^{\eta} \right) .
\end{equation}
The parameters $\alpha_1$ and $\gamma$ govern the extent to which military advantage and geographic proximity are converted into cost advantages. If $\gamma$ is greater than zero, then military advantage reduces the costs of war. Similarly, if $\alpha_1$ is greater than zero, then war costs increase with geographic distance, consistent with a loss of strength gradient. Because costs are now measured relative to baseline utility, I include a measure of the attacking country's g.d.p., $Y_j$ in the cost distribution. If $\alpha_2$ is positive, larger countries sacrifice a smaller percentage of their welfare when prosecuting wars. $\hat{C}$ and $\eta$ are global shape parameters that shift the cost distribution for all potential attackers and are calibrated.^[I set $\hat{C}=$ 25 and $\eta=$ 1.5. By shifting all potential attackers' war costs, $\hat{C}$ modulates the probability of war in the data and could be estimated on data describing the likelihood of war between in-sample countries in any given year. Because no wars occur in the period I study, I do not undertake this exercise.]

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
\hat{H}_i \left( \hat{\tilde{\bm{\tau}}}; \bm{Z}, \bm{\theta}_m \right) = \exp \left( - \sum_{j \neq i} - \frac{1}{\hat{C}} \left( \frac{M_j}{M_i} \right)^{\gamma} W_{ji}^{-\alpha_1} Y_j^{\alpha_2} \left( \hat{V}_j \left( \bm{1}_i; \tilde{\bm{\tau}}_{-i} \right) - \hat{V}_j \left( \hat{\tilde{\bm{\tau}}} \right) \right)^{-\eta} \right) .
$$

Let $\hat{\tilde{\bm{\tau}}}_i^\star(\hat{\tilde{\bm{\tau}}}_{-i})$ denote a solution to policy change proposal problem and $\hat{\tilde{\bm{\tau}}}^\star(\bm{\theta}_m; \bm{Z}_m)$ a Nash equilibrium of this policy change announcement game. 


# Estimation


The model's equilibrium, $\hat{\tilde{\bm{\tau}}}^\star$ depends on a vector of unobserved parameters $\bm{\theta}_m = \left( \bm{v}, \alpha_1, \alpha_2, \gamma \right)$. I assume observed policies are generated by the model up to measurement error
$$
\tilde{\bm{\tau}} = \tilde{\bm{\tau}}^\star(\bm{\theta}_m, \bm{Z}_m) + \bm{\epsilon} . 
$$
$\bm{\epsilon}$ is an $N \times N$ matrix with $\epsilon_{ii} = 0$ for all $i$ and $\E[\epsilon_{ij}] = 0$ for all $i \neq j$. Recall that $\tilde{\bm{\tau}}^\star$ can be reconstructed from $\hat{\tilde{\bm{\tau}}}^\star$, the model's equilibrium, by simply multiplying equilibrium policies by factual policies, $\bm{\tau}$.

Following the assumption that measurement error is mean-zero, I seek an estimator that solves
\begin{equation} \label{eq:estimator}
\min_{\bm{\theta}_m} \sum_i \sum_j \left( \epsilon_{ij}(\bm{\theta}_m, \bm{Z}_m) \right)^2 .
\end{equation}

Solving this problem presents two computational challenges. First, computing government welfare changes for any given $\hat{\bm{\tau}}$ requires solving the system of equations characterizing the equilibrium of the international economy, $\hat{h}(\hat{\bm{\tau}})$. These changes must be computed for both the proposed policies and for policies imposed by each potential war. Second, computing $\tilde{\bm{\tau}}^\star(\bm{\theta}_m)$ requires iteratively solving each government's best response problem until convergence at a Nash equilibrium. I sidestep both of these by recasting the best response problem and estimation problem as mathematical programs with equilibrium constraints (MPECs) [@Su2012; @Ossa2014; @Ossa2016].

To reformulate the best response problem, I consider an equivalent formulation in which each government chooses trade policies and wages, subject to the additional constraint that chosen wages are consistent with the general equilibrium of the international economy ($\hat{h}(\hat{\tilde{\bm{\tau}}}) = \hat{w}$). Let $\hat{\bm{x}}_i = \left( \hat{\tilde{\bm{\tau}}}_i, \hat{\bm{w}} \right)$ store $i$'s choice variables in this problem. Then, this problem can be rewritten as follows, noting explicitly dependencies on $\bm{\theta}_m$
\begin{equation} \label{eq:tauTildeHatMPEC}
\begin{split}
\max_{\hat{\bm{x}}_i} & \quad \hat{G}_i(\hat{\bm{w}}; \bm{\theta}_m) \\
\text{subject to} & \quad \hat{\bm{w}} = \hat{h}(\hat{\tilde{\bm{\tau}}}) .
\end{split}
\end{equation}
Let $\mathcal{L}_i(\hat{\bm{x}}_i, \bm{\lambda}_i)$ denote the associated Lagrangian. This formulation allows me to quickly compute best responses $\hat{\tilde{\bm{\tau}}}_i(\hat{\tilde{\bm{\tau}}}_{-i})$ without iteratively solving $h(\hat{\tilde{\bm{\tau}}})$.

I then reformulate the estimation problem (\ref{eq:estimator}) in a similar manner. At an interior Nash equilibrium, the gradient of the Lagrangian is null
$$
\nabla_{\hat{\tilde{\bm{\tau}}}_i} \mathcal{L}_i(\hat{\bm{x}}_i, \bm{\lambda}_i; \bm{\theta}_m) = \bm{0}
$$
for each government $i$. In the reformulated estimation problem, I seek to choose parameters, trade policies, multipliers, and general equilibrium response variables for the proposed policies and imposed policies in order to minimize measurement error while enforcing these equilibrium constraints, in addition to general equilibrium constraints. Let $\hat{\bm{x}}_i^\prime = \left( \bm{1}_i, \hat{\tilde{\bm{\tau}}}_{-i}, \hat{\bm{w}}_i^\prime \right)$ store general equilibrium equilibrium policies and wages when free trade is imposed on $i$.

Formally, I solve
\begin{equation} \label{eq:estimatorMPEC}
\begin{split}
\min_{ \bm{\theta}_m, \hat{\tilde{\bm{\tau}}}, \hat{\bm{w}}, \hat{\bm{w}}^\prime, \bm{\lambda} } & \quad \sum_i \sum_j \left( \epsilon_{ij} \right)^2 \\
\text{subject to} & \quad \nabla_{\hat{\tilde{\bm{\tau}}}_i} \mathcal{L}_i(\hat{\bm{x}}_i, \bm{\lambda}_i; \bm{\theta}_m) = \bm{0} \text{ for all } i \\
& \quad \hat{\bm{w}} = \hat{h} \left( \hat{\tilde{\bm{\tau}}} \right) \\
& \quad \hat{\bm{w}}_i^\prime = \hat{h} \left( \bm{1}_i, \hat{\tilde{\bm{\tau}}}_{-i} \right) \text{ for all } i
\end{split}
\end{equation}
The constraints collectively ensure $\hat{\tilde{\bm{\tau}}} = \tilde{\bm{\tau}}^\star(\bm{\theta}_m)$ -- or that the policies are consistent with Nash equilibrium in policies, given estimated parameters.

This procedure produces point estimates $\tilde{\bm{\theta}}_m$. I then construct uncertainty intervals through nonparametric bootstrap, taking 250 bootstrapped samples from the distribution of estimated policy barriers in @Cooley2019b and re-solving (\ref{eq:estimator}).

# Results




![Model parameter estimates and 95 percent confidence intervals. The top panel shows protectionism preference parameter estimates ($v_i$) for each country. The bottom panel shows parameter estimates for observables affecting costs of war ($\gamma, \alpha_1, \alpha_2$). \label{fig:ests}](figure/ests-1.pdf)

Figure \ref{fig:ests} displays results from the estimation. Recall that $v_i$ governs the ease with which governments can extract revenues from trade policy distortions. When $v_i$ is higher government $i$ prefers higher barriers to trade, all else equal. When $v_i=1$ the government acts as a classical social welfare maximizer. There is considerable heterogeneity in governments' estimated preferences for protectionism. The United States and Russia are estimated to be relatively liberal, while Australia and Canada are quite protectionist. 

An attacking country's military advantage and g.d.p. are estimated to reduce war costs, facilitating coercion. There are increasing returns to both of these features in reducing the average costs of war ($\gamma, \alpha_2 > 1$). Economically large and military powerful countries are the most effective at coercion, holding the distance of their adversary constant. Figure \ref{fig:war_costs} displays estimated average war costs, relative to those of the United States, holding the distance to the adversary constant. Given its large economy and military, the United States is estimated to enjoy the smallest average war costs. The European Union, China, and Russia pay between 3 and 6 times the costs of the United States to prosecute wars on average. Wars are estimated to cost at least an order of magnitude more than U.S. wars for other countries in the sample.

![Estimated relative war costs against a fixed adversary. The United States' costs serve as baseline ($c=1$). \label{fig:war_costs}](figure/war_costs-1.pdf)

War costs are estimated to depend on the distance between the attacker and potential adversary. Attackers that are more distant from their adversaries are estimated to enjoy smaller war costs. In other words, model estimates imply an inverse loss of strength gradient. This may emerge due to the peculiarities of military technology in 2011, a period in which geographic distance represents a uniquely small impediment to the projection of force.

The model estimates can be readily transformed to deliver empirical quantities that measure the salience of military coercion in international relations. Figure \ref{fig:rcv} plots the estimated conquest value for each potential attacking country vis-à-vis each potential defending country. These quantities differ from those analyzed in the reduced form section above in that they account explicitly for the attacking government's preferences for protectionism. Russia's conquest values are estimated to be among the highest in the sample. This reflects the relatively poor market access conditions it enjoys at the estimated equilibrium. Because their economies are the largest in the sample, the gains that accrue from successfully conquering the United States, China and the European Union tend to be larger than the gains from conquering other countries. Australia, Canada, and China benefit little from conquering others. This result obtains because of their governments' estimated preferences for protectionism. Conquest stimulates trade that is disadvantageous for a government $i$ when $v_i$ is high and $i$'s trade barriers are lowered below the revenue threshold due to the effects of coercion. This variation in conquest values highlights the dependence of the coercive environment on the underlying international economy and government preferences.

![Estimated conquest value for each potential attacking country vis-à-vis each potential defending country. Darker colors indicate higher conquest values. \label{fig:rcv}](figure/rcv-1.pdf)

It is also straightforward to calculate the equilibrium probability of war once the model has been estimated by simply plugging parameter estimates back into the inverse cost distribution given in Equation \ref{eq:inv_costs}.^[These estimated probabilities of war should be interpreted only in relative terms. The overall probability of war is governed by the calibrated parameter $\hat{C}$. Higher values of this parameter would scale down each probability of war but would not shift their relative values.] Figure \ref{fig:pr_peace} plots point estimates and uncertainty intervals surrounding the likelihood of war between all pairs of countries in the sample. In general, governments run very small risks of invasion from other governments. However, the threat of war with the United States looms large in the sample. The probabilities the United States attacks each other country in the sample are highlighted in orange in the Figure. The European Union is also estimated to impose substantial threats.

![Estimated equilibrium probabilities of war, point estimates and 95 percent confidence intervals. Probabilities the United States attacks each other country highlighted in orange. \label{fig:pr_peace}](figure/war_probs-1.pdf)

It is worth noting that the countries with the highest estimated risk of war with the United States, Japan and Australia, happen to be U.S. allies. The security guarantees encapsulated in these alliances are not explicitly modeled. One way to interpret these results is that Australian and Japanese security would deteriorate rapidly in the absence of U.S. military protection, representing an implicit threat the United States can leverage to affect trade policy.^[@Lake2007 would label these relationships "hierarchical" and based on the authority of the United States to dictate the policy of its subordinates. Still, in Lake's conceptualization, "authority is buttressed by the capacity for coercion" (p. 53).]

## Model Fit and Inferences about Policy Preferences

![Correlation between trade barrier data and model predictions. \label{fig:fit}](figure/fit-1.pdf)

Figure \ref{fig:fit} evaluates the ability of the estimated model to predict the level of trade barriers. The model's mean absolute error is 0.27, equivalent to a 27 percent ad valorem tariff. The model's predictions are fairly well correlated with the trade barrier data ($\rho=$ 0.68). In Appendix C I plot the model's predictive error for each directed dyad in the sample, highlighting which observations are well explained by the model and which are not. Of note, Russia faces uniquely poor market access conditions in the data that the model does not fully replicate.

![Effect of modeling military coercion on inferences about governments' preferences for protectionism. Figure plots point estimates and 95 percent confidence intervals for preference parameters under baseline model and model in which coercion is impossible. \label{fig:ests_mil_off}](figure/ests_mil_off-1.pdf)

Modeling coercion explicitly both improves model fit and alters inferences about government's underlying preferences for protectionism. I re-estimate the model under the assumption that coercion is impossible. In this model, equilibrium policies reflect only governments' underlying preferences, $v_i$. Estimated preferences for protectionism under this model are shown in Figure \ref{fig:ests_mil_off}. The estimated preferences of militarily powerful countries are largely unchanged across models. This is not true for less powerful countries. The estimated preferences of Australia, Canada, and China move dramatically when coercion is prohibited. The model developed here rationalizes their trade barriers as the result of highly protectionist latent preferences tempered by the effects of international coercion. The coercion-free model instead infers instead that they are relatively liberal in their preferences. Leaving coercion out of the model exerts downward bias on estimates of governments' welfare-mindedness. A large literature employs the equilibrium trade policies of @Grossman1994 or @Grossman1995 to estimate the weight governments place on the welfare of special interests relative to that of society at large [@Goldberg1999; @Mitra2006; @Gawande2009; @Gawande2012; @Ossa2014; @Gawande2015]. Because the "protection for sale" model incorporates no theory of international coercion, these studies over-estimate governments' social welfare consciousness. 

Modeling coercion explicitly also improves model fit substantially. The correlation coefficient between model predictions and observed trade barriers falls to 0.45 when coercion is prohibited. The mean absolute error increases 19.9 percent to 0.32. In Appendix C I replicate Figure \ref{fig:fit} for the coercion-free model. 


# Counterfactuals: Coercion and the World Economy




How does the shadow of coercion affect the functioning of the world economy? How would patterns of trade and trade protectionism change if governments' power resources or preferences were modified? With model estimates computed, this class of questions can be addressed through recomputing the model's equilibrium at alternative sets of parameters or data. In other words, compute $\tilde{\bm{\tau}}^\star(\bm{\theta}_m^\prime; \bm{Z}_m^\prime)$ where $\bm{\theta}_m^\prime$ and $\bm{Z}_m^\prime$ are alternative arrangements of parameters and observable model primitives, respectively. Changes to the economy can then be computed by substituting these counterfactual equilibrium policies into the model of the world economy, solving $h \left( \tilde{\bm{\tau}}^\star(\bm{\theta}_m^\prime; \bm{Z}_m^\prime) \right)$. I consider three counterfactual scenarios here. First, I quantify the aggregate effects of military coercion by conducting a counterfactual in which military coercion is prohibited. Second, I quantify the effects of the diffusion of military power on trade policy and the international economy by recomputing the model's equilibrium at projected levels of military spending in 2030. Finally, I quantify the effects of liberalizing Chinese trade policy preferences on the probability of various wars.

## A Coercion-Free World

First, I calculate the net economic effects of coercion by calculating the equilibrium to a game in which coercion is impossible, holding governments' preferences at their estimated values. The shadow of coercion is a substantial force for trade liberalization. Moving from this counterfactual "pacifist" world to the coercive equilibrium delivers a 63 percent increase in the value of total global trade. Figure \ref{fig:cfct1_X} disaggregates these changes in trade flows, showing the change in imports induced by demilitarization for each importer-exporter pair. It also shows the changes in equilibrium trade policy that generate these changes in trade flows. 

![Changes in trade flows and trade policy when military coercion is counterfactually prohibited. Top plot shows changes in the (log) value of imports for each country in the sample, disaggregated by trade partner. Bottom plot shows changes in equilibrium trade policies for each country in the sample, again disaggregated by trade partner. Counterfactual import values and trade policies are shown in orange. \label{fig:cfct1_X}](figure/cfct1_X-1.pdf)

U.S. and Russian trade policies remain largely unchanged. Yet their trade patterns are still affected by others' changes in trade policy behavior. Australia, Canada, China, and South Korea become substantially more protectionist, reducing their own trade volumes and shifting patterns of international exchange elsewhere. Trade policies in the coercion-free world are largely homogenous within adopting countries, reflecting the model's ex-ante incentives against policy discrimination. The exception to this rule is for large countries like the United States and European Union, whose counterfactual trade policies reflect dependence on the size of their trading partners, consistent with optimal taxation [@Johnson1953; @Ossa2014].

![Changes in government welfare and consumer welfare (calculated by setting $v_i=1$ for all $i$) induced by moving from coercion-free equilibrium to baseline equilibrium. \label{fig:cfct1_welfare}](figure/cfct1_welfare-1.pdf)

Figure \ref{fig:cfct1_welfare} plots the changes in government and consumer welfare due to coercion, calculated as the difference between the coercion-free equilibrium and the baseline equilibrium. The measure of consumer welfare is calculated by setting $v_i=1$ for all governments and evaluating the representative consumer's indirect utility at equilibrium policies, consistent with the interpretation of $v_i$ as a political economy parameter capturing government incentives to deviate from socially optimal trade policies. Consumers benefit substantially from the trade liberalization induced by military coercion, but highly protectionist governments suffer. Australia, Canada, China, and South Korea suffer welfare losses when military coercion is permitted, relative to the counterfactual "pacifist" world. The United States government gains the most from coercion among non-RoW countries. 

## Multipolarity, Trade Policy, and International Trade

Military power in 2011 was highly concentrated in the hands of the United States (see Figure \ref{fig:milex}). Since 2011, other countries, China in particular, have begun to close this military capability gap with the United States. How would the continued diffusion of military power affect trade policy and patterns of international economic exchange? To answer this question I project each in-sample government's military spending in 2030, assuming military budgets grow (shrink) at their average rate between 2011 and 2018. Projected military spending for 2030 is shown in Figure \ref{fig:milex_2030}. The largest change is the shift in relative military power from the United States and European Union toward China. 

![Projected military spending in 2030, assuming military budgets grow at observed average growth rate between 2011 and 2018. \label{fig:milex_2030}](figure/milex_2030-1.pdf)

Multipolarization impacts globalization in two ways. On the one hand, newly militarily powerful countries can resist others' demands to liberalize, leading to a less-integrated global economy. On the other hand, the diffusion of military power increases the coercive capacity of some states in the system, allowing them to make greater liberalization demands of others and contributing to global economic integration. These effects are mediated by governments' preferences for protectionism, which determine governments' ideal policies and the returns to coercion. In this "multipolarization" scenario, China leverages these increases in military might to adopt more restrictive trade policies. Figure \ref{fig:cfct2_tau} displays the changes in Chinese trade policies that result under multipolarization. On net, multipolarization is a force for liberalization. The value of global trade under multipolarization is 110.3 percent its baseline value.

![Changes in Chinese trade policies under multipolarization. \label{fig:cfct2_tau}](figure/cfct2_tau-1.pdf)

## Chinese Preference Liberalization and the Risk of War

Reducing governments' incentives for protectionism can also decrease the risk of war. By reducing governments incentives to adopt high trade barriers, preference liberalization reduces others' incentives for conquest, in turn, reducing the probability of war. To quantify these effects, I consider a liberalization of Chinese policy preferences, setting their revenue collection parameter to that of the United States ($\hat{v}_{\text{CHN}}=$ 2.77, $v_{\text{CHN}}^\prime=$ 1.52). Figure \ref{fig:war_probs_pp4} shows the change in the probability of war against China that occurs as the result of this change in preferences. The United States still poses a threat of war, but the probability the United States launches a war against China is reduced substantially from 33.3 percent to 5.9 percent. The probability China faces attack from another source is virtually eliminated.

![Changes in probability of war against China after Chinese preference liberalization. \label{fig:war_probs_pp4}](figure/war_probs_pp4-1.pdf)


# Conclusion


The shadow of power plays a central role in international relations theory, but measuring its effects has proved challenging. It is axiomatic that if governments forgo war, then they must at least weakly prefer the policy status quo to the expected policy outcomes that would result from potential wars. In this paper, I have shown that a flexible model of government preferences over trade outcomes can serve to quantify government welfare under this policy counterfactual. I then leverage the difference between factual government welfare and its conquest values to identify parameters governing the technology of coercion in international relations. 

The preliminary estimates of these parameters suggest that military power indeed constrains governments' policy choice in international relations. Military spending advantage translates into battlefield advantage. These military constraints serve to contort trade policy toward the interests of the powerful as well as the resolved — those whose benefits from conquest are the largest. Military threats structure the workings of the international economy.

Drawing these conclusions requires taking seriously extant theoretical models of international conflict and international political economy. On the one hand, this limits the credibility and generalizability of the conclusions reached here — if the models are flawed, so too will our inferences about the world. On the other hand, this provides a foundation upon which empirical and theoretical research in these subfields can progress in tandem. Otherwise intractable empirical questions can be answered, leveraging the identifying assumptions embedded in these theories. And theories can be revised to account for anomalous or unreasonable empirical results that rest on these assumptions. Taking the models seriously provides answers to hard empirical questions, along with a transparent edifice upon which those answers rest.

\clearpage

# Appendix



## A: Economy


The economy is a variant of that of @Eaton2002. I present the model here for clarity, but refer interested readers to their paper and @Alvarez2007 for derivations and proofs of the existence of a general equilibrium of this economy.

### Consumption

Within each country resides a representative consumer which values tradable goods and nontradable services which are aggregated in Cobb-Douglas utility function, $U_i$.

Consumer utility is Cobb-Douglas in a tradable goods aggregate $Q_i$ and non-tradable services
\begin{equation} \label{eq:CD}
U_i = Q_i^{\nu_i} S_i^{1 - \nu_i}
\end{equation}
$\nu_i$ determines the consumer's relative preference for tradables versus services. Total consumer expenditure is $\tilde{E}_i = E_i^q + E_i^s$ where the Cobb-Douglas preference structure imply $E_i^q = \nu_i \tilde{E}_i$ and $E_i^s = (1 - \nu_i) \tilde{E}_i$.

There is a continuum of tradable varieties indexed $\omega \in [0,1]$ aggregated into $Q_i$ through a constant elasticity of substitution function
\begin{equation} \label{eq:CES}
Q_i = \left( \int_{[0,1]} q_i(\omega)^{\frac{\sigma - 1}{\sigma}} d \omega  \right)^{\frac{\sigma}{\sigma - 1}}
\end{equation}
with $\sigma > 0$. With $E_i^q$ fixed by the upper-level preference structure, consumers maximize $Q_i$ subject to their tradable budget constraint
$$
\int_{[0,1]} p_i(\omega) q_i(\omega) d \omega \leq E_i^q
$$
where $p_i(\omega)$ is the price of variety $\omega$ in country $i$. Let $Q_i^\star$ denote a solution to this problem. The tradable price index $P_i^q$ satisfies $P_i^q Q_i^\star = E_i^q$ with 
$$
P_i^q = \left( \int_{[0,1]} p_i(\omega)^{1 - \sigma} \right)^{\frac{1}{1 - \sigma}}
$$

### Production

Consumers are endowed with labor $L_i$ and earn wage $w_i$ for supplying labor to producers. Services are produced competitively at cost
$$
k_i^s = \frac{w_i}{z_i^s}
$$
where $z_i^s$ is country $i$'s productivity in services. All countries can produce each tradable variety $\omega$. Production requires labor and a tradable goods bundle of intermediate inputs ($Q_i$).  Producing a unit of variety $\omega$ costs
$$
k_i(\omega) = \frac{1}{z_i(\omega)} w_i^{1 - \beta} \left( P_i^q \right)^\beta
$$
with $\beta \in [0,1]$ controlling the share of labor required in production. Total expenditure on intermediates in country $i$ is $E_i^x$. $z_i(\omega)$ controls $i$'s productivity in producing variety $\omega$. $z_i(\omega)$ is a Fréchet-distributed random variable. $F_i(z)$ is the probability $i$'s productivity in producing a tradable variety is less than or equal to $z$. With $F \sim$ Fréchet,
$$
F(z) = \exp \left\{ - T_i z^{-\theta} \right\}
$$
where $T_i$ is a country-specific productivity shifter and $\theta > 1$ is a global parameter that controls the variance of productivity draws around the world. When $\theta$ is large, productivity is less stochastic.

### Trade Frictions

Let $p_{ij}(\omega)$ denote the price in $i$ of a variety $\omega$ produced in $j$. With competitive markets in production, local prices are equal to local costs of production,
$$
p_{ii}(\omega) = k_i(\omega)
$$
When shipped from $i$ to $j$, a variety incurs iceberg freight costs $\delta_{ji}$ and policy costs $\tau_{ji}$, meaning
$$
p_{ji}(\omega) = \tau_{ji} \delta_{ji} p_{ii}(\omega)
$$

Producers and consumers alike search around the world for the cheapest variety $\omega$, inclusive of shipping and policy costs. Equilibrium local prices therefore satisfy
$$
p_i^\star(\omega) = \min_{j \in \left\{ 1,...,N \right\}} \left\{ p_{ij} \right\}
$$
The set of varieties $i$ imports from $j$ is 
$$
\Omega_{ij}^\star = \left\{ \omega \in [0,1] \left. \right\vert p_{ij}(\omega) \leq \min_{k \neq j} \left\{ p_{ik} \right\} \right\}
$$

Total expenditure in country $i$ on goods from $j$ (inclusive of freight costs and policy costs) is $X_{ij}$. At the border, the cost, insurance, and freight (c.i.f.) value of these goods is $X_{ij}^{\text{cif}} = \tau_{ij}^{-1} X_{ij}$. Before shipment, their free on board (f.o.b.) value is $X_{ij}^{\text{fob}} = \left( \delta_{ij} \tau_{ij} \right)^{-1} X_{ij}$

### Tariff Revenue (Policy Rents)

Governments collect the difference between each variety's final value and its c.i.f. value. Total rents for government $i$ are
\begin{equation} \label{eq:revenue}
r_i = \sum_j (\tau_{ij} - 1) X_{ij}^{\text{cif}}
\end{equation}
This revenue is returned to the consumer, but is valued by the government independent of its effect on the consumer's budget.^[This formulation requires the "representative consumer" to encompass individuals that have access to rents and those that do not. It avoids "burning" these rents, as would be implied by a model in which the government valued rents but the consumer did not have access to them.]

### Equilibrium

In equilibrium, national accounts balance and international goods markets clear. Total consumer expenditure is equal to the sum of labor income, tariff revenue, and the value of trade deficits $D_i$
$$
\tilde{E}_i = w_i L_i + r_i + D_i
$$
Labor income is equal to the labor share of all sales of tradables globally and local services sales
\begin{equation} \label{eq:income}
w_i L_i = \sum_j (1 - \beta) X_{ji}^{\text{cif}} + X_i^s
\end{equation}
where
$$
X_i^s = E_i^s = (1 - \nu_i) (w_i L_i + r_i)
$$
The remainder of consumer expenditure is spent on tradables
$$
E_i^q = \nu_i (w_i L_i + r_i) + D_i
$$
A $\beta$-fraction of producer income is spent on intermediates
$$
E_i^x = \sum_j \beta X_{ji}^{\text{cif}}
$$
and total tradable expenditure is
\begin{equation} \label{eq:tExp}
E_i = E_i^q + E_i^x
\end{equation}

The share of $i$'s tradable expenditure spent on goods from $j$ is 
\begin{equation} \label{eq:shares}
x_{ij}(\bm{w}) = \frac{1}{E_i} \int_{\Omega_{ij}^\star} p_{ij}(\omega) q_i^\star \left( p_{ij} (\omega) \right) d \omega = \frac{ T_j \left( \tau_{ij} \delta_{ij} w_j^{1 - \beta} P_j^{\beta} \right)^{-\theta} }{ \frac{1}{C} \left( P_i^q(\bm{w}) \right)^{-\theta}}
\end{equation}
$q_i^\star \left( p_{ij} (\omega) \right)$ is equilibrium consumption of variety $\omega$ from both consumers and producers. $C$ is a constant function of exogenous parameters. The tradable price index is
\begin{equation} \label{eq:Pindex}
P_i^q(\bm{w}) = C \left( \sum_j T_j \left( d_{ij} w_j^{1 - \beta} P_j^{\beta} \right)^{- \theta} \right)^{-\frac{1}{\theta}}
\end{equation}

Finally, I normalize wages to be consistent with world gdp in the data. Denoting world gdp with $Y$, I enforce
\begin{equation} \label{eq:normalization}
Y = \sum_i w_i L_i
\end{equation}

The equilibrium of the economy depends on policy choices $\bm{\tau}$, trade deficits $\bm{D}$, and a vector of structural parameters and constants $\bm{\theta}_h = \left\{ L_i, T_i, \bm{\delta}, \sigma, \theta, \beta, \nu_i, \right\}_{i \in \left\{ 1, ..., N \right\}}$. 



**Definition F1:** 
An *international economic equilibrium* is a mapping $h : \left\{ \bm{\tau}, \bm{D}, \bm{\theta}_h \right\} \rightarrow \mathbb{R}_{++}^N$ with $h(\bm{\tau}, \bm{D}; \bm{\theta}_h) = \bm{w}$ solving the system of equations given by \ref{eq:revenue}, \ref{eq:income}, \ref{eq:tExp}, \ref{eq:shares}, \ref{eq:Pindex}, and \ref{eq:normalization}.

@Alvarez2007 demonstrate the existence and uniqueness of such an equilibrium, subject to some restrictions on the values of structural parameters and the magnitude of trade costs. 

### Welfare

With the equilibrium mapping in hand, I can connect trade policies to government welfare given in Equation \ref{eq:G}. Consumer indirect utility is
\begin{equation} \label{eq:V}
V_i(\bm{w}) = \frac{\tilde{E}_i(\bm{w})}{P_i(\bm{w})}
\end{equation}
where $P_i$ is the aggregate price index in country $i$ and can be written
$$
P_i(\bm{w}) = \left( \frac{P_i^q(\bm{w})}{\nu_i} \right)^{\nu_i} \left( \frac{P_i^s(\bm{w})}{1 - \nu_i} \right)^{1 - \nu_i}
$$
$P_i^q$ is given in equation \ref{eq:Pindex} and $P_i^s = \frac{w_i}{A_i}$. Substituting $\bm{w}$ with its equilibrium value $h(\bm{\tau}, \bm{D}; \bm{\theta}_h)$ returns consumer indirect utility as a function of trade policies. Equilibrium trade flows can be computed as
$$
X_{ij}^{\text{cif}}(\bm{w}) = \tau_{ij}^{-1} x_{ij}(\bm{w}) E_i(\bm{w})
$$
Substituting these into the revenue equation (\ref{eq:revenue}) gives the revenue component of the government's objective function.

### Equilibrium in Changes

In "hats," the equilibrium conditions corresponding to \ref{eq:revenue}, \ref{eq:income}, \ref{eq:tExp}, \ref{eq:shares}, \ref{eq:Pindex}, and \ref{eq:normalization} are
\begin{equation} \label{eq:revenueHat}
\hat{r}_i = \frac{1}{r_i} \left( E_i \hat{E}_i(\hat{\bm{w}}) - \sum_j X_{ij}^{\text{cif}} \hat{X}_{ij}^{\text{cif}}(\hat{\bm{w}}) \right)
\end{equation}
\begin{equation} \label{eq:incomeHat}
\hat{w}_i = \frac{1}{\nu_i w_i L_i} \left( \sum_j \left( (1 - \beta) X_{ji}^{\text{cif}} \hat{X}_{ji}^{\text{cif}}(\hat{\bm{w}}) \right) + (1 - \nu_i) r_i \hat{r}_i(\hat{\bm{w}}) \right)
\end{equation}
\begin{equation} \label{eq:tExpHat}
\hat{E}_i(\hat{\bm{w}}) = \frac{1}{E_i} \left( E_i^q \hat{E}_i^q(\hat{\bm{w}}) + E_i^x \hat{E}_i^x(\hat{\bm{w}}) \right)
\end{equation}
\begin{equation} \label{eq:sharesHat}
\hat{x}_{ij}(\hat{\bm{w}}) = \left( \hat{\tau}_{ij} \hat{w}_j^{1 - \beta} \hat{P}_j(\hat{\bm{w}})^\beta \right)^{-\theta} \hat{P}_i(\hat{\bm{w}})^{\theta}
\end{equation}
\begin{equation} \label{eq:PindexHat}
\hat{P}_i(\hat{\bm{w}}) = \left( \sum_j x_{ij} \left( \hat{\tau}_{ij} \hat{w}_j^{1 - \beta} \hat{P}_j(\hat{\bm{w}})^\beta \right)^{-\theta} \right)^{-\frac{1}{\theta}}
\end{equation}
\begin{equation} \label{eq:normalizationHat}
1 = \sum_i y_i \hat{w}_i
\end{equation}
where
$$
y_i = \frac{w_i L_i}{\sum_j w_j L_j}
$$

This transformation reduces the vector of parameters to be calibrated to $\bm{\theta}_h = \left\{\theta, \beta, \nu_i, \right\}_{i \in \left\{ 1, ..., N \right\}}$.



**Definition A2:** 
An *international economic equilibrium in changes* is a mapping $\hat{h} : \left\{ \hat{\bm{\tau}}, \hat{\bm{D}}, \bm{\theta}_h \right\} \rightarrow \mathbb{R}_{++}^N$ with $\hat{h}(\hat{\bm{\tau}}, \hat{\bm{D}}; \bm{\theta}_h) = \hat{\bm{w}}$ solving the system of equations given by \ref{eq:revenueHat}, \ref{eq:incomeHat}, \ref{eq:tExpHat}, \ref{eq:sharesHat}, \ref{eq:PindexHat}, and \ref{eq:normalizationHat}.

### Welfare in Changes

Now changes in consumer welfare can be calculated for any set of trade policy changes $\hat{\bm{\tau}}$. Manipulating \ref{eq:V}, changes in consumer indirect utility are
\begin{equation} \label{eq:VHat}
\hat{V}_i(\bm{w}) = \frac{\hat{\tilde{E}}_i(\hat{\bm{w}})}{\hat{P}_i(\hat{\bm{w}})}
\end{equation}
where
$$
\hat{P}_i(\hat{\bm{w}}) = \hat{P}_i^q(\hat{\bm{w}})^{\nu_i} \hat{P}_i^s(\hat{\bm{w}})^{\nu_i - 1}
$$
and $\hat{P}_i^q(\hat{\bm{w}})$ is given by equation \ref{eq:PindexHat} and $\hat{P}_i^s(\hat{\bm{w}}) = \hat{w}_i$. Changes in policy rents are given by equation \ref{eq:revenueHat}.



## B: Calibration of Economy


Solving for an international equilibrium in changes (Definition A2) requires data on national accounts ($E_i$, $E_i^q$, $E_i^x$, $w_i L_i$), and international trade flows ($X_{ij}^{\text{cif}}$) (collectively, $\bm{Z}_h$), the magnitude of observed policy barriers to trade ($\tau_{ij}$), and the structural parameters $\theta$, $\beta$, and $\bm{\nu}$ (collectively, $\bm{\theta}_h$). Policy barriers are estimated using the methodology developed in @Cooley2019b. To maintain consistency with the model developed there, I employ the same data on the subset of countries analyzed here. I refer readers to that paper for a deeper discussion of these choices, and briefly summarize the calibration of the economy here. 

### Data

Trade flows valued pre-shipment (free on board) are available from [COMTRADE](https://comtrade.un.org/db/default.aspx). I employ cleaned data from [CEPII](http://www.cepii.fr/CEPII/en/welcome.asp)'s [BACI](http://www.cepii.fr/cepii/en/bdd_modele/presentation.asp?id=1). To get trade in c.i.f. values, I add estimated freight costs from @Cooley2019b to these values. Total home expenditure ($X_{ii} + X_i^s$) and aggregate trade imbalances $D_i$ can then be inferred from national accounts data (GDP, gross output, and gross consumption). GDP gives $w_i L_i$ and gross consumption gives $E_i^s + E_i^q + X_i^x$. To isolate expenditure on services, I use data from the World Bank's International Comparison Program, which reports consumer expenditure shares on various good categories. I classify these as tradable and nontradable, and take the sum over expenditure shares on tradables as the empirical analogue to $\nu_i$. Then, expenditure on services is $X_i^s = (1 - \nu_i) w_i L_i$. 

### Structural Parameters

I set $\theta =$ 6, in line with estimates reported in @Head2014 and @Simonovska2014. A natural empirical analogue for $\beta$ is intermediate imports $(E_i - w_i L_i)$ divided by total tradable production. This varies country to country, however, and equilibrium existence requires a common $\beta$. I therefore take the average of this quantity as the value for $\beta$, which is 0.86 in my data. This means that small changes around the factual equilibrium result in discontinuous jumps in counterfactual predictions. I therefore first generate counterfactual predictions with this common $\beta$, and use these as a baseline for analysis.

### Trade Imbalances

As noted by @Ossa2014, the assumption of exogenous and fixed trade imbalances generates implausible counterfactual predictions when trade frictions get large. I therefore first purge aggregate deficits from the data, solving $\hat{h}(\hat{\bm{\tau}}, \bm{0}; \bm{\theta}_h)$, replicating @Dekle2007. This counterfactual, deficit-less economy is then employed as the baseline, where $\hat{h}(\hat{\bm{\tau}}; \bm{\theta}_h)$ referring to a counterfactual prediction from this baseline.

### Trade Barrier Estimates

![Distribution of policy barriers to trade. Each cell reports the magnitude of the policy barrier each importing country (y-axis) imposes on every exporting country (x-axis). \label{fig:tauhm}](figure/tauhm-1.pdf)



## C: Other Measures of Model Fit




![Absolute errors for each directed dyad in the sample. Positive values indicate that the model predicts a higher trade barrier than is observed in the data (point estimate). \label{fig:fit_ddyad}](figure/fit_ddyad-1.pdf)

![Correlation between trade barrier data and coercion-free model predictions. \label{fig:fit_mo}](figure/fit_mo-1.pdf)

\clearpage

# References

<div id="refs"></div>

# Software

Arel-Bundock V (2020). _countrycode: Convert Country Names and Country
Codes_. R package version 1.2.0, <URL:
https://CRAN.R-project.org/package=countrycode>.

Arel-Bundock V (2020). _modelsummary: Summary Tables and Plots for
Statistical Models and Data: Beautiful, Customizable, and
Publication-Ready_. R package version 0.6.0.9000, <URL:
https://vincentarelbundock.github.io/modelsummary/>.

Arel-Bundock V, Enevoldsen N, Yetman C (2018). "countrycode: An R
package to convert country names and country codes." _Journal of Open
Source Software_, *3*(28), 848. <URL:
https://doi.org/10.21105/joss.00848>.

Boettiger C (2019). _knitcitations: Citations for 'Knitr' Markdown
Files_. R package version 1.0.10, <URL:
https://CRAN.R-project.org/package=knitcitations>.

Francois R (2020). _bibtex: Bibtex Parser_. R package version 0.4.2.2,
<URL: https://CRAN.R-project.org/package=bibtex>.

Henry L, Wickham H (2020). _purrr: Functional Programming Tools_. R
package version 0.3.4, <URL: https://CRAN.R-project.org/package=purrr>.

Kassambara A (2020). _ggpubr: 'ggplot2' Based Publication Ready Plots_.
R package version 0.4.0, <URL:
https://CRAN.R-project.org/package=ggpubr>.

Meschiari S (2015). _latex2exp: Use LaTeX Expressions in Plots_. R
package version 0.4.0, <URL:
https://CRAN.R-project.org/package=latex2exp>.

Müller K, Wickham H (2020). _tibble: Simple Data Frames_. R package
version 3.0.3, <URL: https://CRAN.R-project.org/package=tibble>.

Oliphant TE (2006). _A guide to NumPy_, volume 1. Trelgol Publishing
USA.

Ooms J (2020). _magick: Advanced Graphics and Image-Processing in R_. R
package version 2.4.0, <URL:
https://CRAN.R-project.org/package=magick>.

Pedersen TL (2020). _patchwork: The Composer of Plots_. R package
version 1.0.1, <URL: https://CRAN.R-project.org/package=patchwork>.

R Core Team (2020). _R: A Language and Environment for Statistical
Computing_. R Foundation for Statistical Computing, Vienna, Austria.
<URL: https://www.R-project.org/>.

Ram K, Yochum C (2020). _rdrop2: Programmatic Interface to the
'Dropbox' API_. R package version 0.8.2.1, <URL:
https://CRAN.R-project.org/package=rdrop2>.

Robinson D, Hayes A, Couch S (2020). _broom: Convert Statistical
Objects into Tidy Tibbles_. R package version 0.7.0, <URL:
https://CRAN.R-project.org/package=broom>.

Slowikowski K (2020). _ggrepel: Automatically Position Non-Overlapping
Text Labels with 'ggplot2'_. R package version 0.8.2, <URL:
https://CRAN.R-project.org/package=ggrepel>.

Solt F, Hu Y (2018). _dotwhisker: Dot-and-Whisker Plots of Regression
Results_. R package version 0.5.0, <URL:
https://CRAN.R-project.org/package=dotwhisker>.

Ushey K, Allaire J, Tang Y (2020). _reticulate: Interface to 'Python'_.
R package version 1.16, <URL:
https://CRAN.R-project.org/package=reticulate>.

Van Rossum G, Drake FL (2009). _Python 3 Reference Manual_.
CreateSpace, Scotts Valley, CA. ISBN 1441412697.

Virtanen P, Gommers R, Oliphant TE, Haberland M, Reddy T, Cournapeau D,
Burovski E, Peterson P, Weckesser W, Bright J, van der Walt SJ, Brett
M, Wilson J, Jarrod Millman K, Mayorov N, Nelson AR, Jones E, Kern R,
Larson E, Carey C, Polat İ, Feng Y, Moore EW, Vand erPlas J, Laxalde D,
Perktold J, Cimrman R, Henriksen I, Quintero E, Harris CR, Archibald
AM, Ribeiro AH, Pedregosa F, van Mulbregt P, Contributors S10 (2020).
"SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python."
_Nature Methods_. doi: 10.1038/s41592-019-0686-2 (URL:
https://doi.org/10.1038/s41592-019-0686-2).

Wächter A, Biegler LT (2006). "On the implementation of an
interior-point filter line-search algorithm for large-scale nonlinear
programming." _Mathematical Programming_, *106*(1), 25-57.

Wickham H (2020). _forcats: Tools for Working with Categorical
Variables (Factors)_. R package version 0.5.0, <URL:
https://CRAN.R-project.org/package=forcats>.

Wickham H (2020). _reshape2: Flexibly Reshape Data: A Reboot of the
Reshape Package_. R package version 1.4.4, <URL:
https://CRAN.R-project.org/package=reshape2>.

Wickham H (2019). _stringr: Simple, Consistent Wrappers for Common
String Operations_. R package version 1.4.0, <URL:
https://CRAN.R-project.org/package=stringr>.

Wickham H (2020). _tidyr: Tidy Messy Data_. R package version 1.1.2,
<URL: https://CRAN.R-project.org/package=tidyr>.

Wickham H (2019). _tidyverse: Easily Install and Load the 'Tidyverse'_.
R package version 1.3.0, <URL:
https://CRAN.R-project.org/package=tidyverse>.

Wickham H (2016). _ggplot2: Elegant Graphics for Data Analysis_.
Springer-Verlag New York. ISBN 978-3-319-24277-4, <URL:
https://ggplot2.tidyverse.org>.

Wickham H (2007). "Reshaping Data with the reshape Package." _Journal
of Statistical Software_, *21*(12), 1-20. <URL:
http://www.jstatsoft.org/v21/i12/>.

Wickham H, Averick M, Bryan J, Chang W, McGowan LD, François R,
Grolemund G, Hayes A, Henry L, Hester J, Kuhn M, Pedersen TL, Miller E,
Bache SM, Müller K, Ooms J, Robinson D, Seidel DP, Spinu V, Takahashi
K, Vaughan D, Wilke C, Woo K, Yutani H (2019). "Welcome to the
tidyverse." _Journal of Open Source Software_, *4*(43), 1686. doi:
10.21105/joss.01686 (URL: https://doi.org/10.21105/joss.01686).

Wickham H, Chang W, Henry L, Pedersen TL, Takahashi K, Wilke C, Woo K,
Yutani H, Dunnington D (2020). _ggplot2: Create Elegant Data
Visualisations Using the Grammar of Graphics_. R package version 3.3.2,
<URL: https://CRAN.R-project.org/package=ggplot2>.

Wickham H, François R, Henry L, Müller K (2020). _dplyr: A Grammar of
Data Manipulation_. R package version 1.0.2, <URL:
https://CRAN.R-project.org/package=dplyr>.

Wickham H, Hester J, Francois R (2018). _readr: Read Rectangular Text
Data_. R package version 1.3.1, <URL:
https://CRAN.R-project.org/package=readr>.

Xiao N (2018). _ggsci: Scientific Journal and Sci-Fi Themed Color
Palettes for 'ggplot2'_. R package version 2.9, <URL:
https://CRAN.R-project.org/package=ggsci>.

Xie Y (2020). _knitr: A General-Purpose Package for Dynamic Report
Generation in R_. R package version 1.29, <URL:
https://CRAN.R-project.org/package=knitr>.

Xie Y (2015). _Dynamic Documents with R and knitr_, 2nd edition.
Chapman and Hall/CRC, Boca Raton, Florida. ISBN 978-1498716963, <URL:
https://yihui.org/knitr/>.

Xie Y (2014). "knitr: A Comprehensive Tool for Reproducible Research in
R." In Stodden V, Leisch F, Peng RD (eds.), _Implementing Reproducible
Computational Research_. Chapman and Hall/CRC. ISBN 978-1466561595,
<URL: http://www.crcpress.com/product/isbn/9781466561595>.

Zhu H (2020). _kableExtra: Construct Complex Table with 'kable' and
Pipe Syntax_. R package version 1.2.1, <URL:
https://CRAN.R-project.org/package=kableExtra>.



