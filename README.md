# awesome-causal-learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
Causality with machine learning, topic including causal represenation learning, causal reinforcement learning

## Table of Contents

- [Causal Machine Learning](#causal-machine-learning)
- [Causal Representation Learning](#causal-representation-learning)
- [Causal Reinforcement Learning](#causal-reinforcement-learning)

## causal machine learning

### From IID data

#### Individual Treatment Effects (ITE)
|Name|Paper|Code|
|---|---|---|
|Propensity Score Matching|[Rosenbaum, Paul R., and Donald B. Rubin. "The central role of the propensity score in observational studies for causal effects." Biometrika 70, no. 1 (1983): 41-55.](https://academic.oup.com/biomet/article-pdf/70/1/41/662954/70-1-41.pdf)|[Python](https://github.com/akelleh/causality/tree/master/causality/estimation)|
|Nonparametric Regression Adjustment|   |[Python](https://github.com/akelleh/causality)|
|BNN, BLR|[Johansson, Fredrik, Uri Shalit, and David Sontag. "Learning representations for counterfactual inference." 33rd International Conference on Machine Learning (ICML), June 2016.](http://www.jmlr.org/proceedings/papers/v48/johansson16.pdf)||
|TARNet, CFR|[Shalit, Uri, Fredrik D. Johansson, and David Sontag. "Estimating individual treatment effect: generalization bounds and algorithms." 34th International Conference on Machine Learning (ICML), August 2017.](https://arxiv.org/pdf/1606.03976)|[Python](https://github.com/clinicalml/cfrnet)|
|CEVAE|[Louizos, Christos, Uri Shalit, Joris M. Mooij, David Sontag, Richard Zemel, and Max Welling. "Causal effect inference with deep latent-variable models." In Advances in Neural Information Processing Systems, pp. 6446-6456. 2017.](http://papers.nips.cc/paper/7223-causal-effect-inference-with-deep-latent-variable-models.pdf)|[Python](https://github.com/AMLab-Amsterdam/CEVAE)|
|SITE|[Yao, Liuyi, Sheng Li, Yaliang Li, Mengdi Huai, Jing Gao, and Aidong Zhang. "Representation Learning for Treatment Effect Estimation from Observational Data." In Advances in Neural Information Processing Systems, pp. 2638-2648. 2018.](https://papers.nips.cc/paper/7529-representation-learning-for-treatment-effect-estimation-from-observational-data.pdf)|[Python](https://github.com/Osier-Yi/SITE)|
|X-learner|[Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for estimating heterogeneous treatment effects using machine learning." Proceedings of the National Academy of Sciences 116, no. 10 (2019): 4156-4165.](https://www.pnas.org/content/pnas/early/2019/02/14/1804597116.full.pdf)|[R](https://github.com/soerenkuenzel/hte)[R](https://github.com/xnie/rlearner/blob/master/R/xlearner.R)|
|Causal Forest|[Wager, Stefan, and Susan Athey. "Estimation and inference of heterogeneous treatment effects using random forests." Journal of the American Statistical Association just-accepted (2017).](https://www.tandfonline.com/doi/pdf/10.1080/01621459.2017.1319839)|[R](https://github.com/grf-labs/grf) [Python](https://github.com/kjung/scikit-learn)|
|Causal MARS, Causal Boosting, Pollinated Transformed Outcome Forests|[S. Powers et al., “Some methods for heterogeneous treatment effect estimation in high-dimensions,” 2017.](https://arxiv.org/pdf/1707.00102.pdf)|[R](https://github.com/grf-labs/grf) [R](https://github.com/saberpowers/causalLearning)|
|BART|[Hill, Jennifer L. "Bayesian nonparametric modeling for causal inference." Journal of Computational and Graphical Statistics 20, no. 1 (2011): 217-240.](https://www.tandfonline.com/doi/pdf/10.1198/jcgs.2010.08162)|[Python](https://github.com/JakeColtman/bartpy)|
|GANITE|[Yoon, Jinsung, James Jordon, and Mihaela van der Schaar. "GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets." (2018).](https://openreview.net/forum?id=ByKWUeWA-)|[Python](https://github.com/jsyoon0823/GANITE)|
|Perfect Match|[Schwab, Patrick, Lorenz Linhardt, and Walter Karlen. "Perfect match: A simple method for learning representations for counterfactual inference with neural networks." arXiv preprint arXiv:1810.00656 (2018)](https://arxiv.org/pdf/1810.00656)|[Python](https://github.com/d909b/perfect_match)|
|Active Learning for Decision-Making from Imbalanced Observational Data|[Active Learning for Decision-Making from Imbalanced Observational Data](https://arxiv.org/abs/1904.05268)|NA|
|ABCEI|[Adversarial Balancing-based Representation Learning for Causal Effect Inference with Observational Data](https://arxiv.org/pdf/1904.13335.pdf)|NA|
|NSGP (Non-stationary Gaussian Process Prior)|[Alaa, Ahmed, and Mihaela Schaar. "Limits of estimating heterogeneous treatment effects: Guidelines for practical algorithm design." In International Conference on Machine Learning, pp. 129-138. 2018.](http://proceedings.mlr.press/v80/alaa18a/alaa18a.pdf)|NA|
|CMGP (Causal Multi-task Gaussian Processes)|[Alaa, Ahmed M., and Mihaela van der Schaar. "Bayesian inference of individualized treatment effects using multi-task gaussian processes." In Advances in Neural Information Processing Systems, pp. 3424-3432. 2017.](https://papers.nips.cc/paper/6934-bayesian-inference-of-individualized-treatment-effects-using-multi-task-gaussian-processes.pdf)|NA|
|BNR-NNM(balanced and nonlinear representations-nearest neighbor matching)|[Li, Sheng, and Yun Fu. "Matching on balanced nonlinear representations for treatment effects estimation." In Advances in Neural Information Processing Systems, pp. 929-939. 2017.](http://papers.nips.cc/paper/6694-matching-on-balanced-nonlinear-representations-for-treatment-effects-estimation.pdf)|NA|
|Deep Counterfactual Networks (Propensity Dropout)|[Alaa, Ahmed M., Michael Weisz, and Mihaela van der Schaar. "Deep counterfactual networks with propensity-dropout." arXiv preprint arXiv:1706.05966 (2017)](https://arxiv.org/pdf/1706.05966)|NA|
|interval estimation(sensitivity analysis)|[Kallus, Nathan, Xiaojie Mao, and Angela Zhou. "Interval Estimation of Individual-Level Causal Effects Under Unobserved Confounding." In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 2281-2290. 2019.](https://arxiv.org/abs/1810.02894)|NA|


#### Averaged Treatment Effects (ATE)
|Name|Paper|Code|
|---|---|---|
|Dragonnet|[Adapting Neural Networks for the Estimation of Treatment Effects](https://arxiv.org/abs/1906.02120)|[Python](https://github.com/claudiashi57/dragonnet)|
|Inverse Probability Reweighting|[Rosenbaum, Paul R., and Donald B. Rubin. "The central role of the propensity score in observational studies for causal effects." Biometrika 70, no. 1 (1983): 41-55.](https://academic.oup.com/biomet/article-pdf/70/1/41/662954/70-1-41.pdf)|[R](https://github.com/cran/ipw)|
|Doubly Robust Estimation|[Bang, Heejung, and James M. Robins. "Doubly robust estimation in missing data and causal inference models." Biometrics 61, no. 4 (2005): 962-973.](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1541-0420.2005.00377.x)   |[R](https://github.com/gregridgeway/fastDR)|
|Doubly Robust Estimation for High Dimensional Data|[Antonelli, Joseph, Matthew Cefalu, Nathan Palmer, and Denis Agniel. "Doubly robust matching estimators for high dimensional confounding adjustment." Biometrics (2016).](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6347556/)|[R](https://github.com/jantonelli111/DoublyRobustHD)|
|TMLE|[Gruber, Susan, and Mark J. van der Laan. "tmle: An R package for targeted maximum likelihood estimation." (2011).](https://www.jstatsoft.org/article/view/v051i13)|[R](https://cran.r-project.org/web/packages/tmle/index.html)|
|Entropy Balancing|[Hainmueller, Jens. "Entropy balancing for causal effects: A multivariate reweighting method to produce balanced samples in observational studies." Political Analysis 20, no. 1 (2012): 25-46.](https://www.jstor.org/stable/pdf/41403737.pdf)|[R](https://github.com/cran/ebal)|
|CBPS(Covariate Balancing Propensity Score)|[Imai, Kosuke, and Marc Ratkovic. "Covariate balancing propensity score." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 76, no. 1 (2014): 243-263.](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12027)|[R](https://github.com/kosukeimai/CBPS)|
|Approximate Residual Balancing|[Athey, Susan, Guido W. Imbens, and Stefan Wager. "Approximate residual balancing: debiased inference of average treatment effects in high dimensions." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 80, no. 4 (2018): 597-623.](https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/rssb.12268)|[R](https://github.com/swager/balanceHD)|
|Differentiated Confounder Balancing|[Kuang, Kun, Peng Cui, Bo Li, Meng Jiang, and Shiqiang Yang. "Estimating Treatment Effect in the Wild via Differentiated Confounder Balancing." In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 265-274. ACM, 2017.](http://media.cs.tsinghua.edu.cn/~multimedia/cuipeng/papers/CausalDCA.pdf)|NA|
|Adversarial Balancing|[Ozery-Flato, Michal, Pierre Thodoroff, and Tal El-Hay. "Adversarial Balancing for Causal Inference." arXiv preprint arXiv:1810.07406 (2018).](https://arxiv.org/pdf/1810.07406)|NA|
|DeepMatch|[Kallus, Nathan. "Deepmatch: Balancing deep covariate representations for causal inference using adversarial training." arXiv preprint arXiv:1802.05664 (2018).](https://arxiv.org/pdf/1802.05664)|NA|

#### Continuous Treatment Effects
|Name|Paper|Code|
|---|---|---|
|Causal Dose-Response Curves / Causal Curves|[Kobrosly, R. W., (2020). causal-curve: A Python Causal Inference Package to Estimate Causal Dose-Response Curves. Journal of Open Source Software, 5(52), 2523, https://doi.org/10.21105/joss.02523](https://joss.theoj.org/papers/10.21105/joss.02523)|[Python](https://github.com/ronikobrosly/causal-curve)|
|RespSVM|[Kallus, Nathan. "Classifying Treatment Responders Under Causal Effect Monotonicity." arXiv preprint arXiv:1902.05482 (2019)](https://arxiv.org/pdf/1902.05482.pdf)|NA|
|Dose response networks (DRNets)|[Schwab, Patrick, Lorenz Linhardt, Stefan Bauer, Joachim M. Buhmann, and Walter Karlen. "Learning Counterfactual Representations for Estimating Individual Dose-Response Curves." arXiv preprint arXiv:1902.00981 (2019).](https://arxiv.org/pdf/1902.00981.pdf)|[Python](https://github.com/d909b/drnet)|

#### Instrumental Variables

| Name     | Paper                                                        | Code                                          |
| -------- | ------------------------------------------------------------ | --------------------------------------------- |
| DeepIV   | [Hartford, Jason, Greg Lewis, Kevin Leyton-Brown, and Matt Taddy. "Deep iv: A flexible approach for counterfactual prediction." In International Conference on Machine Learning, pp. 1414-1423. 2017.](http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf) | [Python](https://github.com/jhartford/DeepIV) |
| PDSLasso | [Achim Ahrens & Christian B. Hansen & Mark E Schaffer, 2018. "PDSLASSO: Stata module for post-selection and post-regularization OLS or IV estimation and inference," Statistical Software Components S458459, Boston College Department of Economics, revised 24 Jan 2019.](https://ideas.repec.org/c/boc/bocode/s458459.html) | [STATA](https://statalasso.github.io/)        |

#### Multi-Cause: An important discussion
| Name         | Paper                                                        | Code                                                        |
| ------------ | ------------------------------------------------------------ | ----------------------------------------------------------- |
| Deconfounder | [Wang, Yixin, and David M. Blei. "The blessings of multiple causes." arXiv preprint arXiv:1805.06826 (2018).](https://arxiv.org/abs/1805.06826) | [Python](https://github.com/blei-lab/deconfounder_tutorial) |
|              | [Imai, Kosuke, and Zhichao Jiang. "Discussion of "The Blessings of Multiple Causes" by Wang and Blei."](https://imai.fas.harvard.edu/research/files/deconfounder.pdf) | NA                                                          |
|              | [D'Amour, Alexander. "On multi-cause causal inference with unobserved confounding: Counterexamples, impossibility, and alternatives." arXiv preprint arXiv:1902.10286 (2019).](https://arxiv.org/abs/1902.10286) | NA                                                          |
|              | [Ranganath, Rajesh, and Adler Perotte. "Multiple causal inference with latent confounding." arXiv preprint arXiv:1805.08273 (2018).](https://arxiv.org/pdf/1805.08273) | NA                                                          |
|              | [Kong, Dehan, Shu Yang, and Linbo Wang. "Multi-cause causal inference with unmeasured confounding and binary outcome." arXiv preprint arXiv:1907.13323 (2019).](https://arxiv.org/pdf/1907.13323.pdf) | NA                                                          |
|              | [Elizabeth L. Ogburn, Ilya Shpitser, Eric J. Tchetgen Tchetgen "Comment on Blessings of Multiple Causes." arXiv preprint arXiv:1910.05438 (2019)](https://arxiv.org/abs/1910.05438v2?from=timeline) | NA                                                          |

### From Non-IID data
#### social network
|Name|Paper|Code|
|---|---|---|
|Network Deconfounder|[Guo, Ruocheng, Jundong Li, and Huan Liu. "Learning Individual Causal Effects from Networked Observational Data." WSDM 2020.](https://arxiv.org/abs/1906.03485)|[Python](https://github.com/rguo12/network-deconfounder-wsdm20)|
|Causal Inference with Network Embeddings|[Veitch, Victor, Yixin Wang, and David M. Blei. "Using embeddings to correct for unobserved confounding." arXiv preprint arXiv:1902.04114 (2019).](https://arxiv.org/pdf/1902.04114)|[Python](https://github.com/vveitch/causal-network-embeddings)|
### Spillover Effect/Interference
|Name|Paper|Code|
|---|---|---|
|Linked Causal Variational Autoencoder (LCVA)|[Rakesh, Vineeth, Ruocheng Guo, Raha Moraffah, Nitin Agarwal, and Huan Liu. "Linked Causal Variational Autoencoder for Inferring Paired Spillover Effects." CIKM 2018.](http://www.public.asu.edu/~rguo12/CIKM18.pdf)|[Python](https://github.com/rguo12/CIKM18-LCVA)|
|GNN-based Causal Effect Estimators|[Ma, Yunpu, Yuyi Wang, and Volker Tresp. "Causal Inference under Networked Interference." arXiv preprint arXiv:2002.08506 (2020).](https://arxiv.org/pdf/2002.08506.pdf)|NA|
###  Time Varying/Dependent Causal Effects
|Name|Paper|Code|
|---|---|---|
|Time Series Deconfounder|[Bica, Ioana, Ahmed M. Alaa, and Mihaela van der Schaar. "Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders." arXiv preprint arXiv:1902.00450 (2019).](https://arxiv.org/pdf/1902.00450.pdf)|NA|
|Recurrent Marginal Structural Networks|[Lim, Bryan. "Forecasting Treatment Responses Over Time Using Recurrent Marginal Structural Networks." In Advances in Neural Information Processing Systems, pp. 7494-7504. 2018.](http://medianetlab.ee.ucla.edu/papers/nips_rmsn.pdf)|[Python](https://github.com/sjblim/rmsn_nips_2018)|
|Longitudinal Targeted Maximum Likelihood Estimation|[Petersen, Maya, Joshua Schwab, Susan Gruber, Nello Blaser, Michael Schomaker, and Mark van der Laan. "Targeted maximum likelihood estimation for dynamic and static longitudinal marginal structural working models." Journal of causal inference 2, no. 2 (2014): 147-185.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4405134/)|[R](https://github.com/joshuaschwab/ltmle)|
### Application in ML
#### Recommendation

#### NLP

| Name                                                         | Paper                                                        | Code                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| A Review of Using Text to Remove Confounding from Causal Estimates | [Keith, Katherine A., David Jensen, and Brendan O'Connor. "Text and Causal Inference: A Review of Using Text to Remove Confounding from Causal Estimates." ACL 2020.](https://www.aclweb.org/anthology/2020.acl-main.474.pdf) | NA                                                           |
| Causal Analysis with Lexicons                                | [Pryzant, Reid, Kelly Shen, Dan Jurafsky, and Stefan Wagner. "Deconfounded lexicon induction for interpretable social science." NAACL 2018.](https://www.aclweb.org/anthology/N18-1146.pdf) | [Python](https://github.com/rpryzant/causal_attribution)     |
| Causal Text Embeddings                                       | [Veitch, Victor, Dhanya Sridhar, and David M. Blei. "Using Text Embeddings for Causal Inference." arXiv preprint arXiv:1905.12741 (2019).](https://arxiv.org/pdf/1905.12741.pdf) | [Python](https://github.com/blei-lab/causal-text-embeddings) |
| Handling Missing/Noisy Treatment                             | [Wood-Doughty, Zach, Ilya Shpitser, and Mark Dredze. "Challenges of Using Text Classifiers for Causal Inference." In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 4586-4598. 2018.](https://aclweb.org/anthology/D18-1488) | [Python](https://github.com/zachwooddoughty/emnlp2018-causal) |
| Conditional Treatment-adversarial Learning Based Matching    | [Yao, Liuyi, Sheng Li, Yaliang Li, Hongfei Xue, Jing Gao, and Aidong Zhang. "On the estimation of treatment effect with text covariates." In Proceedings of the 28th International Joint Conference on Artificial Intelligence, pp. 4106-4113. AAAI Press, 2019.](https://pdfs.semanticscholar.org/2e2f/39232771711248940f68c3c1d6bd0a22c3e4.pdf) | NA                                                           |
| Causal Inferences Using Texts                                | [Egami, Naoki, Christian J. Fong, Justin Grimmer, Margaret E. Roberts, and Brandon M. Stewart. "How to make causal inferences using texts." arXiv preprint arXiv:1802.02163 (2018).](https://arxiv.org/pdf/1802.02163.pdf) | NA                                                           |


#### CV

## causal representation learning

## causal reinforcement learning

