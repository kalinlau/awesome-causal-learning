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
|VAE|[Louizos, Christos, Uri Shalit, Joris M. Mooij, David Sontag, Richard Zemel, and Max Welling. "Causal effect inference with deep latent-variable models." In Advances in Neural Information Processing Systems, pp. 6446-6456. 2017.](http://papers.nips.cc/paper/7223-causal-effect-inference-with-deep-latent-variable-models.pdf)|[Python](https://github.com/AMLab-Amsterdam/CEVAE)|
|SITE|[Yao, Liuyi, Sheng Li, Yaliang Li, Mengdi Huai, Jing Gao, and Aidong Zhang. "Representation Learning for Treatment Effect Estimation from Observational Data." In Advances in Neural Information Processing Systems, pp. 2638-2648. 2018.](https://papers.nips.cc/paper/7529-representation-learning-for-treatment-effect-estimation-from-observational-data.pdf)|[Python](https://github.com/Osier-Yi/SITE)|
|X-learner|[Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. "Metalearners for estimating heterogeneous treatment effects using machine learning." Proceedings of the National Academy of Sciences 116, no. 10 (2019): 4156-4165.](https://www.pnas.org/content/pnas/early/2019/02/14/1804597116.full.pdf)|[R](https://github.com/soerenkuenzel/hte)[R](https://github.com/xnie/rlearner/blob/master/R/xlearner.R)|
|Causal Forest|[Wager, Stefan, and Susan Athey. "Estimation and inference of heterogeneous treatment effects using random forests." Journal of the American Statistical Association just-accepted (2017).](https://www.tandfonline.com/doi/pdf/10.1080/01621459.2017.1319839)|[R](https://github.com/grf-labs/grf) [Python](https://github.com/kjung/scikit-learn)|
|Causal MARS, Causal Boosting, Pollinated Transformed Outcome Forests|[S. Powers et al., “Some methods for heterogeneous treatment effect estimation in high-dimensions,” 2017.](https://arxiv.org/pdf/1707.00102.pdf)|[R](https://github.com/grf-labs/grf) [R](https://github.com/saberpowers/causalLearning)|
|Bayesian Additive Regression Trees (BART)|[Hill, Jennifer L. "Bayesian nonparametric modeling for causal inference." Journal of Computational and Graphical Statistics 20, no. 1 (2011): 217-240.](https://www.tandfonline.com/doi/pdf/10.1198/jcgs.2010.08162)|[Python](https://github.com/JakeColtman/bartpy)|
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

### From Non-IID data

## causal representation learning

## causal reinforcement learning
