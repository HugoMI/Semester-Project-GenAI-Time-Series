# Semester-Project-GenAI-Time-Series
Semester project at ETH Zurich: Generative AI for time series generation

Abstract

Generation of artificial financial time series is an active research area which aims to develop alternative methods to synthesize data which, in most of the cases, is scarce or difficult to access. In this work, the author compare and evaluate two existing recent and novel machine learning approaches, generative-artificial intelligence (AI) models specifically created for the generation of time series of historical stock prices, against a baseline model to produce long time series.

One of these two models combines the attributes of the Wasserstein Generative Adversarial Networks (WGANs) with the mathematical framework of the signature of a path, which produces an easier formulation of the optimization problem and reduce training times: Conditional Signature-WGAN (Sig-CWGAN), with additional metrics during training. While a second model, created specifically for long time series generation, uses a transformers architecture along with a diffusion model (TransFusion). Both generative models and the baseline model (GARCH) are tuned and trained to generate longer time series than usual and original implementations, over $800$ timestamps.

Lastly, the models are assessed by computing some of the most common _stylized empirical features_ (i.e. observed and characteristic statistical properties of financial time series) of the sampled paths where Sig-CWGAN outperforms in features where GARCH is not able to.


Models were taken from:

*GARCH (from `arch` library)
  [Documentation](https://arch.readthedocs.io/en/latest/univariate/introduction.html)

*Sig-CWGAN
  [Article](https://arxiv.org/abs/2006.05421)
  [Code](https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs)

*TransFusion
  [Article](https://arxiv.org/abs/2307.12667)
  [Code](https://github.com/fahim-sikder/TransFusion)
