# Gaussian processes (one-hour tutorial)

A short Jupyter notebook that implements Gaussian processes from scratch with numpy and matplotlib. It continues Bayesian linear regression with features: raw data $X \in \mathbb{R}^{n \times d}$, feature matrix $\Phi \in \mathbb{R}^{n \times d'}$, weights $w \sim \mathcal{N}(0, \tau^2 I)$, labels $y = \Phi w + \sigma\varepsilon$, and the joint covariance of $\begin{bmatrix}w \\ y\end{bmatrix}$. Aimed at a one-hour class: priors, kernels, posterior regression, and the effect of lengthscale, noise, and $\tau$. There is no hyperparameter fitting.

## Prerequisites

- Bayesian linear regression with features (weight prior $w \sim \mathcal{N}(0, \tau^2 I)$)
- Multivariate Gaussians and joint covariance of $(w, y)$
- Basic linear algebra (matrix multiply, Cholesky)

## Setup

Install [uv](https://docs.astral.sh/uv/), then from this directory:

```bash
uv sync
uv run jupyter notebook gaussian_processes.ipynb
```

## Suggested pacing (60 min)

| Time | Section | What to emphasize |
| --- | --- | --- |
| 0–10 min | From BLR to GPs | Raw $X$ vs features $\Phi$; $w \sim \mathcal{N}(0,\tau^2 I)$, $y = \Phi w + \sigma\varepsilon$, joint $[w,y]$; GP is the view from $f = \Phi w$ |
| 10–18 min | Kernels | $K = \Phi\Phi^\top$; RBF, Matérn-5/2, Laplacian, periodic; linear $u^\top v$; polynomial $(1+u^\top v)^\gamma$ |
| 18–28 min | Sampling from the prior | Draw $f \sim \mathcal{N}(0, \tau^2 K)$ with Cholesky; never form $w$ or $\Phi$ |
| 28–35 min | Comparing kernels | Same noise draws, different $\phi$ |
| 35–48 min | Posterior / GP regression | Condition using $K_y = \tau^2 K + \sigma^2 I$ |
| 48–60 min | Lengthscale, $\sigma$, and $\tau$ | Too small / too large $\ell$; noise; scaling $\tau$ vs inversely scaling $\sigma$ |

The notebook is `gaussian_processes.ipynb`. Helper functions live in the notebook itself so the linear algebra stays on the page.
