# ContrastiveVAE: Robust Representation Learning for Cross-Sectional Asset Pricing

A self-supervised, probabilistic encoder–decoder architecture for extracting **latent macroeconomic factors** from high-dimensional, noisy financial time-series data. The model combines a Pre-LN Transformer with **Gated Linear Units (GLUs)**, an **"Oracle" posterior network** trained on future returns, and a novel **Cross-Sectional Contrastive Learning (CSCL)** framework to produce regime-aware, permutation-invariant latent representations of the equity market.

---

## Motivation

Cross-sectional asset pricing is fundamentally a **representation learning problem**: map thousands of equities into a small set of structural risk factors while filtering out idiosyncratic noise. Classical linear factor models (Fama–French, APT) are interpretable but cannot capture the non-linear, time-varying dynamics of modern markets. Deep generative models like FactorVAE move in the right direction, but two issues remain:

1. **Brittle temporal encoders.** GRU-based feature extractors struggle with long-range dependencies and dilute signal under the extreme noise typical of daily equity data.
2. **No subset invariance.** The same macroeconomic regime should produce the same latent factors regardless of *which* stocks happen to be available on a given day. Standard VAEs offer no inductive bias for this.

ContrastiveVAE addresses both issues with targeted architectural interventions and a self-supervised contrastive objective.

---

## Key Contributions

1. **Dynamic Noise Filtration + Stabilized Temporal Attention.** A GLU-gated input layer learns to mute irrelevant technical indicators per stock, followed by a Pre-Layer Normalized Transformer with a learnable `[CLS]` token for clean temporal aggregation.
2. **Cross-Sectional Contrastive Learning (CSCL).** For each trading day, two disjoint random subsets of the stock universe are drawn as "views." A symmetric InfoNCE loss aligns their latent representations, forcing the model to recognize that the underlying regime is invariant to the specific stocks observed.

---

## Architecture

```
       Historical X                          Future Returns y
            │                                       │
            ▼                                       ▼
   ┌──────────────────┐                  ┌───────────────────┐
   │ GLU Noise Filter │                  │ Dynamic Portfolio │
   │        ↓         │                  │       Layer       │
   │ Pre-LN Transformer│                  └─────────┬─────────┘
   │   + [CLS] Token  │                            │
   └────────┬─────────┘                            ▼
            │ stock features e               ┌──────────────┐
            ▼                                │  MLP + Gauss │
   ┌──────────────────┐                      │   (Oracle)   │
   │  Multi-Head      │                      └──────┬───────┘
   │  Global Attention│                             │
   │       ↓          │                       μ_post, σ_post
   │   MLP + Gauss    │                             │
   │    (Predictor)   │                             │
   └────────┬─────────┘                             │
            │                                       │
       μ_prior, σ_prior ───────► KL divergence ◄────┘
            │
            ▼  reparameterize
        z ~ N(μ, σ²)
            │
            ▼
       ┌─────────┐
       │ Decoder │ ──► predicted returns ŷ
       └─────────┘
```

### 1. Temporal Feature Extractor

- **GLU gate**: `Gate(X) = (XW₁ + b₁) ⊙ σ(XW₂ + b₂)`, with `b₂` initialized to 1.0 to avoid "dead gates" at the start of training.
- **Pre-LN Transformer Encoder** (`norm_first=True`) for stable gradient flow through deep stacks on noisy data.
- **Learnable `[CLS]` token** aggregates the temporal trajectory; final stock representation `eᵢ` is read from the `[CLS]` output state, avoiding the signal dilution of mean-pooling.

### 2. Factor Encoder — "The Oracle" (φ_enc)

The Oracle has privileged access to future returns `y` and produces the *optimal* posterior factors that the Predictor will be trained to approximate.

- **Dynamic Portfolio Layer**: a softmax over stock features maps N stocks to M ≪ N portfolios, then aggregates future returns into portfolio returns `y_port = Wᵀ_port · y`.
- **Gaussian parameterization**: an MLP outputs `μ_post` and `σ_post = Softplus(·)`.

### 3. Factor Predictor — The Prior (φ_pred)

The Predictor uses **only historical data** and is what is actually deployed at inference time.

- **Multi-head global attention** with a learnable query matrix `Q_global ∈ ℝ^(K × d_model)`. Keys and values are linear projections of `e`; heads are concatenated to capture diverse systemic risk premiums.
- **Prior distribution network**: MLP outputs `μ_prior` and `σ_prior`.

### 4. Factor Decoder (φ_dec)

Latent factors `z = μ + σ ⊙ ε` (reparameterization trick) are combined with stock features `e` to produce the predicted cross-sectional return distribution.

### 5. Cross-Sectional Contrastive Learning

For each trading day `t`, two disjoint random subsets `X⁽¹⁾`, `X⁽²⁾` are passed through the Feature Extractor and Predictor, yielding `μ⁽¹⁾_prior` and `μ⁽²⁾_prior`. A symmetric InfoNCE loss aligns same-day views as positives against all other days in the batch as negatives:

```
ℓ(u, v) = -log [ exp(sim(u, v)/τ) / Σⱼ exp(sim(u, kⱼ)/τ) ]

L_CSCL = (1 / 2B) Σᵢ [ ℓ(μ⁽¹⁾ᵢ, μ⁽²⁾ᵢ) + ℓ(μ⁽²⁾ᵢ, μ⁽¹⁾ᵢ) ]
```

---

## Training Objective

End-to-end joint optimization of the ELBO and the contrastive regularizer:

```
L_VAE   = NLL(reconstructed returns) + γ · KL( q(z|x,y) ‖ p(z|x) )
L_Total = L_VAE + λ · L_CSCL
```

`γ` and `λ` control the strength of the KL term and the contrastive penalty, respectively.

---

## Dataset

- **Feature set**: Alpha158 technical indicators (via [Qlib](https://github.com/microsoft/qlib))
- **Universe**: US equities (Yahoo Finance source)
- **Train**: 2010 – 2016
- **Out-of-sample test**: 2018 – 2019

---

## Results

### Predictive Capability

| Metric | Value |
|---|---|
| Rank Information Coefficient (Rank IC) | **0.038** |

### Portfolio Backtest (TDrisk Strategy)

Stocks are ranked by a risk-adjusted score `Score = μ_pred − η · σ_pred` rather than expected return alone. The portfolio is a **daily-rebalanced Top-50 equal-weight book** with realistic turnover constraints (max 5 stocks rotated per day).

| Strategy | Annualized Return | Sharpe Ratio | Max Drawdown |
|---|---|---|---|
| ContrastiveVAE (TDrisk Score) | **17.34%** | **1.73** | **6.53%** |

The asymmetric risk-return profile — high Sharpe with shallow drawdown — validates the core premise: penalizing predicted variance immunizes the portfolio against catastrophic noise events.

---

## Repository Structure

```
.
├── main.py                 # Training / evaluation entry point
├── model/
│   ├── feature_extractor.py    # GLU + Pre-LN Transformer + [CLS]
│   ├── encoder.py              # Oracle (posterior network)
│   ├── predictor.py            # Prior network with global attention
│   ├── decoder.py              # Return decoder
│   └── contrastive.py          # Symmetric InfoNCE loss
├── data/                   # Alpha158 loaders, view sampling
├── backtest/               # TDrisk portfolio simulator
├── run_factor.sh           # Slurm submission script
├── SETUP.md                # Cluster setup + run instructions
└── README.md               # This file
```

---

## Quickstart

Cluster setup, environment creation, and Slurm submission are documented in [`SETUP.md`](SETUP.md). Once the environment is built and the Qlib US dataset is downloaded:

```bash
# Full training
sbatch run_factor.sh

# Quick evaluation only
python main.py --mode eval
```

Monitor live training output:

```bash
tail -f factor_vae_*.out
```

---

## Future Work

The current model assumes Gaussian latent factors. Real return distributions exhibit **skewness and fat tails**, suggesting that replacing the Gaussian prior with **Normalizing Flows** would let the VAE map simple base distributions to richer non-linear manifolds — a closer match to true macroeconomic phenomena. Other directions include:

- Cross-asset transfer (equities → futures, FX, crypto).
- Regime-conditional decoders for explicit bull/bear separation.
- Replacing the InfoNCE positive-pair construction with **time-shifted views** to capture short-term regime persistence.

---

## References

1. Duan, Y., Wang, L., Zhang, Q., & Li, J. (2022). *FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns.* AAAI 36, 4468–4476.
2. Gu, S., Kelly, B., & Xiu, D. (2019). *Autoencoder Asset Pricing Models.* SSRN.
3. Kelly, B., Pruitt, S., & Su, Y. (2017). *Instrumented Principal Component Analysis.* SSRN.

---

## Author

**Jitendra Padmanabhuni** — Department of Statistics and Data Science, Yale University
`jitendra.padmanabhuni@yale.edu`
