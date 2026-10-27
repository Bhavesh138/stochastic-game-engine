# Stochastic Game Theory & Markovian Decision Engine

A quantitative simulation and analytical dynamic programming framework for non-stationary stochastic games with state-dependent payoff matrices and adaptive counter-strategies.

## Mathematical Formulation

### 1. State-Dependent Payoff Matrix
The transition probabilities for round outcomes $\mathcal{S} \in \{\text{Win}, \text{Draw}, \text{Loss}\}$ depend dynamically on the cumulative points scored by agents $(n_A, n_B)$:

$$P(\text{Attack}, \text{Attack}) = \left[ \frac{n_B}{n_A + n_B}, 0, \frac{n_A}{n_A + n_B} \right]$$

### 2. Analytical Dynamic Programming
Exact state transition probabilities $P(i, j)$ (Alice winning $i$ games and Bob winning $j$ games) are computed over $\mathbb{Z}/(10^9+7)\mathbb{Z}$ via the recurrence relation:

$$P(i, j) = \frac{j \cdot P(i-1, j) + i \cdot P(i, j-1)}{i + j - 1} \pmod M$$

Closed-form expectations $E\left[\sum_{k=1}^T X_k\right]$ and variances $\text{Var}\left(\sum_{k=1}^T X_k\right)$ are derived over discrete finite horizons.

### 3. Optimal Stopping Time ($\tau$)
The stopping time $\tau$ to achieve $T$ target victories:

$$\tau = \inf \{ n \in \mathbb{N} : W_A(n) = T \}$$

is estimated empirically via vectorized Monte Carlo trials across $10^5$ iterations.

## Usage
Run the unified benchmark suite:
```bash
python3 main.py