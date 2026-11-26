# Algorithms & Mathematical Formulations

This document provides explanations and mathematical formulas for the algorithms, evaluation metrics, and inventory calculations used in the project.

---

## 1. Machine Learning Algorithms

### A. Linear Regression (Baseline)
Linear Regression models the relationship between the target variable $y$ (daily demand) and feature vector $\mathbf{x} = [x_1, x_2, \dots, x_p]^T$ as a linear combination:

$$\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j x_j$$

- **Optimization Objective**: Ordinary Least Squares (OLS) minimizes the Residual Sum of Squares (RSS):
  $$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2$$
- **Role in Project**: Serves as a transparent, interpretable baseline to establish the performance gain achieved by non-linear machine learning.

### B. Random Forest Regressor (Primary Model)
Random Forest is an ensemble learning method constructed by combining multiple de-correlated decision trees:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$

Where $B$ is the number of trees (e.g., $B = 120$) and $T_b(\mathbf{x})$ is an individual regression tree trained on a bootstrap sample of the training dataset with random feature subspace sampling at each split.

- **Why Random Forest Excels in Retail Forecasting**:
  1. **Non-linear Relationships**: Captures promotional discounts and weekend surges without requiring manual interaction terms.
  2. **Robustness to Outliers**: Extreme daily demand spikes do not skew the entire ensemble.
  3. **Low Overfitting Risk**: Bootstrap aggregation (bagging) significantly reduces variance compared to individual decision trees.

---

## 2. Evaluation Metrics

### A. Mean Absolute Error (MAE)
The average absolute magnitude of prediction errors:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|$$

- **Interpretation**: Expressed in raw demand units. An MAE of 4.62 means predictions deviate from actual daily sales by approximately 4.6 units on average.

### B. Root Mean Squared Error (RMSE)
The square root of the mean squared differences between observed and predicted values:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}$$

- **Interpretation**: Penalizes large forecasting errors more heavily than MAE. Crucial in supply chain forecasting because large demand errors cause sudden stockouts.

### C. Coefficient of Determination ($R^2$ Score)
Measures the proportion of variance in demand explained by the model:

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

- **Interpretation**: Ranges up to 1.0 (perfect prediction). An $R^2$ of 0.943 indicates that 94.3% of the variability in daily customer demand is explained by the model features.

---

## 3. Inventory Optimization Formulations

### A. Lead Time Demand (LTD)
The expected customer demand during supplier replenishment delay:

$$\text{LTD} = \hat{d} \times L$$

- $\hat{d}$: Average daily demand (Traditional) or ML-predicted daily demand (Dynamic).
- $L$: Supplier lead time in days.
- *Example*: If predicted demand is 50 units/day and lead time is 4 days, $\text{LTD} = 50 \times 4 = 200 \text{ units}$.

### B. Safety Stock (SS)
Buffer inventory held to mitigate demand fluctuations during the supplier lead time window:

$$\text{SS} = \lceil Z \times \sigma_d \times \sqrt{L} \rceil$$

- $Z$: Normal distribution quantile corresponding to the target Cycle Service Level (CSL). For a 95% service level, $Z = 1.645$.
- $\sigma_d$: Standard deviation of daily demand (or standard deviation of model prediction errors).
- $L$: Supplier lead time in days.
- *Example*: For $\sigma_d = 10$, $L = 4$, and $Z = 1.645$:
  $$\text{SS} = \lceil 1.645 \times 10 \times \sqrt{4} \rceil = \lceil 32.9 \rceil = 33 \text{ units}$$

### C. Reorder Point (ROP)
The physical threshold that signals when a new purchase order must be issued:

$$\text{ROP} = \text{LTD} + \text{SS} = (\hat{d} \times L) + \text{SS}$$

- *Example*: Using the values above, $\text{ROP} = 200 + 33 = 233 \text{ units}$.
- When inventory drops to or below 233 units, a replenishment order is triggered immediately.

### D. Inventory Position (IP)
Total inventory accountability representing current assets minus liabilities:

$$\text{IP} = \text{On-Hand Stock} + \text{On-Order Stock} - \text{Backorders}$$

- Including in-transit orders prevents "double ordering" while waiting for supplier deliveries.

### E. Order-Up-To Level ($S$) & Recommended Order Quantity ($Q^*$)
Under a periodic review policy with review cycle $R$ (e.g., $R = 7$ days):

$$S = \lceil \hat{d} \times (L + R) + \text{SS} \rceil$$

When $\text{IP} \le \text{ROP}$, the recommended replenishment quantity is:

$$Q^* = \max(0, S - \text{IP})$$

- If $\text{IP} > \text{ROP}$, no order is placed ($Q^* = 0$).
- *Example*: For $\hat{d} = 50$, $L = 4$, $R = 7$, $\text{SS} = 33$, and current $\text{IP} = 150$:
  $$S = 50 \times (4 + 7) + 33 = 550 + 33 = 583 \text{ units}$$
  $$Q^* = 583 - 150 = 433 \text{ units}$$

---

## 4. Cost Accounting Formulations

During each simulated day $t$:

1. **Holding Cost**:
   $$\text{Cost}_{\text{holding}} = I_t \times h$$
   Where $I_t$ is ending on-hand inventory and $h$ is the daily holding cost per unit.

2. **Shortage Cost**:
   $$\text{Cost}_{\text{shortage}} = \text{Shortage}_t \times c_s$$
   Where $\text{Shortage}_t = \max(0, d_t - I_{t-1})$ and $c_s$ is the unit penalty cost of lost sales.

3. **Purchasing Cost**:
   $$\text{Cost}_{\text{purchase}} = Q_t \times c_u$$
   Where $Q_t$ is the units ordered and $c_u$ is unit purchasing cost.

4. **Total Inventory Cost**:
   $$\text{Total Cost} = \sum_{t=1}^{T} \left( \text{Cost}_{\text{holding}} + \text{Cost}_{\text{shortage}} + \text{Cost}_{\text{purchase}} \right)$$
