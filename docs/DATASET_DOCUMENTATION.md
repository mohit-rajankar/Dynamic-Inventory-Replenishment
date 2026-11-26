# Dataset Documentation

## 1. Overview & Rationale for Synthetic Generation

In real-world inventory management, historical point-of-sale (POS) and Enterprise Resource Planning (ERP) datasets are generally:
- Proprietary and confidential to commercial retailers.
- Lacking supplier lead time parameters, holding costs, or stockout penalty costs.
- Marred by irregular recording gaps, inventory shrinkage, and missing fields.

To evaluate dynamic machine-learning-driven replenishment with full traceability and zero data leakage, a synthetic dataset generator was engineered in Python (`src/generate_data.py`).

The generator adheres to empirical microeconomic principles:
- **Price elasticity**: Price reductions stimulate consumption.
- **Promotional lift**: Active marketing campaigns generate substantial demand spikes.
- **Weekly seasonality**: Weekend grocery and snack consumption increases significantly.
- **Annual seasonality**: Sinusoidal demand patterns across months.
- **Supplier physics**: Dedicated lead times, holding costs, and purchasing costs per SKU.

---

## 2. Dataset Structure

- **Total Records**: 800 observations.
- **Unique Products**: 4 SKUs.
- **Records per Product**: 200 consecutive calendar days.
- **Date Range**: January 1, 2025 to July 19, 2025.
- **File Location**: `data/inventory_data.csv`.
- **Reproducibility**: Governed by fixed random seed `RANDOM_SEED = 42`.

---

## 3. Data Dictionary

| Column Name | Data Type | Units / Range | Description |
|---|---|---|---|
| `date` | String (YYYY-MM-DD) | 2025-01-01 to 2025-07-19 | Observation calendar date |
| `product_id` | String | P001 to P004 | Unique product SKU identifier |
| `product_name` | String | Categorical | Commercial product trade name |
| `category` | String | Grocery, Personal Care, Food | High-level retail department |
| `price` | Float | $2.55 to $25.50 | Daily selling price per unit |
| `promotion` | Integer | 0 or 1 | Indicator of active promotional discount |
| `day_of_week` | Integer | 0 (Mon) to 6 (Sun) | Day index of the calendar week |
| `month` | Integer | 1 (Jan) to 7 (Jul) | Month index of the calendar year |
| `demand` | Integer | 19 to 195 units | Actual daily customer units demanded |
| `lead_time` | Integer | 2 to 5 days | Supplier order fulfillment delivery delay |
| `unit_cost` | Float | $1.80 to $18.00 | Wholesale purchasing cost per unit |
| `holding_cost`| Float | $0.04 to $0.20 | Daily warehousing/carrying cost per unit |
| `shortage_cost`| Float | $1.50 to $8.00 | Penalty cost per unit of unsatisfied demand |

---

## 4. Product Profiles

| Product ID | Product Name | Category | Base Demand | Base Price | Lead Time | Unit Cost | Daily Holding Cost | Shortage Cost |
|---|---|---|---|---|---|---|---|---|
| **P001** | Basmati Rice | Grocery | 55 units | $25.00 | 3 days | $18.00 | $0.20 | $8.00 |
| **P002** | Cooking Oil | Grocery | 42 units | $15.00 | 4 days | $10.00 | $0.15 | $6.00 |
| **P003** | Herbal Shampoo | Personal Care | 28 units | $8.00 | 5 days | $4.50 | $0.08 | $4.00 |
| **P004** | Digestive Biscuits | Food | 72 units | $3.00 | 2 days | $1.80 | $0.04 | $1.50 |

---

## 5. Mathematical Simulation of Daily Demand

For product $k$ on day $t$:

1. **Base Demand ($d_{0,k}$)**: Core volume demanded under standard market conditions.
2. **Weekend Boost**:
   $$\text{Mult}_{\text{dow}} = 1.0 + \text{boost}_k \quad \text{if day\_of\_week} \in \{5, 6\}$$
3. **Monthly Seasonality**:
   $$\text{Mult}_{\text{season}} = 1.0 + 0.08 \times \sin\left( \frac{2\pi \times \text{day\_of\_year}}{365.25} \right)$$
4. **Price Elasticity Multiplier**:
   $$\text{Mult}_{\text{price}} = \left( \frac{\text{price}_t}{\text{base\_price}_k} \right)^{\epsilon_k}$$
   Where $\epsilon_k < 0$ is the product-specific price elasticity of demand.
5. **Promotional Lift**:
   $$\text{Mult}_{\text{promo}} = 1.0 + \text{promo\_lift}_k \quad \text{if promotion}_t = 1$$
6. **Stochastic Variation**:
   $$\epsilon_{\text{noise}} \sim \mathcal{N}(0, \sigma_k^2)$$
7. **Observed Demand**:
   $$d_t = \max\left( 0, \left\lfloor d_{0,k} \times \text{Mult}_{\text{dow}} \times \text{Mult}_{\text{season}} \times \text{Mult}_{\text{price}} \times \text{Mult}_{\text{promo}} + \epsilon_{\text{noise}} \right\rceil \right)$$

---

## 6. Assumptions & Limitations

1. **Stationary Lead Time**: Lead times are modeled as deterministic integers (2 to 5 days). In real-world shipping, port congestion and transport delays introduce stochastic lead times.
2. **Independent SKUs**: Demand across the four products is simulated independently without cross-price elasticity (e.g., substitution effects between competing brands).
3. **Uncapped Supplier Capacity**: Suppliers are assumed capable of fulfilling any recommended order quantity without minimum order quantities (MOQ) or batch lot sizes.
