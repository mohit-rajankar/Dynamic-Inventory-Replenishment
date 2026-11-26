# Presentation & Viva Guide: Dynamic Inventory Replenishment System

This guide is designed to help engineering students present and defend this major project during final year project evaluations, external viva examinations, and project demonstrations.

---

## 1. Project Pitch (30-Second Elevator Pitch)

> "Good morning, respected evaluators. Our project is titled **Dynamic Inventory Replenishment System Using Machine Learning**. 
> Retailers routinely lose millions of dollars either by overstocking—which drives up expensive warehouse holding costs—or by understocking—which results in stockouts and lost customer loyalty. 
> Traditional inventory policies rely on static historical averages that fail to anticipate promotional demand spikes and weekend rushes. 
> Our system integrates a **Random Forest demand forecasting model** with **classical operations research formulas** to dynamically compute Reorder Points (ROP) and safety stock. In our comparative simulation, this eliminated stockouts completely across all managed products while providing store managers with an interactive decision-support dashboard."

---

## 2. Key Slide Structure for Presentation

| Slide # | Slide Title | Core Message / Key Talking Points |
|---|---|---|
| **1** | Title & Team | Project Name, Student Name, Roll No., Department, Supervisor. |
| **2** | Motivation & Problem Statement | The double penalty of retail: Carrying Cost vs. Stockout Cost. Static ROP falls short. |
| **3** | Objectives | Demand forecasting, dynamic ROP, safety stock optimization, simulation, and UI. |
| **4** | Dataset & Simulation Design | 800 records across 4 SKUs; realistic promotion, weekend, and price elasticity effects. |
| **5** | Feature Engineering | Leakage-free lag-1, lag-7, and 7-day rolling averages using past data only. |
| **6** | Machine Learning Architecture | Baseline Linear Regression vs. Random Forest Regressor; chronological 80/20 train-test split. |
| **7** | Model Evaluation Results | Random Forest achieved Test RMSE of 6.89 and $R^2$ of 0.943 (39% error reduction over baseline). |
| **8** | Inventory Optimization Formulas | Lead Time Demand, Safety Stock ($Z=1.645$ for 95% CSL), Reorder Point, and Order-Up-To Level. |
| **9** | Comparative Simulation Findings | Traditional policy incurred 2 stockout days; ML-Driven policy achieved 0 stockout days (100% fulfillment). |
| **10** | Live Application Demonstration | Walkthrough of Streamlit dashboard: Executive view, forecasting, management, and alerts. |
| **11** | Limitations & Future Scope | Stochastic lead times, multi-echelon warehousing, deep learning (LSTM/DeepAR). |
| **12** | Conclusion & Q&A | Summary of engineering achievements and thank you. |

---

## 3. How to Demonstrate the Live Application (Step-by-Step)

When presenting the project on screen:
1. **Show Executive Dashboard (Tab 1)**: Point to the aggregate metrics (Total Physical Stock, Projected Daily Demand, and SKUs requiring replenishment).
2. **Show Demand Forecasting (Tab 2)**: Select "Basmati Rice" (`P001`) and display the actual vs. predicted demand on the test set. Highlight how the model accurately catches peaks during promotions. Then show the multi-day future projection.
3. **Show Interactive Inventory Management (Tab 3)**:
   - Change the current stock of "Herbal Shampoo" (`P003`) from 160 to 20 units.
   - Point out that the calculated Reorder Point (ROP) is 164 units.
4. **Switch to Inventory Recommendations (Tab 4)**:
   - Point out that "Herbal Shampoo" immediately triggers a red **"REORDER NOW"** badge with a specific calculated order quantity and system justification.
5. **Show Model Performance (Tab 5)**: Explain the test-set metrics table and why test RMSE was prioritized over training accuracy.
6. **Show Strategy Simulation (Tab 6)**: Emphasize the honest supply chain finding: ML-based replenishment eliminated all stockout days by anticipating demand spikes at the cost of modest additional buffer stock.

---

## 4. Top 10 Viva Questions and Answers

### Q1: Why did you use Machine Learning instead of standard Economic Order Quantity (EOQ)?
**Answer**: Classical EOQ assumes completely constant, static consumer demand over time and instantaneous or fixed replenishment. In modern retail, demand fluctuates heavily due to promotions, price changes, and day-of-week patterns. Machine learning dynamically forecasts these non-linear demand shifts, allowing Reorder Points and Order-Up-To quantities to adapt in real time.

### Q2: Why did you choose Random Forest over Linear Regression as the primary model?
**Answer**: In our chronological test evaluation, Linear Regression achieved a test RMSE of 11.31 and an $R^2$ of 0.845. In contrast, Random Forest achieved an RMSE of 6.89 and an $R^2$ of 0.943—reducing prediction error by approximately 39%. Random Forest captures complex non-linear interactions (such as the combined effect of a 15% price cut and a promotional campaign) that linear models cannot express without manual feature crosses.

### Q3: How did you ensure there is no data leakage in your time-series features?
**Answer**: First, we used chronological train-test splitting (first 80% of days for training, last 20% for testing) rather than random splitting. Second, all autoregressive features (`lag_1`, `lag_7`, and `rolling_avg_7`) were computed strictly after shifting the target variable by at least 1 day (`demand.shift(1)`). Future observations were never included in rolling statistics.

### Q4: What is the significance of the $Z$-score in the Safety Stock formula?
**Answer**: The $Z$-score represents the standard normal quantile corresponding to the desired Cycle Service Level (CSL). We selected $Z = 1.645$, which corresponds to a 95% service level—meaning there is only a 5% statistical probability of running out of stock during the supplier lead time window.

### Q5: In your simulation, why was the holding cost slightly higher for the ML-based strategy?
**Answer**: This is an honest and realistic supply chain trade-off. The Traditional policy maintained a lower, static inventory level, which caused 2 stockout days (lost sales) during sudden promotional surges. The ML policy predicted upcoming promotional demand and proactively ordered replenishment in advance. While this increased average holding inventory slightly, it completely eliminated stockout days (0 stockouts, 100% customer fulfillment).

### Q6: What is the difference between Physical Stock and Inventory Position?
**Answer**: Physical On-Hand Stock is the physical count of units sitting on warehouse shelves right now. Inventory Position includes Physical Stock plus In-Transit Orders (orders placed with suppliers but not yet delivered) minus Backorders. Evaluating replenishment against Inventory Position prevents placing duplicate orders while waiting for a supplier shipment.

### Q7: Why didn't you use Deep Learning (like LSTM or Transformers)?
**Answer**: With 800 observations across 4 products, tabular tree-based models like Random Forest are computationally efficient, highly robust, and less prone to severe overfitting on small-to-medium tabular datasets. Deep learning sequence models typically require tens of thousands of data points to generalize effectively.

### Q8: What happens if a supplier's lead time increases from 3 days to 7 days?
**Answer**: In our formula, Lead Time Demand increases proportionally ($LTD = \hat{d} \times L$), and Safety Stock increases with the square root of lead time ($SS = Z \times \sigma_d \times \sqrt{L}$). The Reorder Point automatically increases, prompting the system to place replenishment orders much earlier to account for the longer delivery window.

### Q9: Could this system be connected to a real SQL database or ERP?
**Answer**: Yes. In `src/preprocessing.py`, the `load_data()` function can be easily swapped with a SQLAlchemy or database connector reading live POS inventory tables (`SELECT * FROM inventory_transactions`). The rest of the feature engineering, model inference, and optimization engine pipelines remain completely modular.

### Q10: What are the main limitations of the current implementation?
**Answer**: 
1. Lead times are modeled as fixed integers rather than stochastic distributions.
2. Orders assume unconstrained supplier capacity without supplier minimum order quantities (MOQ).
3. The dataset covers 4 independent products without multi-product substitution dynamics.
