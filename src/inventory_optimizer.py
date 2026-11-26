"""
Inventory Optimization Engine for Dynamic Inventory Replenishment System.
Implements classical and dynamic inventory management formulas:
Safety Stock, Reorder Point (ROP), Order-Up-To Level, and Replenishment Recommendations.
"""

import math
from typing import Dict, Any

# Z-score lookup table for common cycle service levels
Z_SCORES = {
    0.90: 1.282,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326,
}


def get_z_score(service_level: float = 0.95) -> float:
    """
    Return the standard normal distribution quantile (Z-score) for a given service level.
    """
    return Z_SCORES.get(service_level, 1.645)


def calculate_safety_stock(
    demand_std: float,
    lead_time: float,
    service_level: float = 0.95
) -> int:
    """
    Safety Stock (SS) protects against demand variability during supplier lead time.
    Formula: SS = round(Z * sigma_demand * sqrt(lead_time))
    """
    z = get_z_score(service_level)
    ss = z * demand_std * math.sqrt(max(1.0, lead_time))
    return int(math.ceil(ss))


def calculate_lead_time_demand(
    daily_demand: float,
    lead_time: float
) -> int:
    """
    Lead Time Demand (LTD) = Expected total customer demand during supplier lead time.
    Formula: LTD = round(Average or Predicted Daily Demand * lead_time)
    """
    return int(round(daily_demand * lead_time))


def calculate_reorder_point(
    lead_time_demand: int,
    safety_stock: int
) -> int:
    """
    Reorder Point (ROP) = Threshold stock level that triggers a replenishment purchase order.
    Formula: ROP = Lead Time Demand + Safety Stock
    """
    return int(lead_time_demand + safety_stock)


def calculate_order_up_to_level(
    daily_demand: float,
    lead_time: float,
    review_period: int,
    safety_stock: int
) -> int:
    """
    Order-Up-To Level (S) = Target inventory capacity to cover lead time and review cycle.
    Formula: S = ceil(Daily Demand * (lead_time + review_period) + safety_stock)
    """
    level = (daily_demand * (lead_time + review_period)) + safety_stock
    return int(math.ceil(level))


def calculate_inventory_position(
    current_stock: int,
    on_order_stock: int = 0,
    backorders: int = 0
) -> int:
    """
    Inventory Position (IP) = Current Physical On-Hand Stock + In-Transit Orders - Unfulfilled Backorders.
    """
    return int(current_stock + on_order_stock - backorders)


def evaluate_replenishment(
    product_id: str,
    product_name: str,
    current_stock: int,
    on_order_stock: int,
    backorders: int,
    predicted_daily_demand: float,
    demand_std: float,
    lead_time: int,
    review_period: int = 7,
    service_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Comprehensive inventory replenishment recommendation engine.
    Computes policy parameters and determines whether to order and how much.
    """
    # 1. Core inventory calculations
    ip = calculate_inventory_position(current_stock, on_order_stock, backorders)
    ss = calculate_safety_stock(demand_std, lead_time, service_level)
    ltd = calculate_lead_time_demand(predicted_daily_demand, lead_time)
    rop = calculate_reorder_point(ltd, ss)
    order_up_to = calculate_order_up_to_level(predicted_daily_demand, lead_time, review_period, ss)

    # 2. Recommended Order Quantity: order enough to reach target up-to level, minimum 0
    if ip <= rop:
        raw_qty = order_up_to - ip
        recommended_qty = int(max(0, raw_qty))
    else:
        recommended_qty = 0

    # 3. Decision classification and human-readable explanation
    if current_stock <= 0:
        status = "Reorder Now"
        urgency = "Critical"
        explanation = (
            f"STOCKOUT ALERT: Physical on-hand stock is exhausted ({current_stock} units). "
            f"Recommended emergency replenishment of {recommended_qty} units immediately."
        )
    elif ip <= ss:
        status = "Reorder Now"
        urgency = "Critical"
        explanation = (
            f"CRITICAL SHORTAGE: Inventory position ({ip} units) has dropped below safety stock ({ss} units). "
            f"High risk of stockout. Order {recommended_qty} units now."
        )
    elif ip <= rop:
        status = "Reorder Now"
        urgency = "High"
        explanation = (
            f"REORDER POINT REACHED: Inventory position ({ip} units) is at or below ROP ({rop} units). "
            f"Order {recommended_qty} units to restore inventory to order-up-to level ({order_up_to} units)."
        )
    elif ip <= int(rop * 1.15):
        status = "Reorder Soon"
        urgency = "Moderate"
        explanation = (
            f"APPROACHING ROP: Inventory position ({ip} units) is within 15% of ROP ({rop} units). "
            f"Monitor closely; replenishment will likely be needed within the next review cycle."
        )
    elif ip > int(order_up_to * 1.30):
        status = "Overstocked"
        urgency = "Excess"
        explanation = (
            f"OVERSTOCKING DETECTED: Inventory position ({ip} units) exceeds target order-up-to level "
            f"({order_up_to} units) by more than 30%. Incurring elevated holding costs; do not place new orders."
        )
    else:
        status = "Stock Sufficient"
        urgency = "Normal"
        explanation = (
            f"HEALTHY STOCK: Inventory position ({ip} units) is comfortably above ROP ({rop} units). "
            f"No replenishment action required at this time."
        )

    return {
        "product_id": product_id,
        "product_name": product_name,
        "current_stock": current_stock,
        "on_order_stock": on_order_stock,
        "backorders": backorders,
        "inventory_position": ip,
        "predicted_daily_demand": round(predicted_daily_demand, 2),
        "lead_time": lead_time,
        "lead_time_demand": ltd,
        "safety_stock": ss,
        "reorder_point": rop,
        "order_up_to_level": order_up_to,
        "recommended_reorder_quantity": recommended_qty,
        "status": status,
        "urgency": urgency,
        "explanation": explanation,
    }


if __name__ == "__main__":
    # Test with sample scenarios
    print("Testing Inventory Optimizer Logic...")

    # Case 1: Low Stock (Stockout Risk)
    res1 = evaluate_replenishment(
        product_id="P001",
        product_name="Basmati Rice",
        current_stock=20,
        on_order_stock=0,
        backorders=0,
        predicted_daily_demand=60.0,
        demand_std=5.0,
        lead_time=3,
        review_period=7,
    )
    print("\nCase 1 (Low Stock):", res1["status"], f"(Order: {res1['recommended_reorder_quantity']})")
    print("Explanation:", res1["explanation"])

    # Case 2: Healthy Stock
    res2 = evaluate_replenishment(
        product_id="P001",
        product_name="Basmati Rice",
        current_stock=400,
        on_order_stock=0,
        backorders=0,
        predicted_daily_demand=60.0,
        demand_std=5.0,
        lead_time=3,
        review_period=7,
    )
    print("\nCase 2 (Sufficient Stock):", res2["status"], f"(Order: {res2['recommended_reorder_quantity']})")
    print("Explanation:", res2["explanation"])

    # Case 3: Overstocked
    res3 = evaluate_replenishment(
        product_id="P001",
        product_name="Basmati Rice",
        current_stock=950,
        on_order_stock=0,
        backorders=0,
        predicted_daily_demand=60.0,
        demand_std=5.0,
        lead_time=3,
        review_period=7,
    )
    print("\nCase 3 (Overstocked):", res3["status"], f"(Order: {res3['recommended_reorder_quantity']})")
    print("Explanation:", res3["explanation"])
