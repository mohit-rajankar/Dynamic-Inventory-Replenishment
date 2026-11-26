"""
Automated tests for inventory optimization calculations, safety stock, ROP, and recommendations.
"""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.inventory_optimizer import (
    calculate_safety_stock,
    calculate_lead_time_demand,
    calculate_reorder_point,
    calculate_order_up_to_level,
    calculate_inventory_position,
    evaluate_replenishment,
)


def test_inventory_math_formulas():
    # Example values:
    # Daily demand = 50, Lead time = 4 days, std = 10, service level = 0.95 (Z = 1.645)
    # LTD = 50 * 4 = 200
    # SS = ceil(1.645 * 10 * sqrt(4)) = ceil(1.645 * 10 * 2) = ceil(32.9) = 33
    # ROP = 200 + 33 = 233
    # Order-up-to (Review period = 7) = ceil(50 * (4 + 7) + 33) = 50 * 11 + 33 = 583
    ltd = calculate_lead_time_demand(50.0, 4.0)
    assert ltd == 200

    ss = calculate_safety_stock(10.0, 4.0, service_level=0.95)
    assert ss == 33

    rop = calculate_reorder_point(ltd, ss)
    assert rop == 233

    order_up_to = calculate_order_up_to_level(50.0, 4.0, review_period=7, safety_stock=ss)
    assert order_up_to == 583

    ip = calculate_inventory_position(current_stock=100, on_order_stock=50, backorders=10)
    assert ip == 140


def test_replenishment_low_stock():
    # Low stock case: current stock = 30, ROP = 233 -> must trigger Reorder Now
    res = evaluate_replenishment(
        product_id="P001",
        product_name="Test Product",
        current_stock=30,
        on_order_stock=0,
        backorders=0,
        predicted_daily_demand=50.0,
        demand_std=10.0,
        lead_time=4,
        review_period=7,
        service_level=0.95,
    )
    assert res["status"] == "Reorder Now"
    assert res["recommended_reorder_quantity"] > 0
    assert res["recommended_reorder_quantity"] == res["order_up_to_level"] - 30


def test_replenishment_sufficient_stock():
    # Sufficient stock case: stock = 400 > ROP (233)
    res = evaluate_replenishment(
        product_id="P001",
        product_name="Test Product",
        current_stock=400,
        on_order_stock=0,
        backorders=0,
        predicted_daily_demand=50.0,
        demand_std=10.0,
        lead_time=4,
        review_period=7,
        service_level=0.95,
    )
    assert res["status"] == "Stock Sufficient"
    assert res["recommended_reorder_quantity"] == 0


def test_replenishment_overstocked():
    # Overstocked case: stock = 900 >> Order up to level (583)
    res = evaluate_replenishment(
        product_id="P001",
        product_name="Test Product",
        current_stock=900,
        on_order_stock=0,
        backorders=0,
        predicted_daily_demand=50.0,
        demand_std=10.0,
        lead_time=4,
        review_period=7,
        service_level=0.95,
    )
    assert res["status"] == "Overstocked"
    assert res["recommended_reorder_quantity"] == 0
