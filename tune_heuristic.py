import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supply_chain_env.server.supply_chain_environment import SupplyChainEnvironment
from supply_chain_env.models import SupplyChainAction
from supply_chain_env.graders import GRADER_REGISTRY

def run_test(task_id: str, sf_base: float, sf_mult: float, neg_discount: float, min_batch: int):
    env = SupplyChainEnvironment(task_id=task_id)
    obs = env.reset(task_id=task_id)

    while not obs.done:
        reorders = []
        negotiations = []
        for sku_data in obs.skus:
            sid = sku_data["sku_id"]
            inv = sku_data["current_inventory"]
            in_transit = sku_data["in_transit_qty"]
            lead = sku_data["lead_time_days"]
            forecast = sku_data.get("demand_forecast_7d", [])
            avg_demand = sku_data.get("avg_daily_demand", 10)

            projected_demand = sum(forecast[:lead]) if forecast else 0
            if lead > len(forecast):
                projected_demand += (lead - len(forecast)) * avg_demand
                
            holding_cost = sku_data.get("holding_cost_per_unit_per_day", 1.0)
            penalty = sku_data.get("stockout_penalty_per_unit", 10.0)
            cost_ratio = penalty / (holding_cost * max(1, lead))
            
            safety_factor = sf_base + min(1.0, cost_ratio * sf_mult)
            reorder_point = projected_demand * safety_factor
            available = inv + in_transit

            order_up_to = projected_demand * (safety_factor + 0.5)
            
            order_qty = 0
            if available < reorder_point:
                order_qty = int(order_up_to - available)
                order_qty = max(min_batch, order_qty)
                reorders.append({"sku_id": sid, "quantity": order_qty})
                
            if sku_data.get("negotiation_available") and order_qty >= 50:
                base_price = sku_data.get("supplier_base_price", sku_data["unit_cost"])
                proposed = round(base_price * (1.0 - neg_discount), 2)
                negotiations.append({
                    "sku_id": sid,
                    "proposed_price": proposed,
                    "proposed_lead_time_days": lead,
                    "proposed_batch_size": order_qty
                })

        action = SupplyChainAction(reorders=reorders, negotiations=negotiations)
        obs = env.step(action)

    grader = GRADER_REGISTRY[task_id]()
    return grader.score(env.get_trajectory(), env.config)

for task in ['medium', 'hard']:
    # Test combinations
    for sf_base in [1.0, 1.2]:
        for sf_mult in [0.05, 0.1, 0.2]:
            for nd in [0.1, 0.15, 0.2]:
                score = run_test(task, sf_base, sf_mult, nd, 10)
                print(f"{task:6s} | base={sf_base:.1f} mult={sf_mult:.2f} neg={nd:.2f} => {score:.4f}")
