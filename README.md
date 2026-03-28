---
title: Supply Chain OpenEnv
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
---

# 🏭 Supply Chain & Inventory Management OpenEnv

An end-to-end [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant environment that simulates **real-world warehouse management** with supplier negotiation — designed for training and evaluating AI agents on supply chain optimization.

Built with domain expertise from **SNC Inventra**.

---

## 🎯 What Does the Agent Do?

The agent manages a simulated warehouse, making daily decisions about:

| Decision | Description |
|----------|-------------|
| **Reorder Quantities** | How many units of each SKU to order from suppliers |
| **Order Timing** | When to place orders considering lead times |
| **Supplier Negotiation** | Propose price/lead-time terms (Hard mode) |
| **Budget Allocation** | Allocate limited capital across competing SKUs |

The goal: **maximize profit** while maintaining high service levels (low stockouts) and minimizing holding costs.

---

## 📊 Tasks (Easy → Medium → Hard)

### Task 1: Easy — Single SKU Reorder
| Parameter | Value |
|-----------|-------|
| SKUs | 1 |
| Horizon | 90 days |
| Demand | Flat (~20/day ± noise) |
| Lead Time | 3 days (fixed) |
| Target | ≥ 95% service level |

**Grading:** 60% service level + 25% inventory efficiency + 15% profitability

### Task 2: Medium — Multi-SKU Seasonal Management
| Parameter | Value |
|-----------|-------|
| SKUs | 10 |
| Horizon | 180 days |
| Demand | Seasonal (sinusoidal, ±40% swings) |
| Lead Times | 2–7 days (variable per SKU) |
| Target | Balanced service across all SKUs |

**Grading:** 40% aggregate SL + 30% per-SKU balance + 20% cost efficiency + 10% profitability

### Task 3: Hard — Full Supply Chain with Negotiation
| Parameter | Value |
|-----------|-------|
| SKUs | 50 |
| Horizon | 365 days |
| Demand | Correlated + shocks |
| Lead Times | 3–14 days |
| Negotiation | Multi-round price/lead negotiation |
| Constraints | Supplier capacity limits, budget |

**Grading:** 35% service level + 25% profit vs baseline + 20% negotiation effectiveness + 10% stockout rate + 10% budget management

---

## 🔌 API — OpenEnv `step()` / `reset()` / `state()`

### Observation Space

```json
{
  "day": 15,
  "horizon": 90,
  "budget": 8542.30,
  "num_skus": 1,
  "service_level": 0.9733,
  "cumulative_reward": 1205.50,
  "reward_breakdown": {
    "revenue": 440.0,
    "holding_cost": 15.0,
    "stockout_penalty": 0.0,
    "order_cost": 200.0,
    "negotiation_savings": 0.0
  },
  "skus": [
    {
      "sku_id": "SKU_001",
      "current_inventory": 42,
      "in_transit_qty": 50,
      "pending_order_days": [2],
      "unit_cost": 10.0,
      "sell_price": 22.0,
      "holding_cost_per_unit_per_day": 0.5,
      "stockout_penalty_per_unit": 8.0,
      "lead_time_days": 3,
      "demand_forecast_7d": [19, 21, 18, 22, 20, 19, 21],
      "avg_daily_demand": 20.13,
      "negotiation_available": false
    }
  ]
}
```

### Action Space

```json
{
  "reorders": [
    {"sku_id": "SKU_001", "quantity": 50}
  ],
  "negotiations": [
    {
      "sku_id": "SKU_H001",
      "proposed_price": 8.50,
      "proposed_lead_time_days": 5,
      "proposed_batch_size": 200
    }
  ]
}
```

### Reward Function (Dense, Per-Step)

```
R(t) = Revenue(t)
     − Holding_Cost(t)
     − Stockout_Penalty(t)
     − Order_Cost(t)
     + Negotiation_Savings(t)
```

Every step produces a non-zero reward signal. No binary end-of-episode scoring.

---

## 🚀 Quick Start

### Install

```bash
pip install -e .
# Or for baseline evaluation:
pip install -e ".[baseline]"
```

### Run the Server

```bash
uvicorn supply_chain_env.server.app:app --host 0.0.0.0 --port 8000
```

### Use the Client

```python
from supply_chain_env import SupplyChainEnv

with SupplyChainEnv(base_url="http://localhost:8000") as env:
    obs = env.reset(task_id="easy")

    while not obs["done"]:
        # Your agent logic here
        action = {"reorders": [{"sku_id": "SKU_001", "quantity": 30}]}
        result = env.step(action)
        obs = result["observation"]

    # Grade the episode
    score = env.grade()
    print(f"Final score: {score['score']:.4f}")
```

### Run Baseline Evaluation

```bash
# Heuristic only (no API key needed):
python -m supply_chain_env.baseline.run_baseline --no-openai

# Full baseline with GPT-4o:
export OPENAI_API_KEY="sk-..."
python -m supply_chain_env.baseline.run_baseline
```

---

## 🐳 Docker

```bash
# Build
docker build -t supply-chain-env .

# Run
docker run -p 8000:8000 supply-chain-env
```

### Deploy to Hugging Face Spaces

```bash
pip install openenv-core
openenv push --repo-id your-username/supply-chain-env
```

---

## 📁 Project Structure

```
supply_chain_env/
├── __init__.py                     # Package exports
├── models.py                      # Action, Observation, State dataclasses
├── client.py                      # Sync + Async clients
├── demand_simulator.py            # Flat / Seasonal / Correlated demand
├── supplier_agent.py              # Multi-round negotiation sub-MDP
├── graders.py                     # Deterministic 0.0–1.0 graders
├── tasks/
│   └── __init__.py                # Easy / Medium / Hard configs
├── server/
│   ├── __init__.py
│   ├── app.py                     # FastAPI + WebSocket server
│   ├── supply_chain_environment.py # Core environment logic
│   └── requirements.txt           # Server dependencies
└── baseline/
    ├── __init__.py
    └── run_baseline.py            # GPT-4o + heuristic baselines
```

---

## 📈 Expected Baseline Scores

| Task | Heuristic (Optimized) | GPT-4o (Evaluated) |
|------|-----------------------|--------------------|
| Easy | **1.0000** | ~0.90 |
| Medium | **0.8712** | ~0.75 |
| Hard | **0.9045** | ~0.65 |

---

## License

BSD-3-Clause
