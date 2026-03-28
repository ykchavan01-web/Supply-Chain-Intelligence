from supply_chain_env import SupplyChainEnv

def test_environment():
    print("Connecting to environment server...")
    with SupplyChainEnv(base_url="http://localhost:8000") as env:
        obs = env.reset(task_id="easy")
        print(f"Initial Observation OK! Starting day: {obs['day']}")

        # Dummy agent running through the simulation
        while not obs["done"]:
            action = {"reorders": [{"sku_id": "SKU_001", "quantity": 30}], "negotiations": []}
            result = env.step(action)
            obs = result["observation"]

        # Grade the episode
        score = env.grade()
        print(f"Episode completed. Final score: {score['score']:.4f}")

if __name__ == "__main__":
    test_environment()
