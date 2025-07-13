import pandas as pd
import numpy as np

def generate_historical_deals(
    modules, tiers, num_deals=1000,
    budget_mu=np.log(150e3), budget_sigma=0.8,
    budget_min=50e3, budget_max=500e3
):
    np.random.seed(42)

    module_params = {
        module: {
            tier: round((np.random.uniform(5, 30) * 1000) * np.random.uniform(1.1, 1.5), 2)
            for tier in tiers
        }
        for module in modules
    }

    deals = []
    for deal_id in range(1, num_deals + 1):
        budget = np.random.lognormal(mean=budget_mu, sigma=budget_sigma)
        budget = np.clip(budget, budget_min, budget_max)
        risk_score = round(np.random.beta(a=2, b=5), 3)

        expected_mod_count = 2 + (budget - budget_min) / (budget_max - budget_min) * 4
        num_modules = np.random.poisson(lam=expected_mod_count / 2) + 1

        selected = np.random.choice(modules, size=min(num_modules, len(modules)), replace=False)

        deal_flags = {f"{m}_{t}": 0 for m in modules for t in tiers}
        total_price = 0.0
        for m in selected:
            base_probs = np.array([0.6, 0.3, 0.1])
            shift = (budget - budget_min) / (budget_max - budget_min) * np.array([-0.2, 0.1, 0.1])
            probs = base_probs + shift
            probs = np.clip(probs, 0, 1)
            probs = probs / probs.sum()
            tier = np.random.choice(tiers, p=probs)

            deal_flags[f"{m}_{tier}"] = 1
            total_price += module_params[m][tier]

        total_price *= np.random.uniform(0.9, 1.1)
        total_price = min(total_price, budget)

        deal = {
            'Deal_ID': deal_id,
            'Budget': round(budget, 2),
            'Risk_Score': risk_score,
            'Deal_Price': round(total_price, 2),
        }
        deal.update(deal_flags)
        deals.append(deal)

    return pd.DataFrame(deals)

def main():
    modules = [
        'Platform', 'ContentA', 'ContentB', 'KYC',
        'Payments', 'Risk', 'Affiliate', 'Analytics'
    ]
    tiers = ['Basic', 'Professional', 'Enterprise']

    df = generate_historical_deals(modules, tiers, num_deals=1000)

    output_path = 'historical_deals_vc_style.xlsx'
    df.to_excel(output_path, sheet_name='Historical_Deals', index=False)
    print(f"Generated {len(df)} deals and saved to {output_path}\n")
    print("Sample of first 10 rows:")
    print(df.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
