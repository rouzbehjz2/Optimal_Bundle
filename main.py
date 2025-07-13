import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pulp

def solve_basic_bundle(beta, cost, B, dependencies, exclusions):
    feats = list(beta.keys())
    x = pulp.LpVariable.dicts('x', feats, cat='Binary')
    prob = pulp.LpProblem('bundle_opt', pulp.LpMaximize)

    # Objective: maximize profit = sum (β - cost) * x
    prob += pulp.lpSum((beta[f] - cost[f]) * x[f] for f in feats)

    # Budget cap on customer’s perceived price: sum β * x ≤ B
    prob += pulp.lpSum(beta[f] * x[f] for f in feats) <= B

    # Dependencies
    for pre, posts in dependencies.items():
        for post in posts:
            prob += pulp.lpSum(x[f"{pre}_{t}"] for t in ['Basic','Professional','Enterprise']) \
                  <= pulp.lpSum(x[f"{post}_{t}"] for t in ['Basic','Professional','Enterprise'])

    # Mutual exclusions
    for a, b in exclusions:
        for t1 in ['Basic','Professional','Enterprise']:
            for t2 in ['Basic','Professional','Enterprise']:
                prob += x[f"{a}_{t1}"] + x[f"{b}_{t2}"] <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [f for f in feats if x[f].value() == 1]
    total_profit = pulp.value(prob.objective)
    return selected, total_profit

def main():
    # 1) Load your deals
    df = pd.read_excel('historical_deals_vc_style.xlsx',
                       sheet_name='Historical_Deals')

    # 2) Features / target
    feature_cols = [c for c in df.columns
                    if c not in ['Deal_ID','Budget','Risk_Score','Deal_Price']]
    X = df[feature_cols]
    y = df['Deal_Price']

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Fit Ridge for WTP (β)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)

    # 5) Report regression metrics
    def eval_split(Xs, ys, name):
        pred = model.predict(Xs)
        print(f"{name:5s} R²={r2_score(ys,pred):.3f}, "
              f"RMSE=€{np.sqrt(mean_squared_error(ys,pred)):,.2f}")
    print("Regression Performance:")
    eval_split(X_train, y_train, "Train")
    eval_split(X_test,  y_test,  "Test")

    # 6) Extract β coefficients
    beta_series = pd.Series(model.coef_, index=feature_cols)
    beta = beta_series.to_dict()

    # Print top-5 β
    top5 = beta_series.sort_values(ascending=False).head(5)
    print("\nTop 5 WTP coefficients (β):")
    for feat, val in top5.items():
        print(f"  {feat:<25s} β = €{val:,.2f}")

    # 7) Randomize “cost” around β
    np.random.seed(123)
    cost = {f: float(beta[f] * np.random.uniform(0.5, 1.5))
            for f in feature_cols}

    # 8) Business rules
    dependencies = {'Platform': ['KYC'], 'Affiliate': ['Analytics']}
    exclusions   = {('ContentA','ContentB')}

    # 9) Solve for various budgets, showing details
    for B in [50_000, 100_000, 150_000, 200_000, 300_000]:
        sel, profit = solve_basic_bundle(beta, cost, B, dependencies, exclusions)
        print(f"\nBudget €{B:,.0f}")
        print("Selected bundle:")
        perceived_total = 0.0
        actual_cost     = 0.0
        for f in sel:
            perceived_total += beta[f]
            actual_cost     += cost[f]
            print(f"  {f:<25s} β=€{beta[f]:,.2f}, cost=€{cost[f]:,.2f}, "
                  f"margin=€{beta[f]-cost[f]:,.2f}")
        print(f"Perceived total price: €{perceived_total:,.2f}")
        print(f"Actual total cost:      €{actual_cost:,.2f}")
        print(f"Total profit:           €{profit:,.2f}")

if __name__ == '__main__':
    main()
