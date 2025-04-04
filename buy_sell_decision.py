import numpy as np
import streamlit as st
from buy_sell import process_models

def weighted_decision(data):
    st.markdown(""" 
    Considering the predictions from multiple models, we use a **weighted decision-making approach** to determine whether to **BUY 📈, SELL 📉, or HOLD ⏳** a stock.  

    - Models with **lower RMSE (higher accuracy)** are given more weight in the final decision.  
    - Each model's recommendation is converted into a **numerical score**, and a **weighted sum** is calculated.  
    - If the score leans **strongly positive**, it suggests a **BUY**; if **strongly negative**, a **SELL**; otherwise, a **HOLD** decision is made.  

    This ensures a **balanced, data-driven investment strategy** by considering **multiple perspectives**.
    """)
    decisions = process_models(data)

    # Compute weights (lower RMSE gets higher weight)
    rmse_values = np.array([result["RMSE"] for result in decisions.values()])
    inverse_rmse = 1 / (rmse_values + 1e-5)  # Avoid division by zero
    weights = inverse_rmse / np.sum(inverse_rmse)  # Normalize

    # Map decisions to numerical values
    decision_mapping = {"BUY 📈": 1, "SELL 📉": -1, "HOLD ⏳": 0}
    weighted_sum = 0

    for i, (model, result) in enumerate(decisions.items()):
        decision_value = decision_mapping[result["Decision"]]
        weighted_sum += decision_value * weights[i]  # Weighted score

    # Determine final decision
    if weighted_sum > 0.3:  # Threshold for strong BUY
        final_decision = "BUY 📈"
    elif weighted_sum < -0.3:  # Threshold for strong SELL
        final_decision = "SELL 📉"
    else:
        final_decision = "HOLD ⏳"

    return final_decision, weighted_sum

# # Streamlit display
# final_decision, score = weighted_decision(data)
# st.markdown("`Final Weighted Decision`")
# st.metric(label="Overall Decision", value=final_decision)
# st.write(f"Weighted Score: `{score:.4f}`")
