import streamlit as st
import pickle
import numpy as np

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Cryptocurrency Price Prediction",
    page_icon="🪙",
    layout="wide"
)

# ---------------- Load Trained Model ----------------
model = pickle.load(open("crypto_model.pkl", "rb"))

# ---------------- Realistic Coin Defaults ----------------
coin_defaults = {
    "Bitcoin": (800_000_000_000, 25_000_000_000),
    "Ethereum": (300_000_000_000, 15_000_000_000),
    "Binance Coin": (90_000_000_000, 3_000_000_000),
    "Solana": (60_000_000_000, 4_000_000_000),
    "Cardano": (40_000_000_000, 2_000_000_000),
    "Ripple (XRP)": (30_000_000_000, 2_000_000_000),
    "Dogecoin": (20_000_000_000, 1_000_000_000),
    "Polkadot": (10_000_000_000, 500_000_000),
    "Tron": (9_000_000_000, 400_000_000),
    "Litecoin": (7_000_000_000, 600_000_000),
    "Avalanche": (15_000_000_000, 800_000_000),
    "Polygon (MATIC)": (12_000_000_000, 700_000_000)
}

# ---------------- Title ----------------
st.title("🪙 Cryptocurrency Price Prediction")
st.write("Predict crypto price based on **Market Capitalization & Trading Volume**")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🔧 Input Parameters")

coin = st.sidebar.selectbox(
    "Select Cryptocurrency",
    list(coin_defaults.keys())
)

default_market_cap, default_volume = coin_defaults[coin]

market_cap = st.sidebar.number_input(
    "Market Capitalization (USD)",
    min_value=0.0,
    value=float(default_market_cap),
    step=100_000_000.0
)

total_volume = st.sidebar.number_input(
    "Total Trading Volume (USD)",
    min_value=0.0,
    value=float(default_volume),
    step=50_000_000.0
)

predict_btn = st.sidebar.button("Predict Price")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📈 Prediction", "📊 Market Insight", "ℹ️ About Model"])

# ---------------- Prediction Logic ----------------
if predict_btn:
    # Apply same preprocessing as training
    X = np.log1p(np.array([[market_cap, total_volume]]))

    # Predict log(price)
    log_price = model.predict(X)

    # Convert back to original scale
    predicted_price = np.expm1(log_price[0])

    # Price range (±5%)
    lower = predicted_price * 0.95
    upper = predicted_price * 1.05

    # Market activity ratio
    ratio = total_volume / market_cap if market_cap > 0 else 0

    # ---------------- Tab 1: Prediction ----------------
    with tab1:
        col1, col2 = st.columns(2)

        col1.metric(
            label=f"Predicted {coin} Price (USD)",
            value=f"${predicted_price:,.2f}"
        )

        col2.metric(
            label="Volume / Market Cap Ratio",
            value=f"{ratio:.2f}"
        )

        st.success(f"Estimated Price Range: $ {lower:,.2f} – $ {upper:,.2f}")

        with st.expander("📋 Input Summary"):
            st.write(f"**Cryptocurrency:** {coin}")
            st.write(f"**Market Capitalization:** $ {market_cap:,.0f}")
            st.write(f"**Total Trading Volume:** $ {total_volume:,.0f}")

    # ---------------- Tab 2: Market Insight ----------------
    with tab2:
        if ratio > 0.5:
            st.success("📈 High trading activity — strong market interest.")
        elif ratio > 0.2:
            st.warning("⚖️ Moderate trading activity — stable market conditions.")
        else:
            st.error("📉 Low trading activity — weak market participation.")

        st.markdown("""
        **Insight Explanation:**
        - Market capitalization reflects the size and valuation of a cryptocurrency.
        - Trading volume indicates liquidity and investor participation.
        - A higher volume-to-market-cap ratio suggests stronger trading momentum.
        """)

    # ---------------- Tab 3: About Model ----------------
    with tab3:
        st.markdown("""
        ### 📌 Model Information
        - **Model Used:** Random Forest Regressor  
        - **Input Features (X):** Market Capitalization, Total Trading Volume  
        - **Target Variable (y):** Current Cryptocurrency Price  
        - **Preprocessing:** Log transformation (`log1p`)  
        - **Inverse Transform:** `expm1` during prediction  
        - **Evaluation Metrics:** R² Score, RMSE  

        ### 📊 Project Summary
        This project evaluates multiple regression models including Linear Regression,
        Polynomial Regression, Decision Tree, Random Forest, and Gradient Boosting.
        Random Forest Regressor achieved the best performance and was selected for deployment.
        """)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("📌 Cryptocurrency Price Prediction by **VISHNU RAJ**")


