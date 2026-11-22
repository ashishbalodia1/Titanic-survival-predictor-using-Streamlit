import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import shap

# Configure page settings
st.set_page_config(
    page_title="Titanic Survival Predictor & Explainer",
    page_icon="🚢",
    layout="centered"
)

# Feature list preserving exact training column order
FEATURES = ["pclass", "sex", "age", "fare"]

@st.cache_resource
def load_model(model_filename="model.joblib"):
    """
    Loads and caches the trained machine learning model from disk.
    Cached so it is only loaded once per session rather than on every click.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, model_filename),
        os.path.join(os.getcwd(), model_filename),
        model_filename
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"Model file '{model_filename}' not found.")

@st.cache_resource
def get_explainer(_model):
    """
    Creates and caches the SHAP TreeExplainer for fast real-time explanation.
    """
    return shap.TreeExplainer(_model)

# Header & Introduction
st.title("🚢 Titanic Survival Predictor")
st.write("An interactive Machine Learning web app powered by a tuned Decision Tree and **SHAP Explainable AI** to provide transparent, interpretable predictions.")

# Model loading with graceful error handling
try:
    model = load_model()
    explainer = get_explainer(model)
except FileNotFoundError:
    st.error("The trained model file ('model.joblib') is missing. Please run the training notebook and save 'model.joblib' first.")
    st.stop()
except Exception as e:
    st.error(f"Unable to load the model file. (Error: {e})")
    st.stop()

st.divider()

# Input widgets with sensible defaults and limits
st.subheader("1. Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass_selection = st.selectbox(
        "Ticket Class",
        options=[1, 2, 3],
        index=2,
        format_func=lambda x: f"{x}st Class (Upper)" if x == 1 else f"{x}nd Class (Middle)" if x == 2 else f"{x}rd Class (Lower)",
        help="Socio-economic tier of ticket"
    )
    
    sex_selection = st.radio(
        "Sex",
        options=["female", "male"],
        horizontal=True,
        help="Gender of the passenger"
    )

with col2:
    age_selection = st.slider(
        "Age",
        min_value=1,
        max_value=80,
        value=30,
        step=1,
        help="Passenger age in years"
    )
    
    fare_selection = st.slider(
        "Fare Paid (£)",
        min_value=0,
        max_value=500,
        value=32,
        step=1,
        help="Ticket fare in British pounds (1912 prices)"
    )

st.divider()

# Prediction logic on button click
if st.button("Predict Survival Chance & Explain Reason", type="primary", use_container_width=True):
    # Validate inputs
    if age_selection < 0 or fare_selection < 0:
        st.warning("Please enter valid, non-negative values for age and fare.")
        st.stop()

    try:
        # Build one-row DataFrame in the exact column order used during training
        encoded_sex = 1 if sex_selection == "female" else 0
        row = pd.DataFrame(
            [[pclass_selection, encoded_sex, age_selection, fare_selection]],
            columns=FEATURES
        )

        # Predict probability of class 1 (survived)
        probabilities = model.predict_proba(row)[0]
        survival_chance = probabilities[1]

        # Display result with st.metric
        st.subheader("2. Prediction Result")
        st.metric(
            label="Estimated Chance of Survival",
            value=f"{survival_chance:.0%}",
            delta="Likely to Survive" if survival_chance >= 0.5 else "Unlikely to Survive",
            delta_color="normal" if survival_chance >= 0.5 else "inverse"
        )

        if survival_chance >= 0.5:
            st.success(f"The model estimates a **{survival_chance:.0%}** probability of survival for this passenger profile.")
        else:
            st.info(f"The model estimates a **{survival_chance:.0%}** probability of survival for this passenger profile.")

        # --- SHAP EXPLAINABLE AI SECTION ---
        st.divider()
        st.subheader("3. Why Did the Model Predict This? (SHAP Explainable AI)")
        st.write("SHAP (SHapley Additive exPlanations) breaks down the contribution of each passenger attribute towards the final survival probability.")

        # Compute SHAP values for this instance
        shap_values = explainer(row)
        
        # Extract values for Class 1 (Survived)
        if len(shap_values.shape) == 3:
            feature_contributions = shap_values.values[0, :, 1]
            base_value = explainer.expected_value[1]
        else:
            feature_contributions = shap_values.values[0]
            base_value = explainer.expected_value

        # Baseline info
        st.markdown(
            f"**Baseline Dataset Survival Rate:** `{base_value:.1%}`  \n"
            f"The model starts at this average baseline and adjusts the probability up or down based on the passenger's attributes:"
        )

        # Plot SHAP contributions using matplotlib
        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
        readable_names = [
            f"Class ({pclass_selection})",
            f"Sex ({sex_selection})",
            f"Age ({age_selection}y)",
            f"Fare (£{fare_selection})"
        ]
        
        colors = ['#22c55e' if val >= 0 else '#ef4444' for val in feature_contributions]
        bars = ax.barh(readable_names, feature_contributions, color=colors, edgecolor='black', linewidth=0.8)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

        # Add data labels to bars
        for bar, val in zip(bars, feature_contributions):
            w = bar.get_width()
            offset = 0.015 if w >= 0 else -0.015
            ha = 'left' if w >= 0 else 'right'
            ax.text(w + offset, bar.get_y() + bar.get_height()/2, f"{val:+.1%}", 
                    va='center', ha=ha, fontweight='bold', fontsize=10)

        # Styling
        ax.set_title("SHAP Feature Contributions to Survival Probability", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Impact on Survival Chance (Green = Positive, Red = Negative)", fontsize=10)
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        
        # Dynamic xlim
        max_abs = max(abs(feature_contributions)) + 0.1
        ax.set_xlim(-max_abs, max_abs)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Human-understandable breakdown of reasons
        st.markdown("#### Detailed Reasoning:")
        
        reasons = []
        for name, col, input_val, contrib in zip(readable_names, FEATURES, [pclass_selection, sex_selection, age_selection, fare_selection], feature_contributions):
            direction = "increased" if contrib >= 0 else "decreased"
            icon = "🟢" if contrib >= 0 else "🔴"
            
            if col == "sex":
                context = 'Women had strict boarding priority under the historical "women and children first" maritime evacuation protocol.' if input_val == "female" else 'Men had substantially lower priority for lifeboats under historical evacuation protocol.'
            elif col == "pclass":
                context = '1st class passengers were located on upper decks with direct access to lifeboats.' if input_val == 1 else ('2nd class had moderate access to boat decks.' if input_val == 2 else '3rd class cabins were deep below deck with delayed access to lifeboats.')
            elif col == "age":
                context = 'Children had priority access to lifeboats.' if input_val < 16 else ('Young adults and middle-aged passengers had standard adult priority.' if input_val <= 50 else 'Older passengers faced physical hurdles during the night evacuation.')
            elif col == "fare":
                context = 'Higher fares correlate with upper-deck staterooms and proximity to lifeboat stations.' if input_val > 50 else 'Standard fares reflect typical middle or lower accommodations.'
            
            reasons.append(f"{icon} **{name}:** {direction.capitalize()} survival chance by **{abs(contrib):.1%}**. *{context}*")

        for r in reasons:
            st.markdown(r)

    except Exception as e:
        st.error(f"An error occurred while calculating predictions: {e}")
        st.stop()

st.divider()

# Educational disclaimer caption
st.caption(
    "A learning exercise: This model is a Decision Tree Classifier trained on the historical RMS Titanic passenger dataset. "
    "Predictions and SHAP values illustrate demographic patterns from the 1912 disaster and are strictly intended for educational exploration, not advice."
)
