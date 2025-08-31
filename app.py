# app.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from pandas.plotting import scatter_matrix

import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Optional libs for XAI
try:
    import shap  # pip install shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ========== App Config & Style ==========
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# Minimal modern styling
st.markdown("""
<style>
/* Global */
.main, .block-container { padding-top: 1.2rem; }
h1,h2,h3 { font-weight: 700; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
section[data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }

/* Card */
.card {
  background: rgb(255 255 255 / 70%);
  border: 1px solid #eee;
  border-radius: 16px;
  padding: 1.0rem 1.2rem;
  box-shadow: 0 2px 12px rgba(0,0,0,0.04);
  backdrop-filter: saturate(180%) blur(6px);
  margin-bottom: 1rem;
}

/* Tabs spacing */
div[data-baseweb="tab-list"] { gap: .25rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction — Pro Dashboard")

# ========== Paths / Files ==========
HISTORY_FILE = "prediction_history.csv"

# ========== Caching ==========
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler

@st.cache_data
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_FILE, index=False)

# ========== Load artifacts & state ==========
model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

if "history" not in st.session_state:
    st.session_state["history"] = load_history().to_dict("records")

# ========== Helpers ==========
def build_feature_frame(raw_row):
    """
    Given raw input values (dict), build the full model input dataframe
    with label-encoded gender, one-hot geography, and scaled features.
    Returns: (X_df, X_scaled, feature_names)
    """
    base_df = pd.DataFrame({
        'CreditScore': [raw_row['Credit Score']],
        'Gender': [label_encoder_gender.transform([raw_row['Gender']])[0]],
        'Age': [raw_row['Age']],
        'Tenure': [raw_row['Tenure']],
        'Balance': [raw_row['Balance']],
        'NumOfProducts': [raw_row['Products']],
        'HasCrCard': [raw_row['Has Credit Card']],
        'IsActiveMember': [raw_row['Active Member']],
        'EstimatedSalary': [raw_row['Estimated Salary']],
    })

    geo_encoded = onehot_encoder_geo.transform([[raw_row['Geography']]]).toarray()
    geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
    geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)

    X_df = pd.concat([base_df.reset_index(drop=True), geo_df], axis=1)
    X_scaled = scaler.transform(X_df)
    return X_df, X_scaled, X_df.columns.tolist()

def predict_proba_scaled(X_scaled: np.ndarray) -> float:
    p = model.predict(X_scaled, verbose=0)
    return float(p[0][0])

def add_history_record(rec: dict):
    st.session_state["history"].append(rec)
    save_history(pd.DataFrame(st.session_state["history"]))

def get_history_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state["history"])

def kpi_cards(hist_df: pd.DataFrame):
    left, mid, right = st.columns(3)
    if hist_df.empty:
        with left: st.metric("Total Predictions", 0)
        with mid: st.metric("Avg Churn Prob", "—")
        with right: st.metric("Churn Rate", "—")
        return
    total = len(hist_df)
    avg_prob = hist_df["Churn Probability (%)"].mean()
    churn_rate = (hist_df["Prediction"] == "Churn").mean() * 100
    with left: st.metric("Total Predictions", total)
    with mid: st.metric("Avg Churn Prob", f"{avg_prob:.1f}%")
    with right: st.metric("Churn Rate", f"{churn_rate:.1f}%")

def safe_scatter_matrix(df: pd.DataFrame, features: list):
    if len(df) > 5:
        scatter_matrix(df[features], figsize=(10, 10), diagonal="hist", alpha=0.7)
        plt.suptitle("Scatter Matrix of Features", y=1.02)
        st.pyplot(plt.gcf())
    else:
        st.info("📊 Need at least 6 records to render a scatter matrix.")

def what_if_sensitivity(raw_row: dict, feature_names_display: list = None):
    """
    Simple one-at-a-time perturbation around the current sample.
    For numeric features: +/- 10% (or min step).
    For binary: toggle.
    For Geography/Gender: try other categories, take max delta.
    Returns a DataFrame of |delta prob| by feature (percentage points).
    """
    # Baseline
    X_df, X_scaled, feat_names = build_feature_frame(raw_row)
    base_prob = predict_proba_scaled(X_scaled)

    impacts = {}

    # Numeric & binary in raw_row
    numeric_keys = ["Credit Score", "Age", "Tenure", "Balance", "Products", "Has Credit Card", "Active Member", "Estimated Salary"]

    for k in numeric_keys:
        mod_row = raw_row.copy()
        v = raw_row[k]
        if k in ["Has Credit Card", "Active Member"]:
            mod_row[k] = 1 - int(v)  # toggle
            _, xs, _ = build_feature_frame(mod_row)
            prob_toggle = predict_proba_scaled(xs)
            impacts[k] = abs(prob_toggle - base_prob) * 100.0  # percentage points
        else:
            step = max(1.0, abs(v) * 0.10)
            # down
            mod_row[k] = v - step
            _, xs1, _ = build_feature_frame(mod_row)
            p1 = predict_proba_scaled(xs1)
            # up
            mod_row[k] = v + step
            _, xs2, _ = build_feature_frame(mod_row)
            p2 = predict_proba_scaled(xs2)
            impacts[k] = max(abs(p1 - base_prob), abs(p2 - base_prob)) * 100.0

    # Gender alternative
    try:
        other_gender = [g for g in label_encoder_gender.classes_ if g != raw_row["Gender"]]
        if other_gender:
            mod_row = raw_row.copy()
            mod_row["Gender"] = other_gender[0]
            _, xs, _ = build_feature_frame(mod_row)
            p = predict_proba_scaled(xs)
            impacts["Gender"] = abs(p - base_prob) * 100.0
    except Exception:
        pass

    # Geography alternatives
    try:
        geo_list = list(onehot_encoder_geo.categories_[0])
        deltas = []
        for g in geo_list:
            if g == raw_row["Geography"]: 
                continue
            mod_row = raw_row.copy()
            mod_row["Geography"] = g
            _, xs, _ = build_feature_frame(mod_row)
            p = predict_proba_scaled(xs)
            deltas.append(abs(p - base_prob))
        if deltas:
            impacts["Geography"] = max(deltas) * 100.0
    except Exception:
        pass

    df_imp = pd.DataFrame({
        "Feature": impacts.keys(),
        "Impact (pp)": [round(v, 2) for v in impacts.values()]
    }).sort_values("Impact (pp)", ascending=False)

    if feature_names_display:
        # Optional rename
        pass

    return base_prob * 100.0, df_imp

# ========== Sidebar Navigation ==========
st.sidebar.header("📌 Navigation")
page = st.sidebar.radio(
    "",
    ["🔮 Prediction", "📜 History", "📈 Analytics", "🧠 Explainability", "👥 Segmentation", "💾 Manage"],
    index=0
)

# ========== Prediction ==========
if page == "🔮 Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Input Customer Details")

    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 35)
    balance = st.number_input('💰 Balance', min_value=0.0, value=0.0, step=100.0)
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=1000, value=650, step=1)
    estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, value=50000.0, step=100.0)
    tenure = st.slider('📅 Tenure (Years)', 0, 10, 3)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], index=1)
    is_active_member = st.selectbox('✅ Is Active Member', [0, 1], index=1)

    st.markdown('</div>', unsafe_allow_html=True)

    # Build model input
    raw_input = {
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Credit Score": credit_score,
        "Balance": balance,
        "Estimated Salary": estimated_salary,
        "Tenure": tenure,
        "Products": num_of_products,
        "Has Credit Card": has_cr_card,
        "Active Member": is_active_member,
    }

    X_df, X_scaled, feature_names = build_feature_frame(raw_input)

    # Predict
    prob = predict_proba_scaled(X_scaled)
    pred_label = "Churn" if prob > 0.5 else "Not Churn"

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Churn Probability", f"{prob*100:.2f}%")
    with c2: st.metric("Prediction", pred_label)
    with c3: st.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")

    # Pie
    fig, ax = plt.subplots()
    ax.pie([prob, 1-prob], labels=["Churn", "Not Churn"], autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # Save
    if st.button("💾 Save this prediction", type="primary", use_container_width=True):
        new_record = {
            **raw_input,
            "Churn Probability (%)": round(prob * 100, 2),
            "Prediction": pred_label
        }
        add_history_record(new_record)
        st.success("Saved to history!")

# ========== History (with Filters) ==========
elif page == "📜 History":
    st.subheader("📜 Prediction History")
    hist_df = get_history_df()
    kpi_cards(hist_df)

    if hist_df.empty:
        st.info("No history yet. Make a prediction in the Prediction tab.")
    else:
        with st.expander("🔎 Filters", expanded=True):
            colA, colB, colC = st.columns(3)
            with colA:
                geo_filter = st.multiselect("Geography", sorted(hist_df["Geography"].dropna().unique().tolist()))
                gender_filter = st.multiselect("Gender", sorted(hist_df["Gender"].dropna().unique().tolist()))
            with colB:
                churn_filter = st.multiselect("Prediction", ["Churn", "Not Churn"])
                credit_min, credit_max = st.slider("Credit Score range", 300, 1000, (300, 1000))
            with colC:
                age_min, age_max = st.slider("Age range", int(hist_df["Age"].min()) if not hist_df.empty else 18,
                                             int(hist_df["Age"].max()) if not hist_df.empty else 92,
                                             (int(hist_df["Age"].min()) if not hist_df.empty else 18,
                                              int(hist_df["Age"].max()) if not hist_df.empty else 92))

        fdf = hist_df.copy()
        if geo_filter: fdf = fdf[fdf["Geography"].isin(geo_filter)]
        if gender_filter: fdf = fdf[fdf["Gender"].isin(gender_filter)]
        if churn_filter: fdf = fdf[fdf["Prediction"].isin(churn_filter)]
        fdf = fdf[(fdf["Credit Score"].between(credit_min, credit_max)) & (fdf["Age"].between(age_min, age_max))]

        st.dataframe(fdf.reset_index(drop=True), use_container_width=True)

# ========== Analytics ==========
elif page == "📈 Analytics":
    st.subheader("📈 Analytics")
    hist_df = get_history_df()
    kpi_cards(hist_df)

    if hist_df.empty:
        st.info("No history yet. Make a prediction first.")
    else:
        tab1, tab2, tab3 = st.tabs(["🎯 Scatter", "🔥 Heatmap", "📊 Scatter Matrix"])

        with tab1:
            st.markdown("### 🎯 Age vs Churn Probability")
            fig, ax = plt.subplots()
            colors = ['red' if p == 'Churn' else 'green' for p in hist_df["Prediction"]]
            ax.scatter(hist_df["Age"], hist_df["Churn Probability (%)"], c=colors, alpha=0.75, edgecolors="k")
            ax.set_xlabel("Age")
            ax.set_ylabel("Churn Probability (%)")
            st.pyplot(fig)

        with tab2:
            st.markdown("### 🔥 Feature Correlation Heatmap")
            numeric_cols = ["Credit Score", "Age", "Balance", "Tenure", "Products", "Churn Probability (%)"]
            corr = hist_df[numeric_cols].corr()
            fig, ax = plt.subplots()
            im = ax.imshow(corr, cmap="coolwarm", interpolation="nearest")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)
            fig.colorbar(im)
            st.pyplot(fig)

        with tab3:
            st.markdown("### 📊 Scatter Matrix of Features")
            safe_scatter_matrix(hist_df, ["Credit Score", "Age", "Balance", "Tenure", "Products", "Churn Probability (%)"])

# ========== Explainability ==========
elif page == "🧠 Explainability":
    st.subheader("🧠 Explainability")
    st.caption("Local attributions for the last prediction (or a custom input). If SHAP is not installed, we use What-If sensitivity.")

    hist_df = get_history_df()
    use_last = False
    if not hist_df.empty and st.toggle("Use last saved prediction as baseline", value=True):
        use_last = True

    if use_last:
        last = hist_df.iloc[-1].to_dict()
        raw_input = {
            "Geography": last["Geography"],
            "Gender": last["Gender"],
            "Age": int(last["Age"]),
            "Credit Score": int(last["Credit Score"]),
            "Balance": float(last["Balance"]),
            "Estimated Salary": float(last["Estimated Salary"]),
            "Tenure": int(last["Tenure"]),
            "Products": int(last["Products"]),
            "Has Credit Card": int(last["Has Credit Card"]),
            "Active Member": int(last["Active Member"]),
        }
    else:
        st.info("Enter a custom input in the Prediction tab and save it to use here.")
        st.stop()

    # Try SHAP (KernelExplainer), else fallback
    try:
        X_df, X_scaled, feat_names = build_feature_frame(raw_input)

        if SHAP_AVAILABLE:
            st.markdown("#### 🔍 SHAP (Kernel) — local feature attributions")
            # Background: sample from history if possible
            bg_X = None
            if len(hist_df) >= 20:
                # Build a small background set
                bg_sample = hist_df.sample(n=min(50, len(hist_df)), random_state=7).to_dict("records")
                rows = []
                for r in bg_sample:
                    Xi, _, _ = build_feature_frame({
                        "Geography": r["Geography"],
                        "Gender": r["Gender"],
                        "Age": r["Age"],
                        "Credit Score": r["Credit Score"],
                        "Balance": r["Balance"],
                        "Estimated Salary": r["Estimated Salary"],
                        "Tenure": r["Tenure"],
                        "Products": r["Products"],
                        "Has Credit Card": r["Has Credit Card"],
                        "Active Member": r["Active Member"],
                    })
                    rows.append(Xi.values[0])
                bg_X = np.vstack(rows)
            else:
                # Fallback: small random noise around current sample
                noise = np.random.normal(scale=0.01, size=(25, X_scaled.shape[1]))
                bg_X = X_scaled + noise

            f = lambda x: model.predict(x, verbose=0).reshape(-1, 1)
            explainer = shap.KernelExplainer(f, bg_X)
            shap_values = explainer.shap_values(X_scaled, nsamples="auto")  # for single sample
            sv = shap_values[0].flatten() if isinstance(shap_values, list) else shap_values.flatten()
            # Plot as bar (matplotlib)
            imp_df = pd.DataFrame({"Feature": feat_names, "SHAP value": sv})
            imp_df["Abs"] = imp_df["SHAP value"].abs()
            imp_df = imp_df.sort_values("Abs", ascending=False).head(15)

            fig, ax = plt.subplots()
            ax.barh(imp_df["Feature"], imp_df["SHAP value"])
            ax.invert_yaxis()
            ax.set_xlabel("SHAP value (impact on log-odds/prob)")
            ax.set_title("Local Feature Attributions")
            st.pyplot(fig)
        else:
            raise RuntimeError("SHAP not available")

    except Exception as e:
        st.markdown("#### 🧪 What-If Sensitivity (fallback)")
        base_pp, imp = what_if_sensitivity(raw_input)
        c1, c2 = st.columns([1,2])
        with c1: st.metric("Baseline Churn Prob", f"{base_pp:.2f}%")
        with c2: st.caption("Bar shows the largest change in predicted probability (in percentage points) when the feature is perturbed.")
        if not imp.empty:
            fig, ax = plt.subplots()
            ax.barh(imp["Feature"], imp["Impact (pp)"])
            ax.invert_yaxis()
            ax.set_xlabel("Impact (percentage points)")
            st.pyplot(fig)
        else:
            st.info("Not enough information to compute sensitivity.")

# ========== Segmentation ==========
elif page == "👥 Segmentation":
    st.subheader("👥 Customer Segmentation (KMeans)")
    hist_df = get_history_df()
    if hist_df.empty:
        st.info("No history yet. Make and save some predictions first.")
    else:
        k = st.slider("Number of clusters (k)", 2, 6, 3)
        # Use numeric features + churn prob
        feats = ["Credit Score", "Age", "Balance", "Tenure", "Products", "Churn Probability (%)"]
        sdf = hist_df[feats].copy().dropna()
        if len(sdf) < k:
            st.warning("Not enough points for the selected number of clusters.")
        else:
            scaler_seg = StandardScaler()
            Xs = scaler_seg.fit_transform(sdf.values)
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = km.fit_predict(Xs)
            sdf["Cluster"] = labels

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Cluster Scatter: Age vs Churn Prob")
                fig, ax = plt.subplots()
                for c in sorted(sdf["Cluster"].unique()):
                    sub = sdf[sdf["Cluster"] == c]
                    ax.scatter(sub["Age"], sub["Churn Probability (%)"], alpha=0.8, edgecolors="k", label=f"C{c}")
                ax.set_xlabel("Age")
                ax.set_ylabel("Churn Probability (%)")
                ax.legend()
                st.pyplot(fig)

            with col2:
                st.markdown("#### Cluster Profiles (mean values)")
                profile = sdf.groupby("Cluster").mean(numeric_only=True).round(2)
                st.dataframe(profile, use_container_width=True)

# ========== Manage ==========
elif page == "💾 Manage":
    st.subheader("💾 Manage Prediction History")
    hist_df = get_history_df()
    if not hist_df.empty:
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download as CSV", csv, "prediction_history.csv", "text/csv", use_container_width=True)
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["history"] = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("Prediction history cleared!")
    else:
        st.info("No history available yet.")
