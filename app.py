import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Telco Customer Analytics & Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look (glassmorphism cards, modern fonts, clean layout)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    .stCard {
        background: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #eef2f6;
        margin-bottom: 20px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-left: 5px solid #1e3a8a;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        margin: 10px 0;
    }
    
    .metric-card.churn {
        border-left-color: #ef4444;
    }
    
    .metric-card.retention {
        border-left-color: #10b981;
    }
    
    .metric-card.revenue {
        border-left-color: #8b5cf6;
    }
    
    .metric-card.cltv {
        border-left-color: #f59e0b;
    }
    
    .section-title {
        color: #0f172a;
        font-weight: 700;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    .highlight-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    
    .highlight-box.risk {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
    }
    
    .highlight-box.info {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load data and models
@st.cache_data
def load_data():
    # Load dataset
    data_path = os.path.join("data", "customer_segmentation_processed.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # Fallback to local workspace if run from top level
        df = pd.read_csv(r"c:\Users\bhard\OneDrive\Desktop\customer\customer_segmentation_processed.csv")
    
    # Map Churn Label to 1 and 0 for numerical ops if not already
    df['Churn_Value_Num'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
    return df

@st.cache_resource
def load_models():
    models_dir = "models"
    
    xgb = joblib.load(os.path.join(models_dir, "xgboost.pkl"))
    rf = joblib.load(os.path.join(models_dir, "random_forest.pkl"))
    lr = joblib.load(os.path.join(models_dir, "logistic.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(models_dir, "kmeans.pkl"))
    
    with open(os.path.join(models_dir, "cluster_mapping.json"), "r") as f:
        cluster_mapping = json.load(f)
        
    with open(os.path.join(models_dir, "feature_columns.json"), "r") as f:
        feature_columns = json.load(f)
        
    return xgb, rf, lr, scaler, kmeans, cluster_mapping, feature_columns

# Load dataset and models
try:
    df = load_data()
    xgb, rf, lr, scaler, kmeans, cluster_mapping, feature_columns = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models or dataset: {e}. Please ensure you ran organize_and_train.py first.")
    models_loaded = False

# Premium Sidebar header and navigation
st.sidebar.markdown("""
<div style='text-align: center; padding: 10px 0;'>
    <h2 style='color: #1e3a8a; margin-bottom: 0;'>Telco Analytics</h2>
    <p style='color: #64748b; font-size: 0.85rem;'>Production-Grade DS Portfolio</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Dashboard Navigation",
    ["📊 Executive Summary", 
     "🔍 Exploratory Data Analysis", 
     "👥 Customer Segmentation", 
     "🔮 Churn Prediction Playground", 
     "📈 Feature Importance & XAI", 
     "🗃️ Raw Database Explorer",
     "💡 Business Recommendations"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Model Benchmarks
* **XGBoost (Best Model)**
  * Accuracy: **80.24%**
  * Recall: **58.00%**
  * F1 Score: **61.00%**
* **Logistic Regression**
  * Accuracy: **79.67%**
  * Recall: **57.00%**
  * F1 Score: **60.00%**
* **Random Forest**
  * Accuracy: **79.39%**
  * Recall: **49.00%**
  * F1 Score: **56.00%**
""")

if models_loaded:
    # ----------------------------------------------------
    # Page 1: Executive Summary
    # ----------------------------------------------------
    if page == "📊 Executive Summary":
        st.title("📊 Executive Summary")
        st.markdown("A strategic business overview of customer retention, revenue risks, and customer segments.")
        
        # Calculate KPI Metrics
        total_cust = len(df)
        churn_rate = (df['Churn Label'] == 'Yes').mean() * 100
        retention_rate = 100 - churn_rate
        total_rev = df['Total Charges'].sum()
        avg_cltv = df['CLTV'].mean()
        
        # Display KPIs in styled cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Customers</div>
                <div class="metric-value">{total_cust:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card churn">
                <div class="metric-label">Churn Rate</div>
                <div class="metric-value">{churn_rate:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card retention">
                <div class="metric-label">Retention Rate</div>
                <div class="metric-value">{retention_rate:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card revenue">
                <div class="metric-label">Total Revenue</div>
                <div class="metric-value">${total_rev/1e6:.2f}M</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col5:
            st.markdown(f"""
            <div class="metric-card cltv">
                <div class="metric-label">Average CLTV</div>
                <div class="metric-value">${avg_cltv:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<h2 class='section-title'>Executive Summary & Business Highlights</h2>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("""
            * 🚨 **Early Lifecycle Churn (New Customers)**: Customers in their first year (tenure ≤ 12 months) represent **48% of total churned users**. This highlights onboarding and early engagement gaps.
            * 🔌 **Technology Friction (Fiber Optic)**: Subscribers with **Fiber Optic** internet service have a significantly higher churn rate compared to DSL users. High monthly bills and competitor speed matching are key drivers.
            * 🛡️ **Value Services as Retention Anchors**: Penetration of **Tech Support** and **Online Security** is highly correlated with retention. Customers without these features are **3.5x more likely to leave**.
            * 📜 **Billing & Contract Structures**: Month-to-month contracts account for **88.5% of all churned accounts**, emphasizing the high ROI of contract-conversion campaigns.
            """)
            
            # Actionable Highlight Card
            st.markdown("""
            <div class="highlight-box info">
                <h4>💡 Recruiter Insights (Explainability Focus)</h4>
                This dashboard uses <b>Unsupervised Clustering</b> to segment the base, and an <b>XGBoost Classifier</b> to predict churn risk. 
                By translating probabilities into <b>Revenue at Risk ($)</b>, we tie machine learning outputs directly to financial decisions.
            </div>
            """, unsafe_allow_html=True)
            
        with col_right:
            # Simple pie chart of Churn vs Retention
            fig_pie = px.pie(
                names=["Retained", "Churned"],
                values=[retention_rate, churn_rate],
                color=["Retained", "Churned"],
                color_discrete_map={"Retained": "#10b981", "Churned": "#ef4444"},
                hole=0.5,
                title="Customer Distribution (Retention vs. Churn)"
            )
            fig_pie.update_layout(
                margin=dict(t=40, b=0, l=0, r=0),
                height=260,
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Segment Summary Table
        st.markdown("<h2 class='section-title'>Customer Segment Performance</h2>", unsafe_allow_html=True)
        seg_summary = df.groupby('Segment').agg(
            Count=('CustomerID', 'count'),
            Avg_Tenure=('Tenure Months', 'mean'),
            Avg_Monthly_Charges=('Monthly Charges', 'mean'),
            Avg_CLTV=('CLTV', 'mean'),
            Churn_Rate=('Churn_Value_Num', lambda x: x.mean() * 100)
        ).reset_index()
        
        seg_summary.columns = [
            'Customer Segment', 'Customer Count', 'Avg Tenure (Months)', 
            'Avg Monthly Charges ($)', 'Avg CLTV ($)', 'Churn Rate (%)'
        ]
        
        st.dataframe(
            seg_summary.style.format({
                'Customer Count': '{:,}',
                'Avg Tenure (Months)': '{:.1f}',
                'Avg Monthly Charges ($)': '${:.2f}',
                'Avg CLTV ($)': '${:,.0f}',
                'Churn Rate (%)': '{:.2f}%'
            }).background_gradient(cmap='Blues', subset=['Customer Count', 'Avg CLTV ($)'])
              .background_gradient(cmap='Reds', subset=['Churn Rate (%)']),
            use_container_width=True,
            hide_index=True
        )

    # ----------------------------------------------------
    # Page 2: Exploratory Data Analysis
    # ----------------------------------------------------
    elif page == "🔍 Exploratory Data Analysis":
        st.title("🔍 Exploratory Data Analysis")
        st.markdown("Identify trends and correlations within demographics, services, and billing structures.")
        
        tab1, tab2, tab3 = st.tabs(["Contracts & Demographics", "Charges & Subscriptions", "Churn Reasons"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Contract Type vs Churn
                contract_churn = df.groupby(['Contract', 'Churn Label']).size().reset_index(name='Count')
                fig_contract = px.bar(
                    contract_churn,
                    x="Contract",
                    y="Count",
                    color="Churn Label",
                    color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                    barmode="group",
                    title="Churn by Contract Commitment"
                )
                fig_contract.update_layout(height=350)
                st.plotly_chart(fig_contract, use_container_width=True)
                
            with col2:
                # Tech Support vs Churn
                support_churn = df.groupby(['Tech Support', 'Churn Label']).size().reset_index(name='Count')
                fig_support = px.bar(
                    support_churn,
                    x="Tech Support",
                    y="Count",
                    color="Churn Label",
                    color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                    barmode="group",
                    title="Churn by Tech Support Subscription"
                )
                fig_support.update_layout(height=350)
                st.plotly_chart(fig_support, use_container_width=True)
                
            # Tenure vs Churn distribution
            fig_tenure = px.histogram(
                df,
                x="Tenure Months",
                color="Churn Label",
                color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                marginal="box",
                barmode="overlay",
                title="Tenure Distribution (Months) by Churn Status"
            )
            fig_tenure.update_layout(height=380)
            st.plotly_chart(fig_tenure, use_container_width=True)
            
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly Charges vs Churn density
                fig_charges = px.histogram(
                    df,
                    x="Monthly Charges",
                    color="Churn Label",
                    color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                    marginal="box",
                    barmode="overlay",
                    title="Monthly Charges ($) by Churn Status"
                )
                fig_charges.update_layout(height=350)
                st.plotly_chart(fig_charges, use_container_width=True)
                
            with col2:
                # Payment Method vs Churn
                pay_churn = df.groupby(['Payment Method', 'Churn Label']).size().reset_index(name='Count')
                fig_payment = px.bar(
                    pay_churn,
                    y="Payment Method",
                    x="Count",
                    color="Churn Label",
                    color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                    orientation="h",
                    barmode="group",
                    title="Churn by Payment Method"
                )
                fig_payment.update_layout(height=350, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_payment, use_container_width=True)
                
            # Internet Service vs Churn
            internet_churn = df.groupby(['Internet Service', 'Churn Label']).size().reset_index(name='Count')
            fig_internet = px.bar(
                internet_churn,
                x="Internet Service",
                y="Count",
                color="Churn Label",
                color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                barmode="group",
                title="Churn by Internet Service Type"
            )
            fig_internet.update_layout(height=350)
            st.plotly_chart(fig_internet, use_container_width=True)
            
        with tab3:
            # Top Churn Reasons
            churned_df = df[df['Churn Label'] == 'Yes']
            churn_reasons = churned_df['Churn Reason'].value_counts().reset_index()
            churn_reasons.columns = ['Churn Reason', 'Count']
            
            fig_reasons = px.bar(
                churn_reasons.head(10),
                x="Count",
                y="Churn Reason",
                orientation="h",
                title="Top 10 Logged Reasons for Churn",
                color="Count",
                color_continuous_scale="Reds"
            )
            fig_reasons.update_layout(height=450, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_reasons, use_container_width=True)
            
            st.info("""
            💡 **Business Takeaway**: The leading reasons are competitor product qualities (e.g., 'Competitor made better offer', 'Competitor had better devices'). 
            This validates that pricing campaigns, speed enhancements, and product bundles are direct tools for retention.
            """)

    # ----------------------------------------------------
    # Page 3: Customer Segmentation
    # ----------------------------------------------------
    elif page == "👥 Customer Segmentation":
        st.title("👥 Customer Segmentation (K-Means Clustering)")
        st.markdown("Understand custom customer segments derived from Tenure, Monthly Charges, Total Charges, and CLTV.")
        
        col_pie, col_stats = st.columns([1, 2])
        
        with col_pie:
            seg_counts = df['Segment'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            fig_seg_pie = px.pie(
                seg_counts,
                names="Segment",
                values="Count",
                color="Segment",
                color_discrete_map={
                    "VIP Customers": "#8b5cf6",
                    "Loyal Customers": "#10b981",
                    "New Customers": "#3b82f6",
                    "At-Risk Customers": "#ef4444"
                },
                title="Segment Representation",
                hole=0.4
            )
            fig_seg_pie.update_layout(height=320, margin=dict(t=50, b=0, l=0, r=0))
            st.plotly_chart(fig_seg_pie, use_container_width=True)
            
        with col_stats:
            st.markdown("#### Customer Segment Profiling")
            st.markdown("""
            * 🌟 **VIP Customers**: Long-term subscribers with high-spend services (Average Monthly Charges > $90). Highly loyal but require premium support tiers to maintain satisfaction.
            * 🤝 **Loyal Customers**: Stable base with moderate-to-low charges. They represent consistent recurring baseline revenue with low support overhead.
            * 🆕 **New Customers**: Recently onboarded accounts. Critical cohort: high CLTV potential, but high attrition risk. Needs active success triggers during the first 6 months.
            * ⚠️ **At-Risk Customers**: Short-to-medium term subscribers with moderate spend but low CLTV. Prone to churn quickly; require immediate marketing offers.
            """)
            
        # Interactive Scatter Plot
        st.markdown("<h2 class='section-title'>Interactive 2D Segment Space</h2>", unsafe_allow_html=True)
        
        col_x, col_y, col_sz = st.columns(3)
        with col_x:
            x_axis = st.selectbox("Select X-Axis Feature", ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV'], index=0)
        with col_y:
            y_axis = st.selectbox("Select Y-Axis Feature", ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV'], index=1)
        with col_sz:
            size_axis = st.selectbox("Select Marker Size Feature", [None, 'Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV'], index=3)
            
        fig_scatter = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color="Segment",
            size=size_axis,
            hover_data=['CustomerID', 'Contract', 'Internet Service', 'Churn Label'],
            color_discrete_map={
                "VIP Customers": "#8b5cf6",
                "Loyal Customers": "#10b981",
                "New Customers": "#3b82f6",
                "At-Risk Customers": "#ef4444"
            },
            opacity=0.7,
            title=f"{x_axis} vs {y_axis} by Customer Segment"
        )
        fig_scatter.update_layout(height=550)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ----------------------------------------------------
    # Page 4: Churn Prediction Playground
    # ----------------------------------------------------
    elif page == "🔮 Churn Prediction Playground":
        st.title("🔮 Churn Prediction Sandbox")
        st.markdown("Adjust customer attributes below to test individual churn risk probability across models, review local explainability drivers, and calculate revenue at risk.")
        
        # Multi-model selection comparison
        st.sidebar.markdown("### Prediction Models")
        selected_model_name = st.sidebar.selectbox(
            "Primary Classification Model",
            ["XGBoost (Accuracy: 80.24%)", "Random Forest (Accuracy: 79.39%)", "Logistic Regression (Accuracy: 79.67%)"]
        )
        
        if "XGBoost" in selected_model_name:
            primary_model = xgb
        elif "Random Forest" in selected_model_name:
            primary_model = rf
        else:
            primary_model = lr

        # Interactive Form
        with st.form("prediction_form"):
            st.markdown("### 1. Demographic Profile")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            with col_d2:
                partner = st.selectbox("Has Partner?", ["No", "Yes"])
                dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
            with col_d3:
                city = st.selectbox("City (California)", sorted(df['City'].unique().tolist()), index=sorted(df['City'].unique().tolist()).index("Los Angeles"))
                zip_code = st.number_input("Zip Code", min_value=90001, max_value=96161, value=90003)
                
            st.markdown("### 2. Services Portfolio")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                mult_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            with col_s2:
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            with col_s3:
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                
            st.markdown("### 3. Financial & Plan Parameters")
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            with col_f2:
                payment = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                cltv = st.slider("CLTV Score", min_value=2000, max_value=6500, value=4000)
            with col_f3:
                tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
                monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
            
            # Extrapolate charges
            total_charges_est = monthly_charges * tenure
            total_charges = st.number_input("Estimated Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(total_charges_est))
            
            submit_btn = st.form_submit_button("Run Analytics & Churn Prediction")
            
        if submit_btn:
            # --- 1. Derive Categorical Groups ---
            tenure_group = 'New' if tenure <= 12 else ('Growing' if tenure <= 24 else ('Loyal' if tenure <= 48 else 'Very Loyal'))
            charge_group = 'Low' if monthly_charges <= 35.55 else ('Medium' if monthly_charges <= 70.35 else ('High' if monthly_charges <= 89.85 else 'Premium'))
            cltv_group = 'Low Value' if cltv <= 3469 else ('Medium Value' if cltv <= 4527 else ('High Value' if cltv <= 5381 else 'VIP'))
            
            # --- 2. Predict Cluster / Segment ---
            cluster_input = pd.DataFrame([{
                'Tenure Months': float(tenure),
                'Monthly Charges': float(monthly_charges),
                'Total Charges': float(total_charges),
                'CLTV': float(cltv)
            }])
            scaled_cluster_data = scaler.transform(cluster_input)
            predicted_cluster = int(kmeans.predict(scaled_cluster_data)[0])
            predicted_segment = cluster_mapping[str(predicted_cluster)]
            
            # --- 3. Build Feature Dictionary & One-Hot Encode ---
            input_dict = {
                'Country': 'United States',
                'State': 'California',
                'City': city,
                'Gender': gender,
                'Senior Citizen': senior,
                'Partner': partner,
                'Dependents': dependents,
                'Tenure Months': tenure,
                'Phone Service': phone_service,
                'Multiple Lines': mult_lines,
                'Internet Service': internet_service,
                'Online Security': online_security,
                'Online Backup': online_backup,
                'Device Protection': device_protection,
                'Tech Support': tech_support,
                'Streaming TV': streaming_tv,
                'Streaming Movies': streaming_movies,
                'Contract': contract,
                'Paperless Billing': paperless,
                'Payment Method': payment,
                'Monthly Charges': monthly_charges,
                'Total Charges': total_charges,
                'CLTV': cltv,
                'Zip Code': zip_code,
                'Tenure_Group': tenure_group,
                'Charge_Group': charge_group,
                'CLTV_Group': cltv_group,
                'Cluster': predicted_cluster,
                'Segment': predicted_segment
            }
            
            input_df = pd.DataFrame([input_dict])
            input_encoded = pd.get_dummies(input_df)
            input_final = input_encoded.reindex(columns=feature_columns, fill_value=0)
            
            # --- 4. Model Predictions ---
            xgb_prob = xgb.predict_proba(input_final)[0, 1]
            rf_prob = rf.predict_proba(input_final)[0, 1]
            lr_prob = lr.predict_proba(input_final)[0, 1]
            
            active_prob = xgb_prob if "XGBoost" in selected_model_name else (rf_prob if "Random Forest" in selected_model_name else lr_prob)
            prob_percent = active_prob * 100
            
            # Risk Category setup
            if prob_percent < 30.0:
                risk_cat, risk_color = "Low Risk", "#10b981"
            elif prob_percent <= 70.0:
                risk_cat, risk_color = "Medium Risk", "#f59e0b"
            else:
                risk_cat, risk_color = "High Risk", "#ef4444"
                
            # --- 5. Display Interface Output ---
            st.markdown("<h2 class='section-title'>Customer Analysis & Explanations</h2>", unsafe_allow_html=True)
            
            col_res1, col_res2 = st.columns([1, 1])
            
            with col_res1:
                # Radial Gauges
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_percent,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Predicted Churn Probability (%)", 'font': {'size': 18}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': risk_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.1)'},
                            {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                            {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(t=50, b=0, l=0, r=0))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with col_res2:
                # Quantify Financial Impact
                monthly_rev_risk = monthly_charges * active_prob
                annual_rev_risk = monthly_rev_risk * 12
                cltv_risk = cltv * active_prob
                
                st.markdown(f"""
                <div style='background: white; border-radius: 12px; padding: 22px; border: 1px solid #eef2f6; box-shadow: 0 4px 6px rgba(0,0,0,0.02);'>
                    <h3 style='margin-top: 0; color: #1e3a8a;'>Financial Risk Assessment</h3>
                    <p>📊 <b>Risk Level:</b> <span style='font-size: 1.15rem; font-weight: bold; color: {risk_color};'>{risk_cat} ({prob_percent:.2f}%)</span></p>
                    <p>👥 <b>Segment Assignment:</b> <b>{predicted_segment}</b> (Cluster {predicted_cluster})</p>
                    <p>💸 <b>Monthly Revenue at Risk:</b> <span style='font-weight: bold;'>${monthly_rev_risk:.2f}</span> / month</p>
                    <p>📈 <b>CLTV Value at Risk:</b> <span style='font-weight: bold;'>${cltv_risk:.2f}</span> (Total CLTV: ${cltv})</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Local Explainability (XAI) Details
            st.markdown("### Local Explainability Engine (XAI)")
            col_xai1, col_xai2 = st.columns(2)
            
            with col_xai1:
                # Risk Drivers
                drivers = []
                if contract == "Month-to-month":
                    drivers.append("❌ **Flexible Month-to-Month Contract**: Positively influences churn risk (low contract lock-in).")
                if internet_service == "Fiber optic":
                    drivers.append("❌ **Fiber Optic Service Plan**: Linked to higher bills and competitor promotional switching.")
                if tech_support == "No":
                    drivers.append("❌ **No Tech Support Subscription**: Limits service integration and increases frustration rate.")
                if online_security == "No":
                    drivers.append("❌ **No Online Security Option**: Missing value-add service reduces overall 'stickiness'.")
                if tenure <= 12:
                    drivers.append("❌ **Early Lifecycle Phase**: Customers under 12 months are in the peak attrition window.")
                    
                st.markdown("#### 🔴 Positive Churn Risk Drivers")
                if drivers:
                    for d in drivers:
                        st.markdown(d)
                else:
                    st.markdown("No major risk drivers detected. Account features indicate strong stability.")
                    
            with col_xai2:
                # Anchors
                anchors = []
                if contract in ["One year", "Two year"]:
                    anchors.append("✅ **Long-Term Contract Commitment**: Heavy anchor preventing easy switching.")
                if tech_support == "Yes":
                    anchors.append("✅ **Active Tech Support Plan**: Enhances relationship and limits service issues.")
                if online_security == "Yes":
                    anchors.append("✅ **Active Security Shield**: Increases account dependency and protection value.")
                if tenure > 24:
                    anchors.append("✅ **Established Account History**: High tenure demonstrates trust and brand preference.")
                if dependents == "Yes":
                    anchors.append("✅ **Family/Dependent Profile**: Multi-user accounts show higher switching friction.")
                    
                st.markdown("#### 🟢 Negative Churn Drivers (Retention Anchors)")
                if anchors:
                    for a in anchors:
                        st.markdown(a)
                else:
                    st.markdown("No significant retention anchors found. Account is vulnerable to competitors.")

            # Model Probability Comparison Panel
            st.markdown("### Cross-Classifier Model Comparison")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(label="XGBoost Churn Score", value=f"{xgb_prob*100:.2f}%", delta=f"{xgb_prob*100 - prob_percent:.2f}% vs selected")
            with col_m2:
                st.metric(label="Random Forest Churn Score", value=f"{rf_prob*100:.2f}%", delta=f"{rf_prob*100 - prob_percent:.2f}% vs selected")
            with col_m3:
                st.metric(label="Logistic Regression Churn Score", value=f"{lr_prob*100:.2f}%", delta=f"{lr_prob*100 - prob_percent:.2f}% vs selected")

            # Actions and ROI
            st.markdown("### Strategic Retention Recommendation & ROI Simulation")
            if contract == "Month-to-month":
                roi_discount = monthly_charges * 0.1
                saved_yearly = (monthly_charges - roi_discount) * 12
                st.markdown(f"""
                <div class="highlight-box">
                    ⚙️ <b>Retention Campaign Action Plan</b>:<br/>
                    Offer this customer a <b>10% contract discount (${roi_discount:.2f}/mo)</b> to commit to a <b>1-Year Contract</b>.<br/>
                    * <b>Discount Cost (1 Year)</b>: ${roi_discount * 12:.2f}<br/>
                    * <b>Revenue Secured (1 Year)</b>: ${saved_yearly:.2f}<br/>
                    * <b>Retention Probability Jump</b>: Upgrading to a 1-year contract reduces churn probability by over <b>40%</b> based on model parameters.<br/>
                    * <b>ROI Multiplier</b>: <b>{saved_yearly / (roi_discount * 12):.1f}x</b> yield on discount investment.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="highlight-box">
                    ⚙️ <b>Service Cross-Sell Action Plan</b>:<br/>
                    This customer is already on a committed contract. Focus marketing campaigns on cross-selling **Online Backup** or **Streaming Add-ons** 
                    to increase overall account monthly billing, while keeping customer satisfaction high through regular support audits.
                </div>
                """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # Page 5: Feature Importance & XAI
    # ----------------------------------------------------
    elif page == "📈 Feature Importance & XAI":
        st.title("📈 Feature Importance & Global Explainability")
        st.markdown("Understand the global drivers powering the model's classifications across the entire customer base.")
        
        # Calculate/Get XGBoost feature importances
        importances = xgb.feature_importances_
        feat_imp_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Map feature names to clean names
        def clean_feature_name(name):
            name = name.replace("Contract_Two year", "Two-Year Contract")
            name = name.replace("Contract_One year", "One-Year Contract")
            name = name.replace("Internet Service_Fiber optic", "Fiber Optic Internet")
            name = name.replace("Internet Service_No", "No Internet Service")
            name = name.replace("Payment Method_Electronic check", "Payment: Electronic Check")
            name = name.replace("Tech Support_Yes", "Has Tech Support")
            name = name.replace("Online Security_Yes", "Has Online Security")
            name = name.replace("Online Backup_Yes", "Has Online Backup")
            name = name.replace("Paperless Billing_Yes", "Paperless Billing Enrolled")
            name = name.replace("Dependents_Yes", "Has Dependents")
            name = name.replace("Senior Citizen_Yes", "Is Senior Citizen")
            name = name.replace("Tenure_Group_Very Loyal", "Tenure Group: Very Loyal (49-72m)")
            name = name.replace("Tenure_Group_Loyal", "Tenure Group: Loyal (25-48m)")
            name = name.replace("Tenure_Group_New", "Tenure Group: New (1-12m)")
            name = name.replace("Charge_Group_Premium", "Charges: Premium")
            name = name.replace("Charge_Group_High", "Charges: High")
            name = name.replace("CLTV_Group_VIP", "CLTV: VIP")
            name = name.replace("Segment_At-Risk Customers", "Segment: At-Risk")
            name = name.replace("Segment_New Customers", "Segment: New Customer")
            name = name.replace("Segment_VIP Customers", "Segment: VIP")
            return name
            
        feat_imp_df['Clean_Feature'] = feat_imp_df['Feature'].apply(clean_feature_name)
        top_feats = feat_imp_df.head(15)
        
        fig_imp = px.bar(
            top_feats,
            x="Importance",
            y="Clean_Feature",
            orientation="h",
            title="Top 15 Feature Importances (XGBoost Classifier)",
            color="Importance",
            color_continuous_scale="Viridis"
        )
        fig_imp.update_layout(height=450, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("<h2 class='section-title'>Explainability Report: Interpreting Global Drivers</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 1. Contract commitment (High Impact)
            * **Interpretation**: One-year and Two-year contracts reduce flexibility to churn. This establishes high switching costs, locking users into longer customer lifecycles.
            
            #### 2. Service Plan Type (Fiber Optic vs DSL)
            * **Interpretation**: Fiber Optic represents premium connection speeds but is priced higher. Customers are sensitive to high bills and vulnerable to competitor promotions offering lower monthly rates for equal speeds.
            """)
        with col2:
            st.markdown("""
            #### 3. Protection & Support Features
            * **Interpretation**: Features like Tech Support and Online Security act as relationship builders. When users configure online backups and security suites, it binds them technically to the company, creating product stickiness.
            
            #### 4. Account Lifecycle (Tenure Months)
            * **Interpretation**: Early billing cycles present the highest attrition. Customers who do not integrate services or fail to resolve early onboarding issues are quick to discontinue services.
            """)

    # ----------------------------------------------------
    # Page 6: Raw Database Explorer
    # ----------------------------------------------------
    elif page == "🗃️ Raw Database Explorer":
        st.title("🗃️ Customer Database Explorer")
        st.markdown("Filter and query the customer dataset dynamically. (Recruiter / Data Analyst View)")
        
        # Sidebar/Top bar interactive filters
        st.markdown("### Filters")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        with col_f1:
            segment_filter = st.multiselect("Customer Segment", df['Segment'].unique().tolist(), default=df['Segment'].unique().tolist())
        with col_f2:
            churn_filter = st.multiselect("Churn Status", df['Churn Label'].unique().tolist(), default=df['Churn Label'].unique().tolist())
        with col_f3:
            contract_filter = st.multiselect("Contract Type", df['Contract'].unique().tolist(), default=df['Contract'].unique().tolist())
        with col_f4:
            internet_filter = st.multiselect("Internet Service Type", df['Internet Service'].unique().tolist(), default=df['Internet Service'].unique().tolist())
            
        # Apply filters
        filtered_df = df[
            (df['Segment'].isin(segment_filter)) &
            (df['Churn Label'].isin(churn_filter)) &
            (df['Contract'].isin(contract_filter)) &
            (df['Internet Service'].isin(internet_filter))
        ]
        
        # Metrics for filtered dataset
        st.markdown("#### Cohort Statistics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Cohort Volume", f"{len(filtered_df):,} accounts")
        with c2:
            filt_churn_rate = (filtered_df['Churn Label'] == 'Yes').mean() * 100 if len(filtered_df) > 0 else 0.0
            st.metric("Cohort Churn Rate", f"{filt_churn_rate:.2f}%")
        with c3:
            filt_charges = filtered_df['Monthly Charges'].mean() if len(filtered_df) > 0 else 0.0
            st.metric("Avg Monthly Bill", f"${filt_charges:.2f}")
        with c4:
            filt_cltv = filtered_df['CLTV'].mean() if len(filtered_df) > 0 else 0.0
            st.metric("Avg CLTV Score", f"${filt_cltv:,.0f}")
            
        # Custom query engine
        st.markdown("### Custom Pandas Query sandbox")
        query_input = st.text_input("Enter Pandas filter query (e.g., `CLTV > 5000 and Contract == 'Two year'`):", value="")
        
        display_df = filtered_df.copy()
        if query_input:
            try:
                display_df = display_df.query(query_input)
                st.success(f"Query executed successfully! Found {len(display_df)} records.")
            except Exception as e:
                st.error(f"Invalid query: {e}. Please enter a valid pandas query.")
                
        # Data table
        st.dataframe(
            display_df[['CustomerID', 'City', 'Gender', 'Tenure Months', 'Contract', 'Internet Service', 'Monthly Charges', 'CLTV', 'Segment', 'Churn Label']].head(100),
            use_container_width=True
        )
        
        # Download Cohort
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Cohort as CSV",
            data=csv_data,
            file_name="telco_filtered_cohort.csv",
            mime="text/csv"
        )

    # ----------------------------------------------------
    # Page 7: Business Recommendations
    # ----------------------------------------------------
    else:
        st.title("💡 Strategic Business Recommendations")
        st.markdown("Data-driven business action playbook based on customer segment insights and churn drivers.")
        
        st.markdown("<h2 class='section-title'>1. Customer Retention Programs</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("""
            **Retention Focus**:
            * Target: **New Customers** & **At-Risk Customers**
            * Objective: Decrease first-year churn by 20%
            """)
        with col2:
            st.markdown("""
            * 📬 **Structured 90-Day Onboarding**: Introduce check-ins from success teams during the first 3 billing cycles. Provide tutorials on managing fiber connection features.
            * ⚠️ **Predictive Loyalty Triggers**: Flag accounts automatically if their XGBoost Churn probability exceeds **70%**. Initiate personal customer success calls with contract upgrade promos.
            """)
            
        st.markdown("<h2 class='section-title'>2. Plan & Contract Conversion</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.warning("""
            **Contract Strategy**:
            * Target: **Month-to-Month Contracts**
            * Objective: Upgrade 15% of flexible accounts to commitments
            """)
        with col2:
            st.markdown("""
            * 📜 **Commitment Discounts**: Offer a recurring credit ($5-$10/mo) for month-to-month users who sign up for a 12-month contract. The upfront discount cost is heavily offset by the 4x reduction in churn rate.
            * 💳 **Auto-Pay Bonuses**: Offer a one-time credit of $10 to switch from paper checks or electronic checks to auto-pay.
            """)
            
        st.markdown("<h2 class='section-title'>3. Service Attachment Campaigns</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.success("""
            **Service Expansion**:
            * Target: Accounts without **Tech Support** or **Online Security**
            * Objective: Boost service penetration to increase account stickiness
            """)
        with col2:
            st.markdown("""
            * 🛡️ **Free trials of Sticky Services**: Include 3 months of complimentary Online Backup and Tech Support for all fiber optic setups. 
            * ☎️ **VIP Support Tier**: Guarantee dedicated service agents for the high-value **VIP Customers** segment.
            """)
