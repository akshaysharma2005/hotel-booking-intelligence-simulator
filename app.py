import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import load_and_preprocess, train_or_load_model
from src.visualizer import (
    build_2d_monthly_trends,
    build_market_segment_heatmap,
    build_3d_risk_morphospace,
    build_3d_geo_globe,
    build_4d_temporal_risk_animation
)

st.set_page_config(
    page_title="Hotel Booking Intelligence & 4D Revenue Risk Simulator",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    
    .risk-high {
        color: #f43f5e;
        font-weight: 700;
    }
    .risk-low {
        color: #10b981;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def get_data():
    return load_and_preprocess("data/hotel_bookings.csv")

df = get_data()

# Load / train model
@st.cache_resource
def get_model():
    return train_or_load_model(df)

model, feature_cols = get_model()

# Sidebar
st.sidebar.title("🏨 Hospitality Telemetry")
st.sidebar.markdown("Predictive Cancellation AI, 3D Risk Morphospaces & 4D Revenue Simulator.")

hotel_filter = st.sidebar.multiselect("Hotel Property Type", options=df["hotel"].unique(), default=df["hotel"].unique())
year_filter = st.sidebar.multiselect("Arrival Year", options=sorted(df["arrival_date_year"].unique()), default=sorted(df["arrival_date_year"].unique()))
customer_filter = st.sidebar.multiselect("Customer Type", options=df["customer_type"].unique(), default=df["customer_type"].unique())

filtered_df = df[
    (df["hotel"].isin(hotel_filter)) &
    (df["arrival_date_year"].isin(year_filter)) &
    (df["customer_type"].isin(customer_filter))
]

# Hero KPI Section
col_title, col_kpis = st.columns([2, 3])
with col_title:
    st.title("Hotel Booking Intelligence")
    st.caption("AI-powered cancellation risk forecasting, 3D spatial manifolds, and dynamic revenue simulation.")

total_bookings = len(filtered_df)
cancellations = filtered_df["is_canceled"].sum()
cancel_rate = (cancellations / max(1, total_bookings)) * 100
avg_adr = filtered_df["adr"].mean()
rev_at_risk = (filtered_df[filtered_df["is_canceled"] == 1]["adr"] * filtered_df[filtered_df["is_canceled"] == 1]["total_stays"]).sum()

with col_kpis:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <span style="color: #94a3b8; font-size: 11px;">TOTAL BOOKINGS</span>
            <h3 style="margin: 0; color: #38bdf8;">{total_bookings:,}</h3>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <span style="color: #94a3b8; font-size: 11px;">CANCELLATION RATE</span>
            <h3 style="margin: 0; color: #f43f5e;">{cancel_rate:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <span style="color: #94a3b8; font-size: 11px;">AVERAGE DAILY RATE</span>
            <h3 style="margin: 0; color: #10b981;">${avg_adr:.1f}</h3>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <span style="color: #94a3b8; font-size: 11px;">REVENUE AT RISK</span>
            <h3 style="margin: 0; color: #f59e0b;">${rev_at_risk / 1e6:.2f}M</h3>
        </div>
        """, unsafe_allow_html=True)

# Tabs
tab_2d, tab_3d, tab_4d, tab_ml = st.tabs([
    "📊 2D Revenue & Booking Trends",
    "🌐 3D Risk Space & Global Globe",
    "⏳ 4D Temporal Seasonality Simulator",
    "🤖 Live ML Cancellation Predictor"
])

# ==========================================
# TAB 1: 2D REVENUE & BOOKING TRENDS
# ==========================================
with tab_2d:
    st.subheader("Seasonal Booking Volume, Completed Stays, and Pricing Dynamics")
    st.plotly_chart(build_2d_monthly_trends(filtered_df), use_container_width=True)
    
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.plotly_chart(build_market_segment_heatmap(filtered_df), use_container_width=True)
    with col_h2:
        st.markdown("#### Lead Time Distribution by Booking Outcome")
        box_fig = px.box(
            filtered_df.sample(min(8000, len(filtered_df)), random_state=42),
            x="is_canceled",
            y="lead_time",
            color="is_canceled",
            labels={"is_canceled": "Outcome (0=Completed, 1=Canceled)", "lead_time": "Lead Time (Days)"},
            color_discrete_map={0: "#10b981", 1: "#f43f5e"}
        )
        box_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            height=400,
            showlegend=False
        )
        st.plotly_chart(box_fig, use_container_width=True)

# ==========================================
# TAB 2: 3D RISK SPACE & GLOBAL GLOBE
# ==========================================
with tab_3d:
    st.subheader("3D Empirical Booking Risk Morphospace")
    st.markdown("Visualizes bookings in continuous attribute space: Lead Time × ADR × Length of Stay. The 3D translucent plane marks the empirical decision boundary where cancellations surge.")
    
    sample_3d = filtered_df.sample(min(3000, len(filtered_df)), random_state=42)
    risk_3d_fig = build_3d_risk_morphospace(sample_3d)
    st.plotly_chart(risk_3d_fig, use_container_width=True)
    
    st.divider()
    st.subheader("3D Global Traveler Origin & Cancellation Propensity Globe")
    st.markdown("Interactive orthographic 3D earth globe illustrating country-level traveler volume and cancellation vulnerability.")
    globe_fig = build_3d_geo_globe(filtered_df)
    st.plotly_chart(globe_fig, use_container_width=True)

# ==========================================
# TAB 3: 4D TEMPORAL SEASONALITY SIMULATOR
# ==========================================
with tab_4d:
    st.subheader("4D Spatio-Temporal Seasonality & Revenue Simulation")
    st.markdown("Press the **Play** button below to watch market segments, lead times, and ADR migrate dynamically through 3D space across each month of the calendar year.")
    
    anim_risk_fig = build_4d_temporal_risk_animation(filtered_df)
    st.plotly_chart(anim_risk_fig, use_container_width=True)

# ==========================================
# TAB 4: LIVE ML CANCELLATION PREDICTOR
# ==========================================
with tab_ml:
    st.subheader("🤖 Real-Time Booking Cancellation Risk Scoring Engine")
    st.markdown("Tune reservation parameters to evaluate instant machine learning risk probability and test mitigation strategies.")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        lead_time_in = st.slider("Lead Time (Days before arrival)", 0, 365, 90)
        adr_in = st.number_input("Average Daily Rate ($ / Night)", min_value=10.0, max_value=500.0, value=120.0, step=5.0)
        total_stays_in = st.slider("Total Nights Stay", 1, 21, 3)
    with col_in2:
        total_guests_in = st.slider("Total Guests (Adults + Children)", 1, 8, 2)
        special_requests_in = st.slider("Total Special Requests", 0, 5, 1)
        booking_changes_in = st.slider("Booking Modifications / Changes", 0, 5, 0)
    with col_in3:
        prev_cancel_in = st.number_input("Previous Cancellations", 0, 10, 0)
        repeated_guest_in = st.selectbox("Is Repeated Guest?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
    input_data = pd.DataFrame([{
        "lead_time": lead_time_in,
        "adr": adr_in,
        "total_stays": total_stays_in,
        "total_guests": total_guests_in,
        "previous_cancellations": prev_cancel_in,
        "booking_changes": booking_changes_in,
        "total_of_special_requests": special_requests_in,
        "is_repeated_guest": repeated_guest_in
    }])
    
    prob_cancel = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.markdown("### Risk Evaluation")
        if prob_cancel > 50:
            st.markdown(f"<h1 class='risk-high'>{prob_cancel:.1f}%</h1>", unsafe_allow_html=True)
            st.error("⚠️ HIGH CANCELLATION RISK DETECTED")
        else:
            st.markdown(f"<h1 class='risk-low'>{prob_cancel:.1f}%</h1>", unsafe_allow_html=True)
            st.success("✅ LOW CANCELLATION RISK")
        st.progress(float(prob_cancel / 100))
        
    with res_col2:
        st.markdown("### Automated AI Recommendation")
        if prob_cancel > 50:
            st.warning("""
            **Recommended Interventions**:
            1. **Require Non-Refundable Deposit**: Bookings with lead times > 90 days have a 3x higher cancel rate without deposits.
            2. **Automated Confirmation Engagement**: Send a personalized pre-arrival welcome email 14 days and 3 days prior.
            3. **Incentivize Prepaid Add-ons**: Offer breakfast or airport transfer packages to lock in commitment.
            """)
        else:
            st.info("""
            **Booking Health**:
            - Strong commitment profile (low lead time or active special requests).
            - Recommend standard automated pre-check-in link 24 hours prior to maximize guest satisfaction.
            """)
