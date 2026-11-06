# 🏨 Hotel Booking Intelligence & 4D Revenue Risk Simulator

[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Predictive Cancellation AI, 3D Decision Surface Morphospaces, and 4D Temporal Revenue Simulators across 119,000+ Hotel Bookings.**

---

## 🌟 Overview

The **Hotel Booking Intelligence & 4D Revenue Risk Simulator** transforms hospitality data analysis into an actionable, multi-dimensional decision engine. Designed for revenue managers, hoteliers, and data analysts, this application provides empirical risk boundaries, 3D geographic passenger flows, temporal seasonality projections, and instant machine learning cancellation scoring.

```
                              ┌───────────────────────────────────┐
                              │      Hotel Reservation Data       │
                              │    (119,390 Booking Records)      │
                              └─────────────────┬─────────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
      ┌─────────────────────┐        ┌─────────────────────┐        ┌─────────────────────┐
      │  2D Revenue & ADR   │        │   3D Risk Space &   │        │ 4D Seasonality & ML │
      │  Seasonal Curves    │        │   Global Geo Globe  │        │ What-If Simulator   │
      └──────────┬──────────┘        └──────────┬──────────┘        └──────────┬──────────┘
                 │                              │                              │
                 ▼                              ▼                              ▼
     • Monthly Volume / ADR          • Lead Time × ADR × Stay       • Animated Month Time
     • Market Segment Heatmap        • 3D Risk Surface Boundary     • Live AI Risk Prediction
     • Cancellation Distribution     • 3D Orthographic Globe        • Automated Strategy Plan
```

---

## ✨ Features & Visual Dimensions

### 📊 1. 2D Multi-Dimensional Revenue Analytics
- **Dual-Axis Seasonal Curves**: Combined bar & line visualizations depicting confirmed check-outs, cancellations, and Average Daily Rate (ADR) dynamics across the 12-month calendar.
- **Deposit Type & Market Segment Matrix**: Heatmaps highlighting high-risk booking channels (Direct, Corporate, Online TA, Groups) crossed with refundability terms.
- **Lead Time Propensity Distributions**: Comparative distribution boxplots separating cancelations from fulfilled stays.

### 🌐 2. 3D Risk Space & Orthographic Global Globe
- **3D Empirical Booking Risk Space**: Maps individual reservations onto continuous spatial coordinates:
  - **X-Axis**: Lead Time (Days before check-in)
  - **Y-Axis**: ADR ($ / Night)
  - **Z-Axis**: Total Length of Stay (Nights)
  - **3D Decision Surface**: Visualizes the empirical hyper-surface where cancellation likelihood exceeds sustainable operational thresholds.
- **3D Global Traveler Origin Globe**: Interactive orthographic 3D planetary projection of international guest origins, mapping volume and regional cancellation tendencies.

### ⏳ 3. 4D Spatio-Temporal Seasonality Simulator
- **4D Hyper-Dimensional Projection**:
  - **Dim 1 (X)**: Average Lead Time
  - **Dim 2 (Y)**: Average Daily Rate ($)
  - **Dim 3 (Z)**: Average Stay Length
  - **Dim 4 (Color)**: Cancellation Rate %
  - **Dim 5 (Size)**: Total Booking Volume
  - **Dim 6 (Time / Frame Animation)**: Monthly Temporal Cycle (January through December)
- Watch demand clusters migrate through spatial coordinate space as peak seasons transition to off-peak periods.

### 🤖 4. Live ML Cancellation Predictor & Strategy Advisor
- Real-time Random Forest inference pipeline evaluating custom reservations.
- Instant probability gauge and automated revenue management interventions (e.g., non-refundable deposit recommendations, targeted engagement schedules).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/akshaysharma2005/hotel-booking-intelligence-simulator.git
   cd hotel-booking-intelligence-simulator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

4. Navigate to `http://localhost:8501`.

---

## 📂 Repository Structure

```
hotel-booking-intelligence-simulator/
├── app.py                     # Streamlit multi-tab intelligence dashboard
├── requirements.txt           # Python package dependencies
├── README.md                  # Comprehensive documentation
├── .gitignore                 # Git ignore rules
├── data/
│   └── hotel_bookings.csv     # 119k hotel booking records
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data preprocessing & Random Forest training
│   └── visualizer.py          # Plotly 2D, 3D, and 4D animated visual pipelines
└── notebooks/
    └── Hotel_Bookings.ipynb   # Original exploratory data mining notebook
```

---

## 👤 Author
- **M Akshay Sharma** ([@akshaysharma2005](https://github.com/akshaysharma2005))
- GitHub: [https://github.com/akshaysharma2005](https://github.com/akshaysharma2005)
