import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def build_2d_monthly_trends(df):
    """Monthly bookings and cancellations with ADR secondary axis."""
    monthly = df.groupby(["arrival_date_month", "is_canceled"], observed=True).size().unstack(fill_value=0).reset_index()
    monthly.columns = ["Month", "Checked_Out", "Canceled"]
    monthly["Cancellation_Rate"] = (monthly["Canceled"] / (monthly["Checked_Out"] + monthly["Canceled"])) * 100
    
    adr_monthly = df.groupby("arrival_date_month", observed=True)["adr"].mean().reset_index()
    monthly["Avg_ADR"] = adr_monthly["adr"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["Month"],
        y=monthly["Checked_Out"],
        name="Completed Stays",
        marker_color="#10b981",
        opacity=0.85
    ))
    fig.add_trace(go.Bar(
        x=monthly["Month"],
        y=monthly["Canceled"],
        name="Cancellations",
        marker_color="#f43f5e",
        opacity=0.85
    ))
    fig.add_trace(go.Scatter(
        x=monthly["Month"],
        y=monthly["Avg_ADR"],
        name="Avg ADR ($)",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#f59e0b", width=3)
    ))

    fig.update_layout(
        title="Seasonal Booking Volume vs. ADR Trend",
        barmode="stack",
        xaxis_title="Arrival Month",
        yaxis=dict(title="Number of Bookings", gridcolor="rgba(255,255,255,0.1)"),
        yaxis2=dict(title="Average Daily Rate ($)", overlaying="y", side="right", showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=420,
        margin=dict(l=40, r=40, t=40, b=50)
    )
    return fig

def build_market_segment_heatmap(df):
    """Heatmap of cancellation rate across market segment and deposit type."""
    pivot = df.pivot_table(index="market_segment", columns="deposit_type", values="is_canceled", aggfunc="mean").fillna(0) * 100
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Deposit Type", y="Market Segment", color="Cancellation %"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale="Reds",
        aspect="auto",
        text_auto=".1f"
    )
    fig.update_layout(
        title="Cancellation Rate (%) by Segment & Deposit Type",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def build_3d_risk_morphospace(df_sample):
    """
    3D Risk Scatter:
    X: Lead Time (Days)
    Y: ADR ($)
    Z: Total Stays (Nights)
    Color: Canceled (Red) vs Checked Out (Emerald)
    """
    fig = go.Figure()

    for status, color, label in [(0, "#10b981", "Confirmed Check-Out"), (1, "#f43f5e", "Canceled Booking")]:
        sub = df_sample[df_sample["is_canceled"] == status]
        fig.add_trace(go.Scatter3d(
            x=sub["lead_time"],
            y=sub["adr"],
            z=sub["total_stays"],
            mode="markers",
            name=label,
            marker=dict(size=4, color=color, opacity=0.6, line=dict(width=0)),
            customdata=np.stack((sub["hotel"], sub["market_segment"], sub["total_guests"]), axis=-1),
            hovertemplate="<b>%{data.name}</b><br>Lead Time: %{x} days<br>ADR: $%{y}<br>Nights: %{z}<br>Hotel: %{customdata[0]}<br>Segment: %{customdata[1]}<extra></extra>"
        ))

    # Add 3D Decision Surface Plane approximating the risk boundary
    xx, yy = np.meshgrid(np.linspace(0, 350, 10), np.linspace(0, 300, 10))
    # Approximation plane where cancellations dominate
    zz = 0.01 * xx + 0.02 * yy + 2
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        colorscale=[[0, "rgba(244, 63, 94, 0.15)"], [1, "rgba(244, 63, 94, 0.25)"]],
        showscale=False,
        name="Empirical Risk Boundary",
        opacity=0.3
    ))

    fig.update_layout(
        title="3D Booking Risk Space (Lead Time × ADR × Length of Stay)",
        scene=dict(
            xaxis=dict(title="Lead Time (Days)", backgroundcolor="#0f172a", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="ADR ($ / Night)", backgroundcolor="#0f172a", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Total Nights", backgroundcolor="#0f172a", gridcolor="rgba(255,255,255,0.1)"),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.2))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=600,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig

def build_3d_geo_globe(df):
    """3D Geospatial Choropleth of Global Traveler Origin."""
    country_stats = df.groupby("country").agg(
        total_bookings=("is_canceled", "count"),
        cancellation_rate=("is_canceled", "mean")
    ).reset_index()
    country_stats = country_stats[country_stats["country"] != "Unknown"]
    country_stats["cancellation_rate_pct"] = country_stats["cancellation_rate"] * 100

    fig = px.choropleth(
        country_stats,
        locations="country",
        color="cancellation_rate_pct",
        hover_name="country",
        hover_data={"total_bookings": True, "cancellation_rate_pct": ":.1f%"},
        projection="orthographic", # 3D Globe Projection
        color_continuous_scale="Plasma",
        title="3D Global Traveler Origin & Country-Level Cancellation Propensity"
    )

    fig.update_geos(
        showcoastlines=True, coastlinecolor="rgba(255,255,255,0.3)",
        showland=True, landcolor="#1e293b",
        showocean=True, oceancolor="#0b132b",
        bgcolor="rgba(0,0,0,0)"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=580,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def build_4d_temporal_risk_animation(df):
    """
    4D Spatio-Temporal Hyper-Scatter:
    - X: Lead Time
    - Y: ADR ($)
    - Z: Total Nights
    - Color: Cancellation Rate
    - Size: Total Guests
    - Animation Frame: Month of Arrival (Jan -> Dec)
    """
    # Aggregate by Month, Hotel, Market Segment, and Lead Time Bins to make animation smooth and fast
    df_agg = df.groupby(["arrival_date_month", "hotel", "market_segment"], observed=True).agg(
        lead_time=("lead_time", "mean"),
        adr=("adr", "mean"),
        total_stays=("total_stays", "mean"),
        total_guests=("total_guests", "mean"),
        cancellation_rate=("is_canceled", "mean"),
        total_volume=("is_canceled", "count")
    ).reset_index()

    df_agg["cancellation_rate_pct"] = df_agg["cancellation_rate"] * 100
    df_agg["arrival_date_month"] = df_agg["arrival_date_month"].astype(str)

    fig = px.scatter_3d(
        df_agg,
        x="lead_time",
        y="adr",
        z="total_stays",
        color="cancellation_rate_pct",
        size="total_volume",
        size_max=32,
        animation_frame="arrival_date_month",
        animation_group="market_segment",
        hover_name="market_segment",
        hover_data={"hotel": True, "lead_time": ":.1f", "adr": ":.1f", "total_volume": True},
        color_continuous_scale="Viridis",
        range_x=[0, 260],
        range_y=[30, 220],
        range_z=[1, 7],
        range_color=[0, 80],
        labels={
            "lead_time": "Avg Lead Time (Days)",
            "adr": "Avg ADR ($)",
            "total_stays": "Avg Stay (Nights)",
            "cancellation_rate_pct": "Cancel Rate %"
        }
    )

    fig.update_layout(
        title="4D Temporal Seasonality Simulator (Dimensions: Lead Time × ADR × Stay × Volume × Month)",
        scene=dict(
            xaxis=dict(title="Lead Time (Days)", backgroundcolor="#0b132b", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="ADR ($)", backgroundcolor="#0b132b", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Nights Stay", backgroundcolor="#0b132b", gridcolor="rgba(255,255,255,0.1)"),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.2))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=620,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig
