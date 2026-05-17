import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# 讀取資料
# -------------------------
df = pd.read_csv("Airline_Delay_Cause.csv")

# -------------------------
# 網頁標題
# -------------------------
st.title("✈️ US Flight Delay AI Dashboard")

st.markdown("### Aviation Delay Analysis and Prediction System")

# -------------------------
# Sidebar 篩選器
# -------------------------
st.sidebar.header("Filter Options")

# 年份選單
selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["year"].unique())
)

# 航空公司選單
selected_airline = st.sidebar.selectbox(
    "Select Airline",
    sorted(df["carrier_name"].unique())
)

# -------------------------
# 資料篩選
# -------------------------
filtered_df = df[
    (df["year"] == selected_year) &
    (df["carrier_name"] == selected_airline)
]
# =========================
# Dataset Overview
# =========================

st.subheader("📁 Dataset Overview")

st.write("### Dataset Shape")
st.write(df.shape)

st.write("### Data Types")
st.write(df.dtypes)

st.write("### Missing Values")
st.write(df.isnull().sum())

st.write("### Statistical Summary")
st.write(df.describe())
# -------------------------
# KPI 指標
# -------------------------
st.subheader("📊 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Flights",
    int(filtered_df["arr_flights"].sum())
)

col2.metric(
    "Average Delay",
    round(filtered_df["arr_delay"].mean(), 2)
)

col3.metric(
    "Cancelled Flights",
    int(filtered_df["arr_cancelled"].sum())
)

# -------------------------
# 顯示資料表
# -------------------------
st.subheader("📁 Dataset Preview")
st.write(filtered_df.head())

# -------------------------
# 延誤原因分析
# -------------------------
st.subheader("📌 Delay Cause Analysis")

delay_causes = {
    "Carrier Delay": filtered_df["carrier_delay"].sum(),
    "Weather Delay": filtered_df["weather_delay"].sum(),
    "NAS Delay": filtered_df["nas_delay"].sum(),
    "Security Delay": filtered_df["security_delay"].sum(),
    "Late Aircraft Delay": filtered_df["late_aircraft_delay"].sum()
}

delay_df = pd.DataFrame({
    "Delay Type": delay_causes.keys(),
    "Minutes": delay_causes.values()
})

fig_pie = px.pie(
    delay_df,
    names="Delay Type",
    values="Minutes",
    title="Delay Cause Distribution"
)

st.plotly_chart(fig_pie)

# -------------------------
# 航空公司平均延誤圖
# -------------------------
st.subheader("📈 Airline Average Delay")

avg_delay = (
    df.groupby("carrier_name")["arr_delay"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig_bar = px.bar(
    avg_delay,
    x=avg_delay.index,
    y=avg_delay.values,
    title="Top 10 Airlines by Average Delay",
    labels={
        "x": "Airline",
        "y": "Average Delay"
    }
)
# =========================
# Future Delay Forecast
# =========================

st.subheader("📈 Future Delay Forecast")

forecast_data = (
    df.groupby("month")["arr_delay"]
    .mean()
    .reset_index()
)

fig_forecast = px.line(
    forecast_data,
    x="month",
    y="arr_delay",
    markers=True,
    title="Future Monthly Delay Trend Forecast"
)

st.plotly_chart(fig_forecast)
# =========================
# AI Prediction Simulator
# =========================

st.subheader("🤖 AI Flight Delay Simulator")

carrier_input = st.slider(
    "Carrier Delay",
    0,
    5000,
    100
)

weather_input = st.slider(
    "Weather Delay",
    0,
    5000,
    100
)

nas_input = st.slider(
    "NAS Delay",
    0,
    5000,
    100
)

late_input = st.slider(
    "Late Aircraft Delay",
    0,
    5000,
    100
)

prediction_score = (
    carrier_input * 0.25 +
    weather_input * 0.25 +
    nas_input * 0.25 +
    late_input * 0.25
)

st.metric(
    "Predicted Delay Risk Score",
    round(prediction_score, 2)
)

if prediction_score > 2500:
    st.error("⚠️ Severe Delay Risk")
elif prediction_score > 1000:
    st.warning("🟠 Medium Delay Risk")
else:
    st.success("🟢 Low Delay Risk")
    # =========================
# Delay Heatmap
# =========================

st.subheader("🔥 Monthly Delay Heatmap")

heatmap_data = df.pivot_table(
    values="arr_delay",
    index="carrier_name",
    columns="month",
    aggfunc="mean"
)

fig_heatmap = px.imshow(
    heatmap_data,
    labels=dict(
        x="Month",
        y="Airline",
        color="Average Delay"
    ),
    aspect="auto",
    title="Airline Delay Heatmap"
)

st.plotly_chart(fig_heatmap)
# =========================
# Airport Delay Ranking
# =========================

st.subheader("🌎 Top Delay Airports")

airport_delay = (
    df.groupby("airport_name")["arr_delay"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig_airport = px.bar(
    airport_delay,
    x=airport_delay.values,
    y=airport_delay.index,
    orientation="h",
    title="Top 10 Airports by Average Delay",
    labels={
        "x": "Average Delay",
        "y": "Airport"
    }
)

st.plotly_chart(fig_airport)