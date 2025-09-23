# superstore_dashboard.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="📊 Superstore Dashboard", layout="wide")

# Load cleaned data
@st.cache_data
def load_data():
    path = r"C:/Users/padmavathi/solar_demo/superstore_data/cleaned_superstore.csv"
    df = pd.read_csv(path, parse_dates=['Order Date', 'Ship Date'])
    df['Order Month'] = df['Order Date'].dt.month_name()
    df['Order Year'] = df['Order Date'].dt.year
    return df

df = load_data()

st.title("📦 Global Superstore Sales Dashboard")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    year = st.multiselect("Select Year", sorted(df['Order Year'].unique()), default=df['Order Year'].unique())
    region = st.multiselect("Select Region", sorted(df['Region'].unique()), default=df['Region'].unique())
    category = st.multiselect("Select Category", sorted(df['Category'].unique()), default=df['Category'].unique())

# Apply filters
filtered = df[
    df['Order Year'].isin(year) &
    df['Region'].isin(region) &
    df['Category'].isin(category)
]

# KPI cards
col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Sales", f"${filtered['Sales'].sum():,.2f}")
col2.metric("📦 Total Orders", f"{len(filtered):,}")
col3.metric("📈 Total Profit", f"${filtered['Profit'].sum():,.2f}")

# Visualizations
st.subheader("🗺️ Regional Sales Overview")
region_sales = filtered.groupby('Region')['Sales'].sum().reset_index()
fig1 = px.bar(region_sales, x='Region', y='Sales', color='Region', title="Sales by Region")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📊 Category-wise Profit")
cat_profit = filtered.groupby('Category')['Profit'].sum().reset_index()
fig2 = px.pie(cat_profit, names='Category', values='Profit', title="Profit Share by Category")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📈 Monthly Sales Trend")
monthly_sales = filtered.groupby(filtered['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
monthly_sales['Order Date'] = monthly_sales['Order Date'].astype(str)
fig3 = px.line(monthly_sales, x='Order Date', y='Sales', title="Monthly Sales Over Time", markers=True)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("⚠️ Discount vs Profit")
fig4 = px.scatter(filtered, x='Discount', y='Profit', color='Category', title="Discount Impact on Profit")
st.plotly_chart(fig4, use_container_width=True)

# Optional: Data download
st.subheader("📥 Download Cleaned Data")
st.download_button("Download CSV", data=filtered.to_csv(index=False), file_name="filtered_superstore.csv", mime="text/csv")
