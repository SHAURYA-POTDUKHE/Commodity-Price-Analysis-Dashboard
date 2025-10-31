import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Commodity Price Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stMetric {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Commodity_Cleaned_2020_2025.csv")
        df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')
        df['Year'] = df['Arrival_Date'].dt.year
        df['Month'] = df['Arrival_Date'].dt.month_name()
        df['Month_Year'] = df['Arrival_Date'].dt.to_period('M').astype(str)
        return df
    except FileNotFoundError:
        st.error("⚠️ Could not find 'Commodity_Cleaned_2020_2025.csv'. Please ensure the file exists in the project directory.")
        st.stop()

# Load data
df = load_data()

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select a Section:",
    ["🏠 Overview", "📈 EDA & Visualizations", "⏱️ Time Series Analysis", 
     "🗺️ Geographic Analysis", "🤖 Machine Learning Models", "🔍 Data Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Dataset Info")
st.sidebar.info(f"""
**Total Records:** {len(df):,}  
**Commodities:** {df['Commodity'].nunique()}  
**Markets:** {df['Market'].nunique()}  
**Date Range:** {df['Arrival_Date'].min().strftime('%Y-%m-%d')} to {df['Arrival_Date'].max().strftime('%Y-%m-%d')}
""")

# ====================
# 1. OVERVIEW PAGE
# ====================
if page == "🏠 Overview":
    st.title("🏠 Commodity Price Analysis Dashboard")
    st.markdown("### Welcome to the Commodity Price Analytics Platform (2020-2025)")
    
    st.markdown("---")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📦 Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("🌾 Commodities", df['Commodity'].nunique())
    
    with col3:
        st.metric("🏪 Markets", df['Market'].nunique())
    
    with col4:
        st.metric("💰 Avg Modal Price", f"₹{df['Modal_Price'].mean():.2f}")
    
    with col5:
        st.metric("📅 Years Covered", df['Year'].nunique())
    
    st.markdown("---")
    
    # Summary Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Price Statistics")
        stats_df = df[['Min_Price', 'Max_Price', 'Modal_Price']].describe()
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    with col2:
        st.subheader("🔝 Top 10 Commodities by Frequency")
        top_commodities = df['Commodity'].value_counts().head(10).reset_index()
        top_commodities.columns = ['Commodity', 'Count']
        fig = px.bar(top_commodities, x='Count', y='Commodity', orientation='h',
                     color='Count', color_continuous_scale='Viridis',
                     title="Most Traded Commodities")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent Data Preview
    st.subheader("📋 Recent Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ====================
# 2. EDA & VISUALIZATIONS
# ====================
elif page == "📈 EDA & Visualizations":
    st.title("📈 Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution Analysis", "🥧 Commodity Share", "🔥 Heatmaps", "📉 Price Ranges"])
    
    with tab1:
        st.subheader("Distribution of Modal Prices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram with KDE
            fig = px.histogram(df, x='Modal_Price', nbins=50, 
                             title="Modal Price Distribution",
                             labels={'Modal_Price': 'Modal Price (₹)'},
                             color_discrete_sequence=['#636EFA'])
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df, y='Modal_Price', 
                        title="Modal Price - Box Plot (Outlier Detection)",
                        labels={'Modal_Price': 'Modal Price (₹)'},
                        color_discrete_sequence=['#EF553B'])
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("### 📊 Statistical Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"₹{df['Modal_Price'].mean():.2f}")
        with col2:
            st.metric("Median", f"₹{df['Modal_Price'].median():.2f}")
        with col3:
            st.metric("Std Dev", f"₹{df['Modal_Price'].std():.2f}")
        with col4:
            st.metric("Range", f"₹{df['Modal_Price'].max() - df['Modal_Price'].min():.2f}")
    
    with tab2:
        st.subheader("🥧 Commodity Market Share")
        
        top_n = st.slider("Select number of top commodities to display:", 5, 20, 8)
        top_commodities = df['Commodity'].value_counts().head(top_n)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie Chart
            fig = px.pie(values=top_commodities.values, names=top_commodities.index,
                        title=f"Top {top_n} Commodities Share",
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar Chart
            fig = px.bar(x=top_commodities.index, y=top_commodities.values,
                        labels={'x': 'Commodity', 'y': 'Count'},
                        title=f"Top {top_n} Commodities by Frequency",
                        color=top_commodities.values,
                        color_continuous_scale='Viridis')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔥 Commodity vs Year Heatmap")
        
        # Pivot table
        pivot = df.pivot_table(index='Commodity', columns='Year', values='Modal_Price', aggfunc='mean')
        
        # Limit to top commodities for better visualization
        top_commodities = df['Commodity'].value_counts().head(20).index
        pivot_filtered = pivot.loc[pivot.index.isin(top_commodities)]
        
        fig = px.imshow(pivot_filtered, 
                       labels=dict(x="Year", y="Commodity", color="Avg Modal Price (₹)"),
                       title="Average Modal Price Heatmap: Top 20 Commodities (2020-2025)",
                       color_continuous_scale='YlGnBu',
                       aspect="auto")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📉 Price Range Analysis")
        
        selected_commodity = st.selectbox("Select Commodity:", sorted(df['Commodity'].unique()))
        
        commodity_data = df[df['Commodity'] == selected_commodity]
        
        # Price range over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=commodity_data['Arrival_Date'], 
            y=commodity_data['Max_Price'],
            name='Max Price',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=commodity_data['Arrival_Date'], 
            y=commodity_data['Modal_Price'],
            name='Modal Price',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=commodity_data['Arrival_Date'], 
            y=commodity_data['Min_Price'],
            name='Min Price',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title=f"Price Range Over Time: {selected_commodity}",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics for selected commodity
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Modal Price", f"₹{commodity_data['Modal_Price'].mean():.2f}")
        with col2:
            st.metric("Max Price Peak", f"₹{commodity_data['Max_Price'].max():.2f}")
        with col3:
            st.metric("Min Price Low", f"₹{commodity_data['Min_Price'].min():.2f}")
        with col4:
            st.metric("Total Records", f"{len(commodity_data):,}")

# ====================
# 3. TIME SERIES ANALYSIS
# ====================
elif page == "⏱️ Time Series Analysis":
    st.title("⏱️ Time Series Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📅 Yearly Trends", "📆 Monthly Trends", "🔄 Commodity Comparison"])
    
    with tab1:
        st.subheader("📅 Yearly Average Modal Price Trend")
        
        yearly_avg = df.groupby('Year')['Modal_Price'].mean().reset_index()
        
        fig = px.line(yearly_avg, x='Year', y='Modal_Price', 
                     markers=True,
                     title="Average Modal Price Trend (2020-2025)",
                     labels={'Modal_Price': 'Average Modal Price (₹)', 'Year': 'Year'},
                     color_discrete_sequence=['#00CC96'])
        fig.update_traces(line=dict(width=3), marker=dict(size=10))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Year-over-year change
        st.markdown("### 📊 Year-over-Year Change")
        yearly_avg['YoY_Change'] = yearly_avg['Modal_Price'].pct_change() * 100
        yearly_avg['YoY_Change'] = yearly_avg['YoY_Change'].fillna(0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(yearly_avg.style.format({
                'Modal_Price': '₹{:.2f}',
                'YoY_Change': '{:.2f}%'
            }), use_container_width=True)
        
        with col2:
            fig = px.bar(yearly_avg, x='Year', y='YoY_Change',
                        title="Year-over-Year % Change",
                        labels={'YoY_Change': 'Change (%)', 'Year': 'Year'},
                        color='YoY_Change',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📆 Monthly Trends for Top Commodities")
        
        top_n = st.slider("Number of top commodities:", 3, 10, 5, key="monthly_top")
        top_commodities = df['Commodity'].value_counts().head(top_n).index
        df_top = df[df['Commodity'].isin(top_commodities)]
        
        monthly_avg = df_top.groupby(['Month_Year', 'Commodity'])['Modal_Price'].mean().reset_index()
        
        fig = px.line(monthly_avg, x='Month_Year', y='Modal_Price', 
                     color='Commodity',
                     markers=True,
                     title=f"Monthly Average Modal Price: Top {top_n} Commodities",
                     labels={'Modal_Price': 'Modal Price (₹)', 'Month_Year': 'Month-Year'})
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔄 Compare Commodities Over Time")
        
        all_commodities = sorted(df['Commodity'].unique())
        selected_commodities = st.multiselect(
            "Select commodities to compare:",
            all_commodities,
            default=all_commodities[:3] if len(all_commodities) >= 3 else all_commodities
        )
        
        if selected_commodities:
            df_selected = df[df['Commodity'].isin(selected_commodities)]
            
            # Aggregate by month-year
            comparison_data = df_selected.groupby(['Month_Year', 'Commodity'])['Modal_Price'].mean().reset_index()
            
            fig = px.line(comparison_data, x='Month_Year', y='Modal_Price',
                         color='Commodity', markers=True,
                         title="Commodity Price Comparison Over Time",
                         labels={'Modal_Price': 'Modal Price (₹)', 'Month_Year': 'Month-Year'})
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.markdown("### 📊 Comparison Statistics")
            summary = df_selected.groupby('Commodity')['Modal_Price'].agg(['mean', 'std', 'min', 'max']).reset_index()
            summary.columns = ['Commodity', 'Mean', 'Std Dev', 'Min', 'Max']
            st.dataframe(summary.style.format({
                'Mean': '₹{:.2f}',
                'Std Dev': '₹{:.2f}',
                'Min': '₹{:.2f}',
                'Max': '₹{:.2f}'
            }), use_container_width=True)
        else:
            st.warning("Please select at least one commodity to compare.")

# ====================
# 4. GEOGRAPHIC ANALYSIS
# ====================
elif page == "🗺️ Geographic Analysis":
    st.title("🗺️ Geographic Analysis")
    
    tab1, tab2 = st.tabs(["🏛️ Market Analysis", "🗺️ State Analysis"])
    
    with tab1:
        st.subheader("🏛️ Market-wise Price Distribution")
        
        selected_commodity = st.selectbox("Select Commodity:", sorted(df['Commodity'].unique()), key="market_commodity")
        
        market_data = df[df['Commodity'] == selected_commodity]
        top_markets = market_data['Market'].value_counts().head(15).index
        market_data_filtered = market_data[market_data['Market'].isin(top_markets)]
        
        fig = px.box(market_data_filtered, x='Market', y='Modal_Price',
                    title=f"Market-wise Modal Price Distribution: {selected_commodity}",
                    labels={'Modal_Price': 'Modal Price (₹)', 'Market': 'Market'},
                    color='Market')
        fig.update_layout(height=500, xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Market statistics
        st.markdown("### 📊 Top 10 Markets by Average Price")
        market_stats = market_data_filtered.groupby('Market')['Modal_Price'].agg(['mean', 'count']).reset_index()
        market_stats.columns = ['Market', 'Avg Price', 'Records']
        market_stats = market_stats.sort_values('Avg Price', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(market_stats.style.format({
                'Avg Price': '₹{:.2f}',
                'Records': '{:,.0f}'
            }), use_container_width=True)
        
        with col2:
            fig = px.bar(market_stats, x='Avg Price', y='Market', orientation='h',
                        title="Top 10 Markets by Average Price",
                        color='Avg Price',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🗺️ State-wise Analysis")
        
        # Check if State column exists
        if 'State' in df.columns:
            selected_commodity_state = st.selectbox("Select Commodity:", sorted(df['Commodity'].unique()), key="state_commodity")
            
            state_data = df[df['Commodity'] == selected_commodity_state]
            
            fig = px.box(state_data, x='State', y='Modal_Price',
                        title=f"State-wise Modal Price Distribution: {selected_commodity_state}",
                        labels={'Modal_Price': 'Modal Price (₹)', 'State': 'State'},
                        color='State')
            fig.update_layout(height=500, xaxis_tickangle=-90, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # State statistics
            state_stats = state_data.groupby('State')['Modal_Price'].agg(['mean', 'count']).reset_index()
            state_stats.columns = ['State', 'Avg Price', 'Records']
            state_stats = state_stats.sort_values('Avg Price', ascending=False).head(15)
            
            st.markdown("### 📊 Top 15 States by Average Price")
            st.dataframe(state_stats.style.format({
                'Avg Price': '₹{:.2f}',
                'Records': '{:,.0f}'
            }), use_container_width=True)
        else:
            st.info("State information is not available in the dataset.")
            
            # Alternative: Market-based geographic insights
            st.markdown("### 🏪 Market Distribution")
            market_counts = df['Market'].value_counts().head(20).reset_index()
            market_counts.columns = ['Market', 'Count']
            
            fig = px.bar(market_counts, x='Count', y='Market', orientation='h',
                        title="Top 20 Markets by Trading Volume",
                        color='Count',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

# ====================
# 5. MACHINE LEARNING MODELS
# ====================
elif page == "🤖 Machine Learning Models":
    st.title("🤖 Machine Learning Models")
    
    tab1, tab2, tab3 = st.tabs(["📈 Regression Models", "🔄 Clustering Analysis", "🎯 Classification Models"])
    
    with tab1:
        st.subheader("📈 Linear Regression Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Simple Linear Regression")
            st.markdown("**Predicting Modal Price from Max Price**")
            
            # Prepare data
            X_simple = df[['Max_Price']].dropna()
            y_simple = df.loc[X_simple.index, 'Modal_Price']
            
            X_train, X_test, y_train, y_test = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)
            
            # Train model
            slr_model = LinearRegression()
            slr_model.fit(X_train, y_train)
            y_pred = slr_model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("MSE", f"{mse:.2f}")
            st.metric("RMSE", f"{np.sqrt(mse):.2f}")
            
            st.markdown(f"**Equation:** Modal Price = {slr_model.intercept_:.2f} + {slr_model.coef_[0]:.4f} × Max Price")
            
            # Visualization
            fig = px.scatter(x=X_test['Max_Price'], y=y_test, 
                           labels={'x': 'Max Price (₹)', 'y': 'Actual Modal Price (₹)'},
                           title="Simple Linear Regression: Actual vs Predicted",
                           opacity=0.5)
            fig.add_scatter(x=X_test['Max_Price'], y=y_pred, mode='markers',
                          name='Predicted', marker=dict(color='red', size=3))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Multiple Linear Regression")
            st.markdown("**Predicting Modal Price from Min & Max Price**")
            
            # Prepare data
            X_multi = df[['Min_Price', 'Max_Price']].dropna()
            y_multi = df.loc[X_multi.index, 'Modal_Price']
            
            X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
            
            # Train model
            mlr_model = LinearRegression()
            mlr_model.fit(X_train, y_train)
            y_pred = mlr_model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("MSE", f"{mse:.2f}")
            st.metric("RMSE", f"{np.sqrt(mse):.2f}")
            
            st.markdown(f"**Equation:** Modal Price = {mlr_model.intercept_:.2f} + {mlr_model.coef_[0]:.4f} × Min + {mlr_model.coef_[1]:.4f} × Max")
            
            # Actual vs Predicted plot
            results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            fig = px.scatter(results_df, x='Actual', y='Predicted',
                           title="Multiple Linear Regression: Actual vs Predicted",
                           labels={'Actual': 'Actual Modal Price (₹)', 'Predicted': 'Predicted Modal Price (₹)'},
                           opacity=0.5)
            # Add perfect prediction line
            min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
            max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
            fig.add_scatter(x=[min_val, max_val], y=[min_val, max_val], 
                          mode='lines', name='Perfect Prediction', 
                          line=dict(color='red', dash='dash'))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔄 K-Means Clustering Analysis")
        st.markdown("**Clustering commodities based on yearly price trends (2020-2025)**")
        
        # Prepare data
        pivot = df.pivot_table(index='Commodity', columns='Year', values='Modal_Price', aggfunc='mean').fillna(0)
        
        # Standardization
        scaler = StandardScaler()
        pivot_scaled = scaler.fit_transform(pivot)
        
        # K-Means
        n_clusters = st.slider("Number of Clusters:", 2, 8, 4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        pivot['Cluster'] = kmeans.fit_predict(pivot_scaled)
        
        # PCA for 2D visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pivot_scaled)
        pivot['PCA1'] = pca_result[:, 0]
        pivot['PCA2'] = pca_result[:, 1]
        
        # Reset index to get Commodity as column
        pivot_reset = pivot.reset_index()
        
        # Visualization
        fig = px.scatter(pivot_reset, x='PCA1', y='PCA2', color='Cluster',
                        hover_data=['Commodity'],
                        title=f"K-Means Clustering: {n_clusters} Clusters (PCA Visualization)",
                        labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
                        color_continuous_scale='Viridis')
        fig.update_traces(marker=dict(size=10, opacity=0.7))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster statistics
        st.markdown("### 📊 Cluster Distribution")
        cluster_counts = pivot['Cluster'].value_counts().sort_index().reset_index()
        cluster_counts.columns = ['Cluster', 'Number of Commodities']
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(cluster_counts, use_container_width=True)
        
        with col2:
            fig = px.pie(cluster_counts, values='Number of Commodities', names='Cluster',
                        title="Cluster Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show commodities in each cluster
        selected_cluster = st.selectbox("View commodities in cluster:", sorted(pivot['Cluster'].unique()))
        commodities_in_cluster = pivot_reset[pivot_reset['Cluster'] == selected_cluster]['Commodity'].tolist()
        st.markdown(f"**Commodities in Cluster {selected_cluster}:** ({len(commodities_in_cluster)} items)")
        st.write(", ".join(commodities_in_cluster[:20]) + ("..." if len(commodities_in_cluster) > 20 else ""))
    
    with tab3:
        st.subheader("🎯 Classification Models")
        st.markdown("**Classifying commodities into High/Low price categories**")
        
        # Feature engineering
        avg_price = df.groupby('Commodity')['Modal_Price'].mean().reset_index()
        threshold = avg_price['Modal_Price'].median()
        avg_price['Price_Category'] = avg_price['Modal_Price'].apply(lambda x: 1 if x >= threshold else 0)
        
        df_class = df.merge(avg_price[['Commodity', 'Price_Category']], on='Commodity', how='left')
        
        # Encode categorical features
        label_cols = ['Commodity', 'Variety', 'Grade', 'Market']
        df_encoded = df_class.copy()
        
        for col in label_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        # Prepare features
        feature_cols = [col for col in ['Commodity', 'Variety', 'Grade', 'Market', 'Min_Price', 'Max_Price', 'Year'] if col in df_encoded.columns]
        X = df_encoded[feature_cols].dropna()
        y = df_encoded.loc[X.index, 'Price_Category']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Logistic Regression")
            
            # Train model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            log_model = LogisticRegression(random_state=42, max_iter=1000)
            log_model.fit(X_train_scaled, y_train)
            y_pred_log = log_model.predict(X_test_scaled)
            
            # Metrics
            acc_log = accuracy_score(y_test, y_pred_log)
            st.metric("Accuracy", f"{acc_log:.4f}")
            
            # Confusion Matrix
            cm_log = confusion_matrix(y_test, y_pred_log)
            fig = px.imshow(cm_log, 
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=['Low Price', 'High Price'],
                          y=['Low Price', 'High Price'],
                          title="Confusion Matrix - Logistic Regression",
                          color_continuous_scale='Blues',
                          text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Random Forest Classifier")
            
            # Train model
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            
            # Metrics
            acc_rf = accuracy_score(y_test, y_pred_rf)
            st.metric("Accuracy", f"{acc_rf:.4f}")
            
            # Confusion Matrix
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            fig = px.imshow(cm_rf, 
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=['Low Price', 'High Price'],
                          y=['Low Price', 'High Price'],
                          title="Confusion Matrix - Random Forest",
                          color_continuous_scale='Greens',
                          text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        st.markdown("### 🏆 Model Comparison")
        comparison_df = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest'],
            'Accuracy': [acc_log, acc_rf]
        })
        
        fig = px.bar(comparison_df, x='Model', y='Accuracy',
                    title="Classification Model Accuracy Comparison",
                    color='Accuracy',
                    color_continuous_scale='Viridis',
                    text='Accuracy')
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

# ====================
# 6. DATA EXPLORER
# ====================
elif page == "🔍 Data Explorer":
    st.title("🔍 Data Explorer")
    
    st.markdown("### 🔎 Filter and Explore the Dataset")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        commodities_filter = st.multiselect(
            "Select Commodities:",
            options=sorted(df['Commodity'].unique()),
            default=[]
        )
    
    with col2:
        years_filter = st.multiselect(
            "Select Years:",
            options=sorted(df['Year'].unique()),
            default=[]
        )
    
    with col3:
        markets_filter = st.multiselect(
            "Select Markets:",
            options=sorted(df['Market'].unique()),
            default=[]
        )
    
    # Price range filter
    col1, col2 = st.columns(2)
    with col1:
        min_price_filter = st.number_input("Minimum Modal Price (₹):", value=float(df['Modal_Price'].min()), step=100.0)
    with col2:
        max_price_filter = st.number_input("Maximum Modal Price (₹):", value=float(df['Modal_Price'].max()), step=100.0)
    
    # Apply filters
    filtered_df = df.copy()
    
    if commodities_filter:
        filtered_df = filtered_df[filtered_df['Commodity'].isin(commodities_filter)]
    if years_filter:
        filtered_df = filtered_df[filtered_df['Year'].isin(years_filter)]
    if markets_filter:
        filtered_df = filtered_df[filtered_df['Market'].isin(markets_filter)]
    
    filtered_df = filtered_df[(filtered_df['Modal_Price'] >= min_price_filter) & 
                              (filtered_df['Modal_Price'] <= max_price_filter)]
    
    # Display results
    st.markdown(f"### 📊 Filtered Results: {len(filtered_df):,} records")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Modal Price", f"₹{filtered_df['Modal_Price'].mean():.2f}")
    with col2:
        st.metric("Unique Commodities", filtered_df['Commodity'].nunique())
    with col3:
        st.metric("Unique Markets", filtered_df['Market'].nunique())
    with col4:
        st.metric("Date Range", f"{filtered_df['Year'].min()}-{filtered_df['Year'].max()}")
    
    # Data table
    st.markdown("### 📋 Data Table")
    st.dataframe(filtered_df.sort_values('Arrival_Date', ascending=False), use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f'filtered_commodity_data_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv',
    )
    
    # Quick visualizations
    if len(filtered_df) > 0:
        st.markdown("### 📊 Quick Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(filtered_df['Commodity'].unique()) > 0:
                commodity_dist = filtered_df['Commodity'].value_counts().head(10)
                fig = px.bar(x=commodity_dist.values, y=commodity_dist.index, orientation='h',
                           title="Top 10 Commodities in Filtered Data",
                           labels={'x': 'Count', 'y': 'Commodity'},
                           color=commodity_dist.values,
                           color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(filtered_df['Year'].unique()) > 0:
                yearly_trend = filtered_df.groupby('Year')['Modal_Price'].mean().reset_index()
                fig = px.line(yearly_trend, x='Year', y='Modal_Price',
                            markers=True,
                            title="Average Modal Price Trend",
                            labels={'Modal_Price': 'Avg Modal Price (₹)'})
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Commodity Price Analysis Dashboard | Data Science Project 2020-2025</p>
    </div>
    """, unsafe_allow_html=True)
