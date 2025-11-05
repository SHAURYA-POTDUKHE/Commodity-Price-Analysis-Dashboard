# 📊 Commodity Price Analysis Dashboard

A streamlit interactive dashboard to analyze a trend of commodity prices in 2020-2025 with the exploratory data analysis, time series analysis, geographic analysis, and machine learning models.

**Streamlit Dashboard Link** - https://shaurya-commodity-price-analysis-dashboard.streamlit.app/

## 🚀 Features

- **📈 Overview Dashboard**: Key metrics, statistics, and top commodities
- **📊 EDA & Visualizations**: Distribution analysis, price ranges, heatmaps, and commodity share
- **⏱️ Time Series Analysis**: Yearly and monthly trends with commodity comparisons
- **🗺️ Geographic Analysis**: Market and state-wise price distributions
- **🤖 Machine Learning Models**: 
  - Linear Regression (Simple & Multiple)
  - K-Means Clustering
  - Classification Models (Logistic Regression & Random Forest)
- **🔍 Data Explorer**: Interactive filtering and data export capabilities

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🛠️ Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/akash/Organization/01_Projects/DS_Project_Shaurya
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install packages individually:
   ```bash
   pip install streamlit pandas plotly numpy scikit-learn openpyxl
   ```

## 📂 Required Files

Make sure you have the following file in the project directory:
- `Commodity_Cleaned_2020_2025.csv` - The cleaned dataset (generated from your Jupyter notebook)

If you don't have this file yet, run the data cleaning cells in your `DATA_SCIENCE_CA_1.ipynb` notebook first.

## 🎯 Running the Dashboard

Run the following command in your terminal:

```bash
streamlit run app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

## 📱 Using the Dashboard

### Navigation
Use the sidebar to navigate between different sections:

1. **🏠 Overview**: Obtain a brief overview of the data data using essential metrics.
2. **📈 EDA & Visualizations**: Data distributions chart and analysis.
3. **⏱️ Time Series Analysis**: Analyze price trends over time
4. **🗺️ Geographic Analysis**: Compare prices across markets and states
5. **🤖 Machine Learning Models**: View prediction and clustering results
6. **🔍 Data Explorer**: Filter and download customized data

### Interactive Features
- **Filters**: Choose commodities, date, markets and price ranges.
- **Visualizations**: Hover, zoom, pan Plotly charts.
- **Export**: Download filtered data in CSV
- **Real-time Updates**: The charts are dynamically updated according to your choice.

## 📊 Dashboard Sections

### 1. Overview
- Total records, markets, and commodities.
- Average modal price
- 10 most common commodities frequency.
- Recent data preview

### 2. EDA & Visualizations
- **Distribution Analysis**: Histogram and box plot for price distribution
- **Commodity Share**: Pie and bar charts of market share.
- **Heatmaps**: Years vs Price trends of commodities.
- **Price Ranges**: Min, Max and Modal price comparisons.

### 3. Time Series Analysis
- **Yearly Trends**: The average price trends of 2020-2025.
- **Monthly Trends**: Top commodities monthly analysis.
- **Commodity Comparison**: Comparison of several commodities over time.

### 4. Geographic Analysis
- **Market Analysis**: Price distribution across top markets
- **State Analysis**: State-wise price comparisons (if available)

### 5. Machine Learning Models
- **Regression Models**: 
  - Simple Linear Regression (Max Price → Modal Price)
  - Multiple Linear Regression (Min & Max → Modal Price)
  - R² scores, MSE, and RMSE metrics
- **Clustering Analysis**: 
  - K-Means clustering of commodities
  - PCA visualization
  - Cluster distribution
- **Classification Models**: 
  - Logistic Regression and Random Forest
  - Accuracy scores and confusion matrices

### 6. Data Explorer
- Multi-dimensional filtering
- Interactive data table
- CSV export functionality
- Quick visualizations of filtered data

## 🔧 Customization

### Changing Colors
Edit the color schemes in `app.py`:
```python
color_continuous_scale='Viridis'  # Change to 'Blues', 'Greens', 'Reds', etc.
```

### Adjusting Default Values
Modify slider defaults and filter options in the respective sections.

### Adding New Features
The modular structure makes it easy to add new tabs or visualizations.

## 📝 Notes

- The dashboard uses caching (`@st.cache_data`) for optimal performance
- Large datasets may take a few seconds to load initially
- All visualizations are interactive - hover for details, click-and-drag to zoom
- The app automatically handles missing data and invalid values

## 🐛 Troubleshooting

**Issue**: "Could not find 'Commodity_Cleaned_2020_2025.csv'"
- **Solution**: Run the data cleaning cells in your Jupyter notebook first to generate the file

**Issue**: Package import errors
- **Solution**: Ensure all packages are installed: `pip install -r requirements.txt`

**Issue**: Dashboard not opening
- **Solution**: Check if port 8501 is available, or use: `streamlit run app.py --server.port 8502`

**Issue**: Slow performance
- **Solution**: Filter the data to reduce the number of records being processed

## 📚 Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualizations
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models
- **openpyxl**: Excel file support (for clustering output)

## 💡 Tips

1. Use the sidebar filters to focus on specific commodities or time periods
2. Export filtered data for further analysis in Excel or other tools
3. Hover over charts to see detailed information
4. Data Explorer provides the opportunity to locate the necessary records with ease.
5. Compare different commodities as Time Series.

## 🤝 Contributing

The dashboard can be customized to meet your requirements. Some ideas:
- Strengthen the amount of machine learning models.
- Include weather data correlations
- Add forecasting capabilities
- Create custom reports

## 📄 License

The project is developed as an educational project in a Data Science and Machine Learning course project.

---

**Created with ❤️ using Streamlit**

For questions or issues, please refer to the [Streamlit Documentation](https://docs.streamlit.io/)
# testing
