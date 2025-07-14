import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Online Retail EDA Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the Online Retail dataset"""
    try:
        # Try to load from URL first
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        df = pd.read_excel(url, engine='openpyxl')
    except:
        # If URL fails, create sample data structure
        st.warning("Could not load data from UCI. Please upload your own Online Retail dataset.")
        return None
    
    # Data preprocessing
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    
    # Extract date components
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['Hour'] = df['InvoiceDate'].dt.hour
    
    return df

def display_data_overview(df):
    """Feature 1: Data Overview and Basic Statistics"""
    st.markdown('<div class="feature-header">📊 Feature 1: Data Overview & Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique Customers", f"{df['CustomerID'].nunique():,}")
    with col3:
        st.metric("Unique Products", f"{df['StockCode'].nunique():,}")
    with col4:
        st.metric("Countries", f"{df['Country'].nunique():,}")
    
    # Data quality metrics
    st.subheader("Data Quality Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            fig = px.bar(x=missing_data.index, y=missing_data.values, 
                        title="Missing Values by Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found!")
    
    with col2:
        # Data types
        st.subheader("Data Types")
        st.dataframe(df.dtypes.to_frame('Data Type'), use_container_width=True)
    
    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

def sales_trend_analysis(df):
    """Feature 2: Sales Trend Analysis"""
    st.markdown('<div class="feature-header">📈 Feature 2: Sales Trend Analysis</div>', unsafe_allow_html=True)
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df['InvoiceDate'].min().date())
    with col2:
        end_date = st.date_input("End Date", df['InvoiceDate'].max().date())
    
    # Filter data
    mask = (df['InvoiceDate'].dt.date >= start_date) & (df['InvoiceDate'].dt.date <= end_date)
    filtered_df = df[mask]
    
    # Time aggregation options
    time_agg = st.selectbox("Time Aggregation", ["Daily", "Weekly", "Monthly"])
    
    if time_agg == "Daily":
        time_series = filtered_df.groupby(filtered_df['InvoiceDate'].dt.date).agg({
            'TotalAmount': 'sum',
            'Quantity': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
    elif time_agg == "Weekly":
        time_series = filtered_df.groupby(filtered_df['InvoiceDate'].dt.to_period('W')).agg({
            'TotalAmount': 'sum',
            'Quantity': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        time_series['InvoiceDate'] = time_series['InvoiceDate'].dt.start_time
    else:  # Monthly
        time_series = filtered_df.groupby(filtered_df['InvoiceDate'].dt.to_period('M')).agg({
            'TotalAmount': 'sum',
            'Quantity': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        time_series['InvoiceDate'] = time_series['InvoiceDate'].dt.start_time
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Revenue Trend', 'Quantity Trend', 'Number of Orders'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(go.Scatter(x=time_series['InvoiceDate'], y=time_series['TotalAmount'],
                            mode='lines+markers', name='Revenue'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_series['InvoiceDate'], y=time_series['Quantity'],
                            mode='lines+markers', name='Quantity'), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_series['InvoiceDate'], y=time_series['InvoiceNo'],
                            mode='lines+markers', name='Orders'), row=3, col=1)
    
    fig.update_layout(height=800, title_text=f"{time_agg} Sales Trends")
    st.plotly_chart(fig, use_container_width=True)

def customer_analysis(df):
    """Feature 3: Customer Analysis"""
    st.markdown('<div class="feature-header">👥 Feature 3: Customer Analysis</div>', unsafe_allow_html=True)
    
    # Customer metrics
    customer_metrics = df.groupby('CustomerID').agg({
        'TotalAmount': ['sum', 'mean', 'count'],
        'Quantity': 'sum',
        'InvoiceNo': 'nunique'
    }).round(2)
    
    customer_metrics.columns = ['Total_Spent', 'Avg_Order_Value', 'Total_Orders', 'Total_Quantity', 'Unique_Invoices']
    customer_metrics = customer_metrics.reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top customers by revenue
        top_customers = customer_metrics.nlargest(20, 'Total_Spent')
        fig = px.bar(top_customers, x='CustomerID', y='Total_Spent',
                    title='Top 20 Customers by Revenue')
        fig.update_xaxes(type='category')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer distribution
        fig = px.histogram(customer_metrics, x='Total_Spent', nbins=50,
                          title='Customer Spending Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # RFM Analysis
    st.subheader("RFM Analysis")
    
    # Calculate RFM metrics
    reference_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # RFM Scoring
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
    
    # RFM visualization
    fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                       color='CustomerID', title='RFM 3D Scatter Plot')
    st.plotly_chart(fig, use_container_width=True)

def product_analysis(df):
    """Feature 4: Product Analysis"""
    st.markdown('<div class="feature-header">🛍️ Feature 4: Product Analysis</div>', unsafe_allow_html=True)
    
    # Product performance metrics
    product_metrics = df.groupby(['StockCode', 'Description']).agg({
        'Quantity': 'sum',
        'TotalAmount': 'sum',
        'CustomerID': 'nunique',
        'InvoiceNo': 'nunique'
    }).reset_index()
    
    product_metrics.columns = ['StockCode', 'Description', 'Total_Quantity', 'Total_Revenue', 'Unique_Customers', 'Unique_Orders']
    
    # Analysis type selector
    analysis_type = st.selectbox("Analysis Type", 
                                ["Top Products by Revenue", "Top Products by Quantity", 
                                 "Product Performance Matrix", "Price Analysis"])
    
    if analysis_type == "Top Products by Revenue":
        top_products = product_metrics.nlargest(20, 'Total_Revenue')
        fig = px.bar(top_products, x='Total_Revenue', y='Description',
                    orientation='h', title='Top 20 Products by Revenue')
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Top Products by Quantity":
        top_products = product_metrics.nlargest(20, 'Total_Quantity')
        fig = px.bar(top_products, x='Total_Quantity', y='Description',
                    orientation='h', title='Top 20 Products by Quantity Sold')
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Product Performance Matrix":
        fig = px.scatter(product_metrics, x='Total_Quantity', y='Total_Revenue',
                        size='Unique_Customers', hover_data=['Description'],
                        title='Product Performance Matrix')
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Price Analysis":
        price_analysis = df.groupby('StockCode')['UnitPrice'].agg(['mean', 'std', 'min', 'max']).reset_index()
        fig = px.box(df, x='UnitPrice', title='Unit Price Distribution')
        st.plotly_chart(fig, use_container_width=True)

def geographical_analysis(df):
    """Feature 5: Geographical Analysis"""
    st.markdown('<div class="feature-header">🌍 Feature 5: Geographical Analysis</div>', unsafe_allow_html=True)
    
    # Country-wise analysis
    country_metrics = df.groupby('Country').agg({
        'TotalAmount': 'sum',
        'Quantity': 'sum',
        'CustomerID': 'nunique',
        'InvoiceNo': 'nunique'
    }).reset_index()
    
    country_metrics.columns = ['Country', 'Total_Revenue', 'Total_Quantity', 'Unique_Customers', 'Unique_Orders']
    country_metrics = country_metrics.sort_values('Total_Revenue', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top countries by revenue
        top_countries = country_metrics.head(15)
        fig = px.bar(top_countries, x='Country', y='Total_Revenue',
                    title='Top 15 Countries by Revenue')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer distribution by country
        fig = px.pie(country_metrics.head(10), values='Unique_Customers', names='Country',
                    title='Customer Distribution by Country (Top 10)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Country performance table
    st.subheader("Country Performance Summary")
    st.dataframe(country_metrics, use_container_width=True)

def time_pattern_analysis(df):
    """Feature 6: Time Pattern Analysis"""
    st.markdown('<div class="feature-header">⏰ Feature 6: Time Pattern Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Daily Patterns", "Monthly Patterns", "Hourly Patterns"])
    
    with tab1:
        # Day of week analysis
        dow_analysis = df.groupby('DayOfWeek').agg({
            'TotalAmount': 'sum',
            'Quantity': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_analysis['DayOfWeek'] = pd.Categorical(dow_analysis['DayOfWeek'], categories=day_order, ordered=True)
        dow_analysis = dow_analysis.sort_values('DayOfWeek')
        
        fig = px.bar(dow_analysis, x='DayOfWeek', y='TotalAmount',
                    title='Revenue by Day of Week')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Monthly analysis
        monthly_analysis = df.groupby('Month').agg({
            'TotalAmount': 'sum',
            'Quantity': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        
        fig = px.line(monthly_analysis, x='Month', y='TotalAmount',
                     title='Monthly Revenue Trend', markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Hourly analysis
        hourly_analysis = df.groupby('Hour').agg({
            'TotalAmount': 'sum',
            'Quantity': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        
        fig = px.bar(hourly_analysis, x='Hour', y='TotalAmount',
                    title='Revenue by Hour of Day')
        st.plotly_chart(fig, use_container_width=True)

def advanced_analytics(df):
    """Feature 7: Advanced Analytics"""
    st.markdown('<div class="feature-header">🔬 Feature 7: Advanced Analytics</div>', unsafe_allow_html=True)
    
    analysis_option = st.selectbox("Select Analysis", 
                                  ["Correlation Analysis", "Cohort Analysis", "Market Basket Analysis", "Seasonal Decomposition"])
    
    if analysis_option == "Correlation Analysis":
        # Correlation matrix
        numeric_cols = ['Quantity', 'UnitPrice', 'TotalAmount']
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                       title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_option == "Cohort Analysis":
        # Simplified cohort analysis
        df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
        
        # Get customer's first purchase month
        customer_first_purchase = df.groupby('CustomerID')['InvoiceMonth'].min().reset_index()
        customer_first_purchase.columns = ['CustomerID', 'CohortMonth']
        
        # Merge with original dataframe
        df_cohort = df.merge(customer_first_purchase, on='CustomerID')
        df_cohort['PeriodNumber'] = (df_cohort['InvoiceMonth'] - df_cohort['CohortMonth']).apply(attrgetter('n'))
        
        # Create cohort table
        cohort_data = df_cohort.groupby(['CohortMonth', 'PeriodNumber'])['CustomerID'].nunique().reset_index()
        cohort_sizes = customer_first_purchase.groupby('CohortMonth')['CustomerID'].nunique()
        
        cohort_table = cohort_data.pivot(index='CohortMonth', columns='PeriodNumber', values='CustomerID')
        cohort_table = cohort_table.divide(cohort_sizes, axis=0)
        
        fig = px.imshow(cohort_table, text_auto=True, aspect="auto",
                       title="Cohort Analysis - Customer Retention")
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_option == "Market Basket Analysis":
        # Simple market basket analysis
        basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        # Calculate support for item pairs
        frequent_items = basket.mean().sort_values(ascending=False).head(10)
        
        fig = px.bar(x=frequent_items.index, y=frequent_items.values,
                    title="Top 10 Most Frequent Items")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_option == "Seasonal Decomposition":
        # Time series decomposition
        daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalAmount'].sum().reset_index()
        daily_sales.columns = ['Date', 'Revenue']
        
        # Simple moving average
        daily_sales['MA_7'] = daily_sales['Revenue'].rolling(window=7).mean()
        daily_sales['MA_30'] = daily_sales['Revenue'].rolling(window=30).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['Revenue'], name='Daily Revenue'))
        fig.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['MA_7'], name='7-Day MA'))
        fig.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['MA_30'], name='30-Day MA'))
        fig.update_layout(title='Revenue Trends with Moving Averages')
        st.plotly_chart(fig, use_container_width=True)

def interactive_filters(df):
    """Feature 8: Interactive Filters & Custom Analysis"""
    st.markdown('<div class="feature-header">🎛️ Feature 8: Interactive Filters & Custom Analysis</div>', unsafe_allow_html=True)
    
    # Create filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        countries = st.multiselect("Select Countries", df['Country'].unique(), default=df['Country'].unique()[:5])
    
    with col2:
        date_range = st.date_input("Date Range", 
                                  value=(df['InvoiceDate'].min().date(), df['InvoiceDate'].max().date()),
                                  min_value=df['InvoiceDate'].min().date(),
                                  max_value=df['InvoiceDate'].max().date())
    
    with col3:
        min_price = st.slider("Minimum Unit Price", 0.0, float(df['UnitPrice'].max()), 0.0)
    
    with col4:
        min_quantity = st.slider("Minimum Quantity", 1, int(df['Quantity'].max()), 1)
    
    # Apply filters
    filtered_df = df[
        (df['Country'].isin(countries)) &
        (df['InvoiceDate'].dt.date >= date_range[0]) &
        (df['InvoiceDate'].dt.date <= date_range[1]) &
        (df['UnitPrice'] >= min_price) &
        (df['Quantity'] >= min_quantity)
    ]
    
    # Display filtered results
    st.subheader(f"Filtered Results: {len(filtered_df):,} records")
    
    # Summary metrics for filtered data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${filtered_df['TotalAmount'].sum():,.2f}")
    with col2:
        st.metric("Total Quantity", f"{filtered_df['Quantity'].sum():,}")
    with col3:
        st.metric("Unique Customers", f"{filtered_df['CustomerID'].nunique():,}")
    with col4:
        st.metric("Unique Products", f"{filtered_df['StockCode'].nunique():,}")
    
    # Custom visualization
    viz_type = st.selectbox("Choose Visualization", 
                           ["Revenue by Country", "Top Products", "Customer Distribution", "Time Series"])
    
    if viz_type == "Revenue by Country":
        country_revenue = filtered_df.groupby('Country')['TotalAmount'].sum().reset_index()
        fig = px.bar(country_revenue, x='Country', y='TotalAmount', title='Revenue by Country')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Top Products":
        top_products = filtered_df.groupby('Description')['TotalAmount'].sum().nlargest(10).reset_index()
        fig = px.bar(top_products, x='TotalAmount', y='Description', orientation='h', 
                    title='Top 10 Products by Revenue')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Customer Distribution":
        customer_dist = filtered_df.groupby('CustomerID')['TotalAmount'].sum().reset_index()
        fig = px.histogram(customer_dist, x='TotalAmount', nbins=30, title='Customer Spending Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Time Series":
        daily_revenue = filtered_df.groupby(filtered_df['InvoiceDate'].dt.date)['TotalAmount'].sum().reset_index()
        fig = px.line(daily_revenue, x='InvoiceDate', y='TotalAmount', title='Daily Revenue Trend')
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown('<div class="main-header">🛒 Online Retail EDA Dashboard</div>', unsafe_allow_html=True)
    
    # File upload option
    uploaded_file = st.file_uploader("Upload your Online Retail dataset (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Preprocessing
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
        df = df.dropna(subset=['CustomerID'])
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        
        # Extract date components
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Day'] = df['InvoiceDate'].dt.day
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
        df['Hour'] = df['InvoiceDate'].dt.hour
        
    else:
        df = load_data()
        if df is None:
            st.error("Please upload a dataset to continue.")
            return
    
    # Sidebar for feature selection
    st.sidebar.title("📊 EDA Features")
    
    features = [
        "📊 Data Overview & Statistics",
        "📈 Sales Trend Analysis", 
        "👥 Customer Analysis",
        "🛍️ Product Analysis",
        "🌍 Geographical Analysis",
        "⏰ Time Pattern Analysis",
        "🔬 Advanced Analytics",
        "🎛️ Interactive Filters"
    ]
    
    selected_feature = st.sidebar.selectbox("Select Feature", features)
    
    # Feature routing
    if selected_feature == "📊 Data Overview & Statistics":
        display_data_overview(df)
    elif selected_feature == "📈 Sales Trend Analysis":
        sales_trend_analysis(df)
    elif selected_feature == "👥 Customer Analysis":
        customer_analysis(df)
    elif selected_feature == "🛍️ Product Analysis":
        product_analysis(df)
    elif selected_feature == "🌍 Geographical Analysis":
        geographical_analysis(df)
    elif selected_feature == "⏰ Time Pattern Analysis":
        time_pattern_analysis(df)
    elif selected_feature == "🔬 Advanced Analytics":
        advanced_analytics(df)
    elif selected_feature == "🎛️ Interactive Filters":
        interactive_filters(df)

if __name__ == "__main__":
    main()
