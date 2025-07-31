import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
from generate_dataset import ECommerceDataGenerator
from fraud_detector import ReturnFraudDetector
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AutoFlag",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: transparent;
        padding: 0;
        border-radius: 0;
    }
    .fraud-alert {
        background-color: transparent;
        padding: 0;
        border-radius: 0;
    }
    .safe-customer {
        background-color: transparent;
        padding: 0;
        border-radius: 0;
    }
</style>
""", unsafe_allow_html=True)

def load_and_process_data(custom_files, synth_params, force_reload=False):
    """Load data and run analysis with session state caching"""
    # Check if we have cached results and don't need to reload
    if not force_reload and 'detector' in st.session_state and 'analysis_results' in st.session_state:
        return st.session_state.detector, st.session_state.analysis_results
    
    # If we get here, we need to (re)load the data
    detector = ReturnFraudDetector()
    
    # Option 1: Generate synthetic dataset on the fly
    if synth_params is not None and synth_params.get('generate', False):
        key = f"{synth_params['customers']}_{synth_params['orders']}_{synth_params['fraud_ratio']}"
        if 'synth_cache_key' not in st.session_state or st.session_state.synth_cache_key != key or force_reload:
            gen = ECommerceDataGenerator()
            c_df, p_df, o_df, r_df = gen.run(
                n_customers=synth_params['customers'],
                fraud_ratio=synth_params['fraud_ratio'],
                n_orders=synth_params['orders']
            )
            st.session_state.synth_cache = (c_df, p_df, o_df, r_df)
            st.session_state.synth_cache_key = key
        else:
            c_df, p_df, o_df, r_df = st.session_state.synth_cache
            
        detector.customers = c_df
        detector.products = p_df
        detector.orders = o_df
        detector.returns = r_df
    
    # Option 2: User-uploaded files
    elif all(v is not None for v in custom_files.values()):
        try:
            detector.customers = pd.read_csv(custom_files['customers'])
            detector.orders = pd.read_csv(custom_files['orders'])
            detector.returns = pd.read_csv(custom_files['returns'])
            detector.products = pd.read_csv(custom_files['products'])
        except Exception as e:
            st.error(f"Failed to read uploaded CSVs: {e}")
            return None, None
    else:
        return None, None
    
    # Run analysis
    with st.spinner("Analyzing data..."):
        # Engineer features and detect anomalies
        detector.engineer_features()
        results = detector.detect_anomalies()
    
    # Cache the results in session state
    st.session_state.detector = detector
    st.session_state.analysis_results = results
    
    return detector, results

def main():
    # Initialize session state if not already done
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.data_loaded = False
    
    st.markdown('<h1 class="main-header">🛡️ AutoFlag</h1>', unsafe_allow_html=True)
    
    # Sidebar upload widgets
    st.sidebar.header("📁 Upload Your Dataset (optional)")
    st.sidebar.markdown("""
    **Expected columns:**
    • `customers.csv` — `customer_id`, `registration_date`
    • `orders.csv` — `order_id`, `customer_id`, `product_id`, `order_date`, `delivery_date`, `refund_amount`
    • `returns.csv` — `return_id`, `order_id`, `return_date`, `return_reason`
    • `products.csv` — `product_id`, `category`, `price`
    """)
    
    # File uploaders
    customers_file = st.sidebar.file_uploader("customers.csv", type=["csv"], key="cust")
    orders_file = st.sidebar.file_uploader("orders.csv", type=["csv"], key="ord")
    returns_file = st.sidebar.file_uploader("returns.csv", type=["csv"], key="ret")
    products_file = st.sidebar.file_uploader("products.csv", type=["csv"], key="prod")

    custom_files = {
        'customers': customers_file,
        'orders': orders_file,
        'returns': returns_file,
        'products': products_file
    }
    
    # Synthetic data generation
    st.sidebar.header("🛠️ Generate Synthetic Data")
    gen_toggle = st.sidebar.toggle("Generate synthetic dataset now")
    synth_params = None
    
    if gen_toggle:
        num_cust = st.sidebar.number_input("# Customers", 1000, 50000, 10000, step=1000)
        num_orders = st.sidebar.number_input("# Orders", 5000, 300000, 60000, step=5000)
        fraud_ratio = st.sidebar.slider("Fraud Ratio", 0.0, 0.5, 0.1, 0.01)
        synth_params = {
            'generate': True,
            'customers': int(num_cust),
            'orders': int(num_orders),
            'fraud_ratio': float(fraud_ratio),
        }
    
    # Add Run Analysis button
    if st.sidebar.button("▶️ Run Analysis"):
        # Validate inputs
        has_upload = all(f is not None for f in [customers_file, orders_file, returns_file, products_file])
        
        if synth_params and has_upload:
            st.error("Please choose either uploaded data OR synthetic generation, not both.")
            st.stop()
            
        if not synth_params and not has_upload:
            st.error("Not all required CSV files have been uploaded. Please provide customers.csv, orders.csv, returns.csv, and products.csv, or choose synthetic generation.")
            st.stop()
        
        # Set flag to indicate we should load data
        st.session_state.should_load_data = True
        st.session_state.synth_params = synth_params
        st.session_state.custom_files = {
            'customers': customers_file,
            'orders': orders_file,
            'returns': returns_file,
            'products': products_file
        }
        st.rerun()
    
    # Check if we should load data
    if st.session_state.get('should_load_data', False):
        with st.spinner("Loading and analyzing data..."):
            try:
                detector, results = load_and_process_data(
                    st.session_state.custom_files,
                    st.session_state.synth_params,
                    force_reload=True
                )
                if detector is not None and results is not None:
                    st.session_state.detector = detector
                    st.session_state.analysis_results = results
                    st.session_state.data_loaded = True
                    st.session_state.should_load_data = False
                    st.rerun()
                else:
                    st.error("Failed to load or process data. Please check your inputs and try again.")
                    st.session_state.should_load_data = False
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.should_load_data = False
        return
    
    # Only show the initial message if we're not in the middle of loading data
    if not st.session_state.get('should_load_data', False) and not st.session_state.data_loaded:
        st.info("⬅️  Upload your CSVs and click Run Analysis, or configure the synthetic generator and click Run Analysis.")
        return
    
    # If we have data loaded, show the dashboard
    if st.session_state.data_loaded and 'detector' in st.session_state and 'analysis_results' in st.session_state:
        detector = st.session_state.detector
        results = st.session_state.analysis_results
    else:
        return
    
    # Offer download if synthetic dataset was generated
    if synth_params is not None and synth_params.get('generate', False):
        st.sidebar.header("⬇️ Download Generated CSVs")
        def _csv_download(df):
            return df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("customers.csv", _csv_download(detector.customers), "customers.csv", "text/csv")
        st.sidebar.download_button("orders.csv", _csv_download(detector.orders), "orders.csv", "text/csv")
        st.sidebar.download_button("returns.csv", _csv_download(detector.returns), "returns.csv", "text/csv")
        if not detector.products.empty:
            st.sidebar.download_button("products.csv", _csv_download(detector.products), "products.csv", "text/csv")

    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Initialize filter values in session state if they don't exist
    if 'fraud_threshold' not in st.session_state:
        st.session_state.fraud_threshold = 0.33
    if 'min_return_rate' not in st.session_state:
        st.session_state.min_return_rate = 0.0
    
    # Update filter values from UI
    new_fraud_threshold = st.sidebar.slider(
        "Fraud Score Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.fraud_threshold, 
        step=0.1,
        help="Customers above this threshold are flagged as suspicious"
    )
    
    new_min_return_rate = st.sidebar.slider(
        "Minimum Return Rate", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.min_return_rate, 
        step=0.1
    )
    
    # Check if filters changed
    filters_changed = (new_fraud_threshold != st.session_state.fraud_threshold or 
                      new_min_return_rate != st.session_state.min_return_rate)
    
    # Update session state if filters changed
    if filters_changed:
        st.session_state.fraud_threshold = new_fraud_threshold
        st.session_state.min_return_rate = new_min_return_rate
        st.rerun()
    
    # Get filter values from session state with defaults
    fraud_threshold = st.session_state.get('fraud_threshold', 0.33)
    min_return_rate = st.session_state.get('min_return_rate', 0.0)
    
    # Filter the results
    filtered_results = results[
        (results['fraud_score'] >= fraud_threshold) &
        (results['return_rate'] >= min_return_rate)
    ]
    
    # DataFrame to display throughout dashboard
    display_df = filtered_results
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Total Customers", 
            f"{len(results):,}",
            help="Total number of customers analyzed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        suspicious_count = len(display_df)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Suspicious Customers", 
            f"{suspicious_count:,}",
            delta=f"{suspicious_count/len(results)*100:.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Always show total confirmed fraudsters regardless of return-rate slider
        actual_fraudsters = results['is_fraudster'].sum()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Actual Fraudsters", 
            f"{actual_fraudsters:,}",
            delta=f"{actual_fraudsters/len(results)*100:.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        avg_return_rate = display_df['return_rate'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Avg Return Rate", 
            f"{avg_return_rate:.1%}",
            help="Average return rate across all customers"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Store filter values in a dict to pass to tabs
    filter_params = {
        'fraud_threshold': fraud_threshold,
        'min_return_rate': min_return_rate
    }
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Suspicious Customers", 
        "📊 Analytics", 
        "🎯 Model Performance", 
        "🔍 Customer Deep Dive",
        "📈 Fraud Patterns"
    ])
    
    with tab1:
        st.header("Most Suspicious Customers")
        
        # Get top suspicious customers
        top_suspicious = display_df.sort_values('fraud_score', ascending=False).head(20)
        
        # Display with color coding
        for idx, customer in top_suspicious.iterrows():
            fraud_score = customer['fraud_score']
            is_actual_fraud = customer['is_fraudster']
            
            # Determine textual risk level but use a neutral card style for all
            if fraud_score >= 0.67:
                risk_level = "🔴 HIGH RISK"
            elif fraud_score >= 0.33:
                risk_level = "🟡 MEDIUM RISK"
            else:
                risk_level = "🟢 LOW RISK"
            # Always use neutral metric card styling (no colored bar)
            card_class = "metric-card"
            
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Customer ID:** {customer['customer_id']}")
                st.write(f"**Risk Level:** {risk_level}")
                if pd.isna(customer['is_fraudster']):
                    st.write("**Confirmed Fraudster:** Unknown")
                elif is_actual_fraud:
                    fraud_type = customer['fraud_type'] if pd.notna(customer['fraud_type']) else "Yes"
                    st.write(f"**⚠️ Confirmed Fraudster:** {fraud_type}")
                else:
                    st.write("**Confirmed Fraudster:** No")
            
            with col2:
                st.metric("Fraud Score", f"{fraud_score:.2f}")
                st.metric("Return Rate", f"{customer['return_rate']:.1%}")
            
            with col3:
                st.metric("Total Returns", f"{customer['total_returns']:.0f}")
                st.metric("Late Returns", f"{customer['late_return_rate']:.1%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
    
    with tab2:
        st.header("Fraud Detection Analytics")
        
        # Fraud score distribution
        fig_dist = px.histogram(
            results, 
            x='fraud_score', 
            color='is_fraudster',
            title="Distribution of Fraud Scores",
            labels={'fraud_score': 'Fraud Score', 'count': 'Number of Customers'},
            color_discrete_map={True: 'red', False: 'blue'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Feature correlation heatmap
        st.subheader("Feature Correlations")
        feature_cols = ['return_rate', 'avg_return_days', 'late_return_rate', 
                       'suspicious_reason_rate', 'expensive_return_rate', 'fraud_score']
        
        corr_matrix = results[feature_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Return patterns by fraud type
        if 'fraud_type' in results.columns:
            fraud_data = results[results['is_fraudster'] == True]
            if len(fraud_data) > 0:
                fig_patterns = px.box(
                    fraud_data,
                    x='fraud_type',
                    y='return_rate',
                    title="Return Rate Patterns by Fraud Type"
                )
                st.plotly_chart(fig_patterns, use_container_width=True)
    
    with tab3:
        st.header("Model Performance Evaluation")
        
        # Calculate performance metrics
        methods = ['isolation_anomaly', 'dbscan_anomaly', 'statistical_anomaly']
        performance_data = []
        
        for method in methods:
            predictions = results[method]
            actual = results['is_fraudster']
            
            tp = ((predictions == True) & (actual == True)).sum()
            fp = ((predictions == True) & (actual == False)).sum()
            tn = ((predictions == False) & (actual == False)).sum()
            fn = ((predictions == False) & (actual == True)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            performance_data.append({
                'Method': method.replace('_', ' ').title(),
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
        
        # Add composite score
        composite_predictions = results['fraud_score'] >= fraud_threshold
        actual = results['is_fraudster']
        
        tp = ((composite_predictions == True) & (actual == True)).sum()
        fp = ((composite_predictions == True) & (actual == False)).sum()
        tn = ((composite_predictions == False) & (actual == False)).sum()
        fn = ((composite_predictions == False) & (actual == True)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        performance_data.append({
            'Method': 'Composite Score',
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        perf_df = pd.DataFrame(performance_data)
        
        # Performance metrics chart
        fig_perf = px.bar(
            perf_df.melt(id_vars=['Method'], var_name='Metric', value_name='Score'),
            x='Method',
            y='Score',
            color='Metric',
            title="Model Performance Comparison",
            barmode='group'
        )
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Performance table
        st.subheader("Detailed Performance Metrics")
        st.dataframe(perf_df.round(3), use_container_width=True)
        
        # Confusion matrix for composite score
        st.subheader("Confusion Matrix (Composite Score)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("True Positives", tp)
            st.metric("False Positives", fp)
        
        with col2:
            st.metric("True Negatives", tn)
            st.metric("False Negatives", fn)
    
    with tab4:
        st.header("Customer Deep Dive")
        
        # Customer selector
        customer_ids = results['customer_id'].tolist()
        selected_customer = st.selectbox("Select Customer ID", customer_ids)
        
        if selected_customer:
            customer_data = results[results['customer_id'] == selected_customer].iloc[0]
            
            # Customer overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Customer Profile")
                st.write(f"**Customer ID:** {customer_data['customer_id']}")
                st.write(f"**Fraud Score:** {customer_data['fraud_score']:.3f}")
                if pd.isna(customer_data['is_fraudster']):
                    st.write("**Actual Fraudster:** Unknown")
                elif customer_data['is_fraudster']:
                    st.write("**Actual Fraudster:** Yes")
                    st.write(f"**Fraud Type:** {customer_data['fraud_type']}")
                else:
                    st.write("**Actual Fraudster:** No")
            
            with col2:
                st.subheader("Behavioral Metrics")
                st.metric("Return Rate", f"{customer_data['return_rate']:.1%}")
                st.metric("Total Orders", f"{customer_data['total_orders']:.0f}")
                st.metric("Total Returns", f"{customer_data['total_returns']:.0f}")
                st.metric("Avg Return Days", f"{customer_data['avg_return_days']:.1f}")
            
            # Customer's order and return history
            customer_orders = detector.orders[detector.orders['customer_id'] == selected_customer]
            customer_returns = detector.returns[detector.returns['customer_id'] == selected_customer]
            
            if len(customer_returns) > 0:
                st.subheader("Return History")
                
                # Timeline of returns
                returns_timeline = customer_returns.copy()
                returns_timeline['return_date'] = pd.to_datetime(returns_timeline['return_date'])
                
                fig_timeline = px.scatter(
                    returns_timeline,
                    x='return_date',
                    y='days_to_return',
                    color='return_status',
                    size=returns_timeline['refund_amount'].fillna(0),
                    hover_data=['return_reason'],
                    title="Return Timeline"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Return reasons
                reason_counts = customer_returns['return_reason'].value_counts()
                fig_reasons = px.pie(
                    values=reason_counts.values,
                    names=reason_counts.index,
                    title="Return Reasons Distribution"
                )
                st.plotly_chart(fig_reasons, use_container_width=True)
    
    with tab5:
        st.header("Fraud Pattern Analysis")
        
        # Fraud patterns over time
        if len(detector.returns) > 0:
            returns_with_fraud = detector.returns.merge(
                results[['customer_id', 'fraud_score']],
                on='customer_id'
            )
            # Treat customers above the threshold as fraudsters for visualisation
            returns_with_fraud['is_fraudster'] = returns_with_fraud['fraud_score'] >= fraud_threshold
            
            returns_with_fraud['return_date'] = pd.to_datetime(returns_with_fraud['return_date'])
            returns_with_fraud['month'] = returns_with_fraud['return_date'].dt.to_period('M')
            
            # Monthly fraud trends
            monthly_fraud = returns_with_fraud.groupby(['month', 'is_fraudster']).size().unstack(fill_value=0)
            
            if True in monthly_fraud.columns and False in monthly_fraud.columns:
                fig_trends = go.Figure()
                fig_trends.add_trace(go.Scatter(
                    x=monthly_fraud.index.astype(str),
                    y=monthly_fraud[True],
                    mode='lines+markers',
                    name='Fraudulent Returns',
                    line=dict(color='red')
                ))
                fig_trends.add_trace(go.Scatter(
                    x=monthly_fraud.index.astype(str),
                    y=monthly_fraud[False],
                    mode='lines+markers',
                    name='Legitimate Returns',
                    line=dict(color='blue')
                ))
                fig_trends.update_layout(title="Monthly Return Trends")
                st.plotly_chart(fig_trends, use_container_width=True)
            
            # Fraud by product category
            # Attach product category for analysis
            if 'category' not in returns_with_fraud.columns and 'product_id' in returns_with_fraud.columns:
                returns_with_fraud = returns_with_fraud.merge(
                    detector.products[['product_id', 'category']],
                    on='product_id', how='left'
                )
            fraud_by_category = returns_with_fraud[returns_with_fraud['is_fraudster'] == True]
            if 'category' in fraud_by_category.columns and len(fraud_by_category) > 0:
                category_fraud = fraud_by_category.groupby('category').size().sort_values(ascending=False)
                
                fig_category = px.bar(
                    x=category_fraud.index,
                    y=category_fraud.values,
                    title="Fraudulent Returns by Product Category"
                )
                st.plotly_chart(fig_category, use_container_width=True)

                # Return timing patterns
                if (
                    'days_to_return' in returns_with_fraud.columns and
                    returns_with_fraud['days_to_return'].notna().any()
                ):
                    fraud_timing = (
                        returns_with_fraud
                        .dropna(subset=['days_to_return'])
                        .groupby('is_fraudster')['days_to_return']
                        .describe()
                    )
                    if not fraud_timing.empty:
                        st.subheader("Return Timing Analysis")
                        st.dataframe(fraud_timing.round(2))
                    else:
                        st.info("Not enough return timing data to compute statistics.")
                else:
                    st.info("Return timing data not available for this dataset.")
            else:
                st.info("No fraudulent returns found in the dataset.")

if __name__ == "__main__":
    main()
