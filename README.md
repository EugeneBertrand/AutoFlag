# 🛡️ Return Fraud Detection in E-Commerce

A comprehensive machine learning system to detect fraudulent return behavior in e-commerce platforms using behavioral modeling, anomaly detection, and interactive dashboards.

## 🎯 Project Overview

This system identifies common fraud patterns including:
- **Wardrobing**: Buying expensive items, using them, then returning
- **Empty box scams**: Claiming packages arrived empty
- **Excessive returners**: Abusing liberal return policies
- **Fake return tracking**: Uploading fraudulent tracking information

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
```bash
python generate_dataset.py
```

### 3. Run Fraud Detection Analysis
```bash
python fraud_detector.py
```

### 4. Launch Interactive Dashboard
```bash
streamlit run dashboard.py
```

## 📊 Features

### 🔍 Behavioral Feature Engineering
- **Customer-level metrics**: Return rates, timing patterns, category diversity
- **Product-level analysis**: Return rates by product and category
- **Location/IP tracking**: Multi-account detection patterns
- **Financial patterns**: Refund amounts and expensive item returns

### 🤖 Multi-Algorithm Detection
- **Isolation Forest**: Unsupervised anomaly detection
- **DBSCAN Clustering**: Density-based outlier identification  
- **Statistical Analysis**: Z-score based outlier detection
- **Composite Scoring**: Combined fraud risk assessment

### 📈 Interactive Dashboard
- **Real-time monitoring**: Live fraud score tracking
- **Customer deep dive**: Individual behavior analysis
- **Performance metrics**: Model evaluation and tuning
- **Pattern visualization**: Fraud trends and insights

## 🗂️ Dataset Schema

### Customers (`customers.csv`)
| Column | Description |
|--------|-------------|
| customer_id | Unique customer identifier |
| registration_date | Account creation date |
| location | Customer location |
| ip_address | IP address for multi-account detection |
| is_fraudster | Ground truth fraud label |
| fraud_type | Type of fraud (wardrobing, empty_box, etc.) |

### Products (`products.csv`)
| Column | Description |
|--------|-------------|
| product_id | Unique product identifier |
| category | Product category |
| price | Product price |
| base_return_rate | Historical return rate |

### Orders (`orders.csv`)
| Column | Description |
|--------|-------------|
| order_id | Unique order identifier |
| customer_id | Customer who placed order |
| product_id | Product ordered |
| order_date | Purchase date |
| delivery_date | Delivery date |
| total_amount | Order value |

### Returns (`returns.csv`)
| Column | Description |
|--------|-------------|
| return_id | Unique return identifier |
| order_id | Associated order |
| return_date | Return initiation date |
| return_reason | Customer-provided reason |
| return_status | Approved/Rejected |
| refund_amount | Amount refunded |
| days_to_return | Days between delivery and return |

## 🧠 Machine Learning Pipeline

### 1. Feature Engineering
```python
# Customer behavioral features
- return_rate = total_returns / total_orders
- avg_return_days = average time to return
- late_return_rate = returns after 25+ days
- suspicious_reason_rate = suspicious return reasons
- expensive_return_rate = high-value item returns
```

### 2. Anomaly Detection
```python
# Multiple detection methods
isolation_forest = IsolationForest(contamination=0.1)
dbscan = DBSCAN(eps=0.5, min_samples=5)
statistical_outliers = z_score > 3
```

### 3. Fraud Scoring
```python
# Composite fraud score (0-1)
fraud_score = (isolation + dbscan + statistical) / 3
```

## 📊 Performance Metrics

The system evaluates performance using:
- **Precision**: Accuracy of fraud predictions
- **Recall**: Coverage of actual fraud cases  
- **F1-Score**: Balanced precision/recall metric
- **Manual Review**: Top-K suspicious customer inspection

## 🎛️ Dashboard Features

### 🚨 Suspicious Customer Monitoring
- Real-time fraud score ranking
- Risk level categorization (High/Medium/Low)
- Behavioral pattern analysis
- Historical return tracking

### 📈 Analytics & Insights
- Fraud score distribution analysis
- Feature correlation heatmaps
- Return pattern visualization
- Seasonal fraud trends

### 🎯 Model Performance
- Algorithm comparison metrics
- Confusion matrix analysis
- ROC curve visualization
- Threshold optimization

### 🔍 Customer Deep Dive
- Individual customer profiles
- Return timeline analysis
- Reason pattern identification
- Risk factor breakdown

## 🛠️ Advanced Features

### 💡 Policy Simulation
Test impact of policy changes:
- Return window adjustments
- Rejection rate modifications
- Category-specific rules

### 🔤 NLP Analysis
Analyze return reason texts:
- Sentiment analysis
- Keyword clustering
- Suspicious phrase detection

### 🏆 Trust Scoring
Customer reliability metrics:
- Historical behavior scoring
- Risk-adjusted trust levels
- Dynamic threshold adjustment

## 📁 Project Structure

```
return-fraud-detection/
├── generate_dataset.py     # Synthetic data generation
├── fraud_detector.py       # ML detection pipeline
├── dashboard.py            # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── customers.csv          # Generated customer data
├── products.csv           # Generated product catalog
├── orders.csv             # Generated order history
├── returns.csv            # Generated return records
└── fraud_detection_results.csv  # Analysis results
```

## 🚀 Next Steps

1. **Real Data Integration**: Connect to actual e-commerce databases
2. **Real-time Processing**: Implement streaming fraud detection
3. **Advanced ML**: Deep learning models (autoencoders, neural networks)
4. **API Development**: REST API for fraud score queries
5. **Alert System**: Automated notifications for high-risk customers

## 🤝 Contributing

This project demonstrates advanced ML techniques for fraud detection. Feel free to extend with:
- Additional fraud patterns
- New detection algorithms  
- Enhanced visualization features
- Real-world dataset integration

## 📄 License

Open source project for educational and research purposes.
