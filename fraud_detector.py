import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class ReturnFraudDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.features = None
        
    def load_data(self):
        """Load all datasets"""
        try:
            self.customers = pd.read_csv('customers.csv')
            self.products = pd.read_csv('products.csv')
            self.orders = pd.read_csv('orders.csv')
            self.returns = pd.read_csv('returns.csv')
            
            # Convert date columns
            date_columns = ['registration_date', 'order_date', 'delivery_date', 'return_date']
            for df_name, df in [('orders', self.orders), ('returns', self.returns), ('customers', self.customers)]:
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
            
            print("Data loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def engineer_features(self):
        """Create behavioral features for fraud detection"""
        print("Engineering features...")
        
        # Customer-level behavioral features
        customer_features = self._create_customer_features()
        
        # Product-level features
        product_features = self._create_product_features()
        
        # Merge features
        self.features = customer_features.merge(
            product_features, 
            left_on='customer_id', 
            right_on='customer_id', 
            how='left'
        ).fillna(0)
        
        print(f"Created {len(self.features.columns)-1} features for {len(self.features)} customers")
        return self.features
    
    def _create_customer_features(self):
        """Create customer-level behavioral features"""
        # Merge orders and returns
        order_return_data = self.orders.merge(
            self.returns, on='order_id', how='left', suffixes=('', '_return')
        )
        
        # Merge with products to get category information
        order_return_data = order_return_data.merge(
            self.products[['product_id', 'category', 'price']], 
            on='product_id', 
            how='left'
        )
        
        # Customer aggregations
        customer_stats = []
        
        for customer_id in self.customers['customer_id'].unique():
            customer_orders = order_return_data[order_return_data['customer_id'] == customer_id]
            customer_returns = customer_orders.dropna(subset=['return_id'])
            
            # Basic stats
            total_orders = len(customer_orders)
            total_returns = len(customer_returns)
            return_rate = total_returns / total_orders if total_orders > 0 else 0
            
            # Financial stats
            total_spent = customer_orders['total_amount'].sum()
            total_refunded = customer_returns['refund_amount'].sum()
            avg_order_value = customer_orders['total_amount'].mean() if total_orders > 0 else 0
            
            # Timing patterns
            if total_returns > 0:
                avg_return_days = customer_returns['days_to_return'].mean()
                returns_within_7_days = (customer_returns['days_to_return'] <= 7).sum()
                returns_after_25_days = (customer_returns['days_to_return'] >= 25).sum()
                late_return_rate = returns_after_25_days / total_returns
            else:
                avg_return_days = 0
                returns_within_7_days = 0
                late_return_rate = 0
            
            # Category diversity
            returned_categories = customer_returns['category'].nunique() if total_returns > 0 else 0
            
            # Expensive item returns (top 25% price range)
            expensive_threshold = self.products['price'].quantile(0.75)
            expensive_returns = customer_returns[
                customer_returns['price'] >= expensive_threshold
            ]
            expensive_return_rate = len(expensive_returns) / total_returns if total_returns > 0 else 0
            
            # Rejection patterns
            rejected_returns = customer_returns[customer_returns['return_status'] == 'Rejected']
            rejection_rate = len(rejected_returns) / total_returns if total_returns > 0 else 0
            
            # Suspicious reasons
            suspicious_reasons = ['Empty box received', 'Item used/worn']
            suspicious_returns = customer_returns[
                customer_returns['return_reason'].isin(suspicious_reasons)
            ]
            suspicious_reason_rate = len(suspicious_returns) / total_returns if total_returns > 0 else 0
            
            customer_stats.append({
                'customer_id': customer_id,
                'total_orders': total_orders,
                'total_returns': total_returns,
                'return_rate': return_rate,
                'total_spent': total_spent,
                'total_refunded': total_refunded,
                'avg_order_value': avg_order_value,
                'avg_return_days': avg_return_days,
                'returns_within_7_days': returns_within_7_days,
                'late_return_rate': late_return_rate,
                'returned_categories': returned_categories,
                'expensive_return_rate': expensive_return_rate,
                'rejection_rate': rejection_rate,
                'suspicious_reason_rate': suspicious_reason_rate
            })
        
        return pd.DataFrame(customer_stats)
    
    def _create_product_features(self):
        """Create product-level features aggregated by customer"""
        # Product return rates
        product_returns = self.returns.groupby('product_id').agg({
            'return_id': 'count',
            'refund_amount': 'mean'
        }).rename(columns={'return_id': 'product_return_count'})
        
        product_orders = self.orders.groupby('product_id').size().rename('product_order_count')
        
        product_stats = pd.concat([product_returns, product_orders], axis=1).fillna(0)
        product_stats['product_return_rate'] = (
            product_stats['product_return_count'] / product_stats['product_order_count']
        ).fillna(0)
        
        # Merge with products for category info
        product_stats = product_stats.merge(
            self.products[['product_id', 'category']], 
            left_index=True, 
            right_on='product_id'
        )
        
        # Customer-product interactions
        customer_product_features = self.orders.merge(
            product_stats, on='product_id'
        ).groupby('customer_id').agg({
            'product_return_rate': 'mean',
            'refund_amount': 'mean'
        }).rename(columns={
            'product_return_rate': 'avg_product_return_rate',
            'refund_amount': 'avg_product_refund'
        })
        
        return customer_product_features.reset_index()
    
    def detect_anomalies(self):
        """Apply multiple anomaly detection methods"""
        print("Detecting anomalies...")
        
        # Prepare feature matrix (exclude customer_id)
        feature_cols = [col for col in self.features.columns if col != 'customer_id']
        X = self.features[feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Isolation Forest
        isolation_scores = self.isolation_forest.fit_predict(X_scaled)
        isolation_anomalies = (isolation_scores == -1)
        
        # 2. DBSCAN Clustering
        cluster_labels = self.dbscan.fit_predict(X_scaled)
        dbscan_anomalies = (cluster_labels == -1)
        
        # 3. Statistical outliers (Z-score > 3)
        z_scores = np.abs((X - X.mean()) / X.std())
        statistical_anomalies = (z_scores > 3).any(axis=1)
        
        # Combine results
        self.features['isolation_anomaly'] = isolation_anomalies
        self.features['dbscan_anomaly'] = dbscan_anomalies
        self.features['statistical_anomaly'] = statistical_anomalies
        
        # Create composite fraud score
        self.features['fraud_score'] = (
            isolation_anomalies.astype(int) + 
            dbscan_anomalies.astype(int) + 
            statistical_anomalies.astype(int)
        ) / 3

        # Keep all customers, but treat those with no returns as non-suspicious
        zero_return_mask = self.features['total_returns'] == 0
        self.features.loc[zero_return_mask, ['isolation_anomaly', 'dbscan_anomaly', 'statistical_anomaly']] = False
        self.features.loc[zero_return_mask, 'fraud_score'] = 0
        
        # Add actual fraud labels for evaluation
        self.features = self.features.merge(
            self.customers[['customer_id', 'is_fraudster', 'fraud_type']], 
            on='customer_id'
        )
        
        return self.features
    
    def get_top_suspicious_customers(self, n=20):
        """Get top N most suspicious customers"""
        suspicious = self.features.nlargest(n, 'fraud_score')[
            ['customer_id', 'fraud_score', 'return_rate', 'total_returns', 
             'late_return_rate', 'suspicious_reason_rate', 'is_fraudster', 'fraud_type']
        ]
        return suspicious
    
    def evaluate_performance(self):
        """Evaluate detection performance against ground truth"""
        if 'is_fraudster' not in self.features.columns:
            print("No ground truth available for evaluation")
            return
        
        # Calculate metrics for each method
        methods = ['isolation_anomaly', 'dbscan_anomaly', 'statistical_anomaly']
        
        print("\n=== Detection Performance ===")
        for method in methods:
            predictions = self.features[method]
            actual = self.features['is_fraudster']
            
            tp = ((predictions == True) & (actual == True)).sum()
            fp = ((predictions == True) & (actual == False)).sum()
            tn = ((predictions == False) & (actual == False)).sum()
            fn = ((predictions == False) & (actual == True)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n{method.replace('_', ' ').title()}:")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
        
        # Composite score performance
        fraud_threshold = 0.33  # At least 1/3 methods flagged as anomaly
        composite_predictions = self.features['fraud_score'] >= fraud_threshold
        actual = self.features['is_fraudster']
        
        tp = ((composite_predictions == True) & (actual == True)).sum()
        fp = ((composite_predictions == True) & (actual == False)).sum()
        tn = ((composite_predictions == False) & (actual == False)).sum()
        fn = ((composite_predictions == False) & (actual == True)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nComposite Fraud Score (threshold={fraud_threshold}):")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")

def main():
    """Run fraud detection pipeline"""
    detector = ReturnFraudDetector()
    
    # Load data
    if not detector.load_data():
        return
    
    # Engineer features
    features = detector.engineer_features()
    
    # Detect anomalies
    results = detector.detect_anomalies()
    
    # Show top suspicious customers
    print("\n=== Top 10 Most Suspicious Customers ===")
    suspicious = detector.get_top_suspicious_customers(10)
    print(suspicious.to_string(index=False))
    
    # Evaluate performance
    detector.evaluate_performance()
    
    # Save results
    results.to_csv('fraud_detection_results.csv', index=False)
    print(f"\nResults saved to fraud_detection_results.csv")

if __name__ == "__main__":
    main()
