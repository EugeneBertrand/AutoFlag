import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
import time
from functools import lru_cache
warnings.filterwarnings('ignore')

# Global flag to enable/disable progress bars
SHOW_PROGRESS = True

class ProgressWrapper:
    def __init__(self, iterable, desc=None, total=None, disable=False):
        self.iterable = iterable
        self.desc = desc
        self.total = total
        self.disable = disable or not SHOW_PROGRESS
        self._pbar = None
    
    def __enter__(self):
        if not self.disable:
            self._pbar = tqdm(total=self.total, desc=self.desc, ncols=100)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pbar is not None:
            self._pbar.close()
    
    def update(self, n=1):
        if self._pbar is not None:
            self._pbar.update(n)

class ReturnFraudDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.features = None
        
    def load_data(self, data_dir="."):
        """Load all datasets"""
        import os
        try:
            self.customers = pd.read_csv(os.path.join(data_dir, "customers.csv"))
            self.products = pd.read_csv(os.path.join(data_dir, "products.csv"))
            self.orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
            self.returns = pd.read_csv(os.path.join(data_dir, "returns.csv"))
            
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
        start_time = time.time()
        print("Engineering features...")
        
        # Customer-level behavioral features
        customer_features = self._create_customer_features()
        
        # Product-level features
        product_features = self._create_product_features()
        
        # Merge features
        self.features = customer_features.merge(
            product_features, 
            on='customer_id', 
            how='left'
        ).fillna(0)
        
        # Add memory-efficient dtypes
        self._optimize_dtypes()
        
        elapsed = time.time() - start_time
        print(f"✅ Created {len(self.features.columns)-1} features for {len(self.features):,} customers in {elapsed:.1f}s")
        return self.features
        
    def _optimize_dtypes(self):
        """Optimize DataFrame dtypes for memory efficiency"""
        if self.features is None:
            return
            
        for col in self.features.columns:
            col_type = self.features[col].dtype
            
            # Downcast numeric columns
            if col_type == 'float64':
                self.features[col] = pd.to_numeric(self.features[col], downcast='float')
            elif col_type == 'int64':
                self.features[col] = pd.to_numeric(self.features[col], downcast='integer')
    
    def _create_customer_features(self):
        """Create customer-level behavioral features using vectorized operations"""
        start_time = time.time()
        
        # Pre-merge all data for vectorized operations
        merged = self.orders.merge(
            self.returns, 
            on='order_id', 
            how='left', 
            suffixes=('', '_return')
        )
        
        # Add product info
        merged = merged.merge(
            self.products[['product_id', 'category', 'price']], 
            on='product_id', 
            how='left'
        )
        
        # Calculate days to return if not exists
        if 'days_to_return' not in merged.columns and 'return_date' in merged.columns:
            merged['days_to_return'] = (pd.to_datetime(merged['return_date']) - pd.to_datetime(merged['order_date'])).dt.days
        
        # Calculate expensive threshold once
        expensive_threshold = self.products['price'].quantile(0.75)
        suspicious_reasons = ['Empty box received', 'Item used/worn']
        
        # Group by customer and compute all metrics in one pass
        customer_groups = merged.groupby('customer_id')
        
        # Basic metrics
        result = customer_groups.agg(
            total_orders=pd.NamedAgg(column='order_id', aggfunc='nunique'),
            total_returns=pd.NamedAgg(column='return_id', aggfunc='count'),
            total_spent=pd.NamedAgg(column='total_amount', aggfunc='sum'),
            total_refunded=pd.NamedAgg(column='refund_amount', aggfunc='sum'),
            avg_order_value=pd.NamedAgg(column='total_amount', aggfunc='mean'),
            returned_categories=pd.NamedAgg(column='category', aggfunc=lambda x: x.nunique() if x.notna().any() else 0)
        ).reset_index()
        
        # Calculate return rate
        result['return_rate'] = result['total_returns'] / result['total_orders']
        result['return_rate'].fillna(0, inplace=True)
        
        # Calculate return timing metrics
        if 'days_to_return' in merged.columns:
            return_timing = merged[merged['return_id'].notna()].groupby('customer_id').agg(
                avg_return_days=pd.NamedAgg(column='days_to_return', aggfunc='mean'),
                returns_within_7_days=pd.NamedAgg(
                    column='days_to_return', 
                    aggfunc=lambda x: (x <= 7).sum()
                ),
                late_return_rate=pd.NamedAgg(
                    column='days_to_return',
                    aggfunc=lambda x: (x >= 25).mean() if len(x) > 0 else 0
                )
            )
            result = result.merge(return_timing, on='customer_id', how='left')
            
            # Fill NaN values for customers with no returns
            result['avg_return_days'].fillna(0, inplace=True)
            result['returns_within_7_days'].fillna(0, inplace=True)
            result['late_return_rate'].fillna(0, inplace=True)
        
        # Calculate expensive return rate
        if 'price' in merged.columns:
            expensive_returns = merged[
                (merged['price'] >= expensive_threshold) & 
                (merged['return_id'].notna())
            ].groupby('customer_id').size().reset_index(name='expensive_returns')
            
            result = result.merge(expensive_returns, on='customer_id', how='left')
            result['expensive_returns'].fillna(0, inplace=True)
            result['expensive_return_rate'] = result['expensive_returns'] / result['total_returns']
            result['expensive_return_rate'].fillna(0, inplace=True)
            result.drop('expensive_returns', axis=1, inplace=True)
        
        # Calculate rejection rate
        if 'return_status' in merged.columns:
            rejected = merged[merged['return_status'] == 'Rejected']
            if not rejected.empty:
                rejection_counts = rejected.groupby('customer_id').size().reset_index(name='rejected_returns')
                result = result.merge(rejection_counts, on='customer_id', how='left')
                result['rejected_returns'].fillna(0, inplace=True)
                result['rejection_rate'] = result['rejected_returns'] / result['total_returns']
                result['rejection_rate'].fillna(0, inplace=True)
                result.drop('rejected_returns', axis=1, inplace=True)
        
        # Calculate suspicious reason rate
        if 'return_reason' in merged.columns:
            suspicious = merged[merged['return_reason'].isin(suspicious_reasons)]
            if not suspicious.empty:
                suspicious_counts = suspicious.groupby('customer_id').size().reset_index(name='suspicious_returns')
                result = result.merge(suspicious_counts, on='customer_id', how='left')
                result['suspicious_returns'].fillna(0, inplace=True)
                result['suspicious_reason_rate'] = result['suspicious_returns'] / result['total_returns']
                result['suspicious_reason_rate'].fillna(0, inplace=True)
                result.drop('suspicious_returns', axis=1, inplace=True)
        
        # Ensure all expected columns exist
        expected_columns = {
            'avg_return_days': 0,
            'returns_within_7_days': 0,
            'late_return_rate': 0,
            'expensive_return_rate': 0,
            'rejection_rate': 0,
            'suspicious_reason_rate': 0
        }
        
        for col, default in expected_columns.items():
            if col not in result.columns:
                result[col] = default
        
        # Reorder columns for consistency
        column_order = [
            'customer_id', 'total_orders', 'total_returns', 'return_rate',
            'total_spent', 'total_refunded', 'avg_order_value', 'avg_return_days',
            'returns_within_7_days', 'late_return_rate', 'returned_categories',
            'expensive_return_rate', 'rejection_rate', 'suspicious_reason_rate'
        ]
        
        result = result[[col for col in column_order if col in result.columns]]
        
        elapsed = time.time() - start_time
        print(f"  ✓ Processed customer features in {elapsed:.1f}s")
        
        return result
    
    def _create_product_features(self):
        """Create product-level features aggregated by customer"""
        start_time = time.time()
        
        # Calculate product return stats in one pass
        if not self.returns.empty:
            # Get return counts and average refunds by product
            product_returns = self.returns.groupby('product_id').agg(
                product_return_count=('return_id', 'count'),
                avg_refund=('refund_amount', 'mean')
            ).reset_index()
            
            # Get order counts by product
            product_orders = self.orders.groupby('product_id').size().reset_index(name='product_order_count')
            
            # Merge and calculate return rate
            product_stats = pd.merge(
                product_returns,
                product_orders,
                on='product_id',
                how='right'
            ).fillna(0)
            
            product_stats['product_return_rate'] = (
                product_stats['product_return_count'] / 
                product_stats['product_order_count'].replace(0, np.nan)
            ).fillna(0)
            
            # Merge with orders to get customer-product interactions
            if not self.orders.empty:
                # Merge orders with product stats
                orders_with_returns = pd.merge(
                    self.orders,
                    product_stats[['product_id', 'product_return_rate', 'avg_refund']],
                    on='product_id',
                    how='left'
                )
                
                # Group by customer and calculate mean metrics
                customer_product_features = orders_with_returns.groupby('customer_id').agg(
                    avg_product_return_rate=('product_return_rate', 'mean'),
                    avg_product_refund=('avg_refund', 'mean')
                ).reset_index()
                
                elapsed = time.time() - start_time
                print(f"  ✓ Processed product features in {elapsed:.1f}s")
                
                return customer_product_features
        
        # Return empty DataFrame with expected columns if no data
        return pd.DataFrame(columns=['customer_id', 'avg_product_return_rate', 'avg_product_refund'])
    
    def detect_anomalies(self, sample_size=10000):
        """Apply multiple anomaly detection methods with optimizations"""
        print("Detecting anomalies...")
        start_time = time.time()
        
        # Prepare feature matrix (exclude customer_id and non-numeric columns)
        non_feature_cols = ['customer_id', 'is_fraudster', 'fraud_type']
        feature_cols = [col for col in self.features.columns if col not in non_feature_cols]
        
        # Convert to numpy for faster operations
        X = self.features[feature_cols].fillna(0).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # For large datasets, use a sample for DBSCAN which is O(n²)
        use_sample = len(X_scaled) > sample_size
        
        if use_sample:
            print(f"  Large dataset detected ({len(X_scaled):,} records). Using a sample for DBSCAN...")
            sample_indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
            X_sample = X_scaled[sample_indices]
        
        # 1. Isolation Forest (handles large datasets well)
        print("  Running Isolation Forest...")
        isolation_scores = self.isolation_forest.fit_predict(X_scaled)
        isolation_anomalies = (isolation_scores == -1)
        
        # 2. DBSCAN (expensive, use sample for large datasets)
        print("  Running DBSCAN...")
        if use_sample:
            # Train on sample, predict on full dataset
            self.dbscan.fit(X_sample)
            # Approximate nearest neighbors for prediction on full dataset
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=5).fit(X_sample)
            distances, indices = nbrs.kneighbors(X_scaled)
            # Assign cluster based on nearest neighbor
            dbscan_anomalies = np.array([self.dbscan.labels_[indices[i][0]] == -1 for i in range(len(X_scaled))])
        else:
            # Small dataset, run directly
            cluster_labels = self.dbscan.fit_predict(X_scaled)
            dbscan_anomalies = (cluster_labels == -1)
        
        # 3. Statistical outliers (Z-score > 3)
        print("  Detecting statistical outliers...")
        z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10))  # Add small epsilon to avoid division by zero
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
        if 'total_returns' in self.features.columns:
            zero_return_mask = self.features['total_returns'] == 0
            self.features.loc[zero_return_mask, ['isolation_anomaly', 'dbscan_anomaly', 'statistical_anomaly']] = False
            self.features.loc[zero_return_mask, 'fraud_score'] = 0
        
        # Add actual fraud labels for evaluation if they exist
        if all(col in self.customers.columns for col in ['customer_id', 'is_fraudster', 'fraud_type']):
            self.features = self.features.merge(
                self.customers[['customer_id', 'is_fraudster', 'fraud_type']], 
                on='customer_id',
                how='left'
            )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Completed anomaly detection in {elapsed:.1f}s")
        
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
