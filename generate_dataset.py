import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import json

import argparse

fake = Faker()
np.random.seed(42)
random.seed(42)

class ECommerceDataGenerator:
    def __init__(self):
        self.product_categories = [
            'Electronics', 'Fashion', 'Home & Garden', 'Sports', 'Books',
            'Beauty', 'Jewelry', 'Automotive', 'Toys', 'Health'
        ]
        
        self.return_reasons = [
            'Defective item', 'Wrong size', 'Not as described', 'Changed mind',
            'Damaged in shipping', 'Wrong item sent', 'Quality issues',
            'Empty box received', 'Item used/worn', 'Better price found'
        ]
        
        self.fraud_reasons = [
            'Empty box received', 'Item used/worn', 'Defective item'
        ]
    
    def generate_customers(self, n_customers=1000, fraud_ratio=0.1):
        """Generate customer profiles with fraud propensity"""
        customers = []
        
        for i in range(n_customers):
            # Determine if customer is fraudulent based on desired ratio
            is_fraudster = np.random.random() < fraud_ratio
            
            customer = {
                'customer_id': f'CUST_{i:06d}',
                'registration_date': fake.date_between(start_date='-2y', end_date='-6m'),
                'location': fake.city(),
                'ip_address': fake.ipv4(),
                'is_fraudster': is_fraudster,
                'fraud_type': self._get_fraud_type() if is_fraudster else None
            }
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def _get_fraud_type(self):
        """Assign fraud type to fraudulent customers"""
        fraud_types = ['wardrobing', 'empty_box', 'excessive_returner', 'fake_tracking']
        return np.random.choice(fraud_types)
    
    def generate_products(self, n_products=500):
        """Generate product catalog"""
        products = []
        
        for i in range(n_products):
            category = np.random.choice(self.product_categories)
            
            # Fashion and Jewelry have higher return rates
            base_return_rate = 0.15 if category in ['Fashion', 'Jewelry'] else 0.08
            
            product = {
                'product_id': f'PROD_{i:06d}',
                'category': category,
                'price': np.random.uniform(10, 500),
                'base_return_rate': base_return_rate
            }
            products.append(product)
        
        return pd.DataFrame(products)
    
    def generate_orders_and_returns(self, customers_df, products_df, n_orders=10000):
        """Generate orders and returns with fraud patterns"""
        orders = []
        returns = []
        
        for _ in range(n_orders):
            # Select customer and product
            customer = customers_df.sample(1).iloc[0]
            product = products_df.sample(1).iloc[0]
            
            # Generate order
            order_date = fake.date_between(start_date='-1y', end_date='today')
            delivery_date = order_date + timedelta(days=np.random.randint(1, 7))
            
            order = {
                'order_id': f'ORD_{len(orders):08d}',
                'customer_id': customer['customer_id'],
                'product_id': product['product_id'],
                'order_date': order_date,
                'delivery_date': delivery_date,
                'quantity': np.random.randint(1, 4),
                'total_amount': product['price'] * np.random.randint(1, 4),
                'shipping_address': fake.address(),
                'ip_address': customer['ip_address']
            }
            orders.append(order)
            
            # Determine if this order will be returned
            return_probability = self._calculate_return_probability(customer, product)
            
            if np.random.random() < return_probability:
                return_record = self._generate_return(order, customer, product, delivery_date)
                returns.append(return_record)
        
        return pd.DataFrame(orders), pd.DataFrame(returns)
    
    def _calculate_return_probability(self, customer, product):
        """Calculate return probability based on customer and product"""
        base_prob = product['base_return_rate']
        
        if customer['is_fraudster']:
            if customer['fraud_type'] == 'excessive_returner':
                return min(0.8, base_prob * 4)  # Much higher return rate
            elif customer['fraud_type'] in ['wardrobing', 'empty_box']:
                return min(0.6, base_prob * 2.5)  # Higher return rate
        
        return base_prob
    
    def _generate_return(self, order, customer, product, delivery_date):
        """Generate return record with fraud patterns"""
        # Return timing patterns
        if customer['is_fraudster']:
            if customer['fraud_type'] == 'wardrobing':
                # Return just before policy expires (e.g., 29 days for 30-day policy)
                return_days = np.random.randint(25, 31)
            else:
                return_days = np.random.randint(1, 15)
        else:
            return_days = np.random.exponential(7)  # Normal returns happen sooner
        
        return_date = delivery_date + timedelta(days=int(return_days))
        
        # Return reason patterns
        if customer['is_fraudster']:
            if customer['fraud_type'] == 'empty_box':
                reason = 'Empty box received'
            elif customer['fraud_type'] == 'wardrobing':
                reason = np.random.choice(['Wrong size', 'Not as described', 'Quality issues'])
            else:
                reason = np.random.choice(self.return_reasons)
        else:
            reason = np.random.choice([r for r in self.return_reasons if r not in self.fraud_reasons])
        
        # Return status (fraudulent returns more likely to be rejected)
        if customer['is_fraudster'] and np.random.random() < 0.3:
            status = 'Rejected'
            refund_amount = 0
        else:
            status = 'Approved'
            refund_amount = order['total_amount'] * np.random.uniform(0.8, 1.0)
        
        return {
            'return_id': f'RET_{len(returns) if "returns" in locals() else 0:08d}',
            'order_id': order['order_id'],
            'customer_id': customer['customer_id'],
            'product_id': product['product_id'],
            'return_date': return_date,
            'return_reason': reason,
            'return_status': status,
            'refund_amount': refund_amount,
            'days_to_return': return_days,
            'return_address': fake.address(),
            'tracking_id': fake.uuid4() if np.random.random() > 0.1 else None
        }

    def run(self, n_customers=1000, fraud_ratio=0.1, n_orders=10000, output_dir="."):
        """Generate complete e-commerce dataset with fraud cases"""
        print("Generating E-Commerce Return Fraud Dataset...")
        
        # Generate data
        print("1. Generating customers...")
        customers_df = self.generate_customers(n_customers, fraud_ratio)
        
        print("2. Generating products...")
        products_df = self.generate_products(500)
        
        print("3. Generating orders and returns...")
        orders_df, returns_df = self.generate_orders_and_returns(customers_df, products_df, n_orders)
        
        # Save datasets
        print("4. Saving datasets...")
        customers_df.to_csv(f"{output_dir}/customers.csv", index=False)
        products_df.to_csv(f"{output_dir}/products.csv", index=False)
        orders_df.to_csv(f"{output_dir}/orders.csv", index=False)
        returns_df.to_csv(f"{output_dir}/returns.csv", index=False)
        
        # Generate summary statistics
        print("\n=== Dataset Summary ===")
        print(f"Customers: {len(customers_df):,}")
        print(f"Products: {len(products_df):,}")
        print(f"Orders: {len(orders_df):,}")
        print(f"Returns: {len(returns_df):,}")
        print(f"Return Rate: {len(returns_df)/len(orders_df)*100:.1f}%")
        print(f"Fraudulent Customers: {customers_df['is_fraudster'].sum():,}")
        
        # Fraud analysis
        
    
        
        
        print("\nDataset generation complete!")
        return customers_df, products_df, orders_df, returns_df
    
    

    


    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic e-commerce dataset")
    parser.add_argument("--customers", type=int, default=10000)
    parser.add_argument("--fraud-ratio", type=float, default=0.1)
    parser.add_argument("--orders", type=int, default=10000, help="Total number of orders to simulate")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    gen = ECommerceDataGenerator()
    gen.run(n_customers=args.customers, fraud_ratio=args.fraud_ratio, n_orders=args.orders, output_dir=args.output_dir)
