"""Preprocess the Olist Brazilian E-commerce Public Dataset
and emit the four unified CSVs expected by the fraud-detection
pipeline: customers.csv, products.csv, orders.csv, returns.csv.

Usage:
    python olist_preprocessor.py --data-dir ./data/olist

Expect the following original files inside --data-dir:
    olist_orders_dataset.csv
    olist_order_items_dataset.csv
    olist_customers_dataset.csv
    olist_products_dataset.csv
    olist_order_reviews_dataset.csv

These filenames are exactly as provided by Kaggle after extraction.
The script produces cleaned CSVs in the project root (or --out-dir).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def load_raw_frames(data_dir: Path):
    """Load all raw CSVs and return as DataFrames."""
    files = {
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv',
    }
    frames = {}
    for key, filename in files.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected {filename} in {data_dir}.")
        frames[key] = pd.read_csv(path, low_memory=False)
    return frames

def build_customers(customers_df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe in unified format for customers.csv."""
    customers = customers_df.copy()
    customers = customers.rename(columns={
        'customer_id': 'customer_id',
        'customer_city': 'location',
    })
    customers = customers[['customer_id', 'location']]
    customers['registration_date'] = pd.NaT  # unknown
    customers['ip_address'] = np.nan
    customers['is_fraudster'] = np.nan  # unknown ground-truth
    customers['fraud_type'] = np.nan
    return customers

def build_products(products_df: pd.DataFrame) -> pd.DataFrame:
    products = products_df.copy()
    products = products.rename(columns={
        'product_id': 'product_id',
        'product_category_name': 'category',
        'product_name_lenght': 'name_length',
        'product_description_lenght': 'description_length',
        'product_photos_qty': 'photos_qty',
        'product_weight_g': 'weight_g',
        'product_length_cm': 'length_cm',
        'product_height_cm': 'height_cm',
        'product_width_cm': 'width_cm',
    })
    products['price'] = np.nan  # filled later from order items avg.
    products['base_return_rate'] = np.nan
    return products[['product_id', 'category', 'price', 'base_return_rate']]

def build_orders_orders_items(orders_df: pd.DataFrame, order_items_df: pd.DataFrame) -> pd.DataFrame:
    """Merge orders + first item to generate simplified orders table."""
    # For simplicity, take first item for each order_id
    first_items = order_items_df.sort_values('order_item_id').groupby('order_id').first().reset_index()

    merged = orders_df.merge(first_items, on='order_id', how='left', suffixes=('', '_item'))
    merged['total_amount'] = merged['price'] * merged['order_item_id']  # rough

    orders_out = merged.rename(columns={
        'customer_id': 'customer_id',
        'order_id': 'order_id',
        'product_id': 'product_id',
        'order_purchase_timestamp': 'order_date',
        'price': 'unit_price',
        'order_item_id': 'quantity',
    })
    orders_out['delivery_date'] = pd.to_datetime(merged['order_delivered_customer_date'])
    orders_out['shipping_address'] = np.nan  # not present
    orders_out['ip_address'] = np.nan
    orders_out = orders_out[['order_id', 'customer_id', 'product_id', 'order_date', 'delivery_date', 'quantity', 'unit_price', 'total_amount', 'shipping_address', 'ip_address']]
    return orders_out

def build_returns(orders_df: pd.DataFrame, reviews_df: pd.DataFrame, order_items_df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic: treat negative reviews (score<=2) or order_status=='canceled'
    as returns/refunds. This is a proxy because Olist lacks explicit returns."""
    # Negative reviews
    negative = reviews_df[reviews_df['review_score'] <= 2].copy()
    negative['return_reason'] = 'Negative review'

    # Canceled orders
    canceled_orders = orders_df[orders_df['order_status'] == 'canceled'].copy()
    canceled_orders = canceled_orders[['order_id']].copy()
    canceled_orders['return_reason'] = 'Order canceled'

    # Combine and deduplicate
    returns = pd.concat([negative[['order_id', 'return_reason']], canceled_orders], ignore_index=True)
    returns = returns.drop_duplicates('order_id')

    # Merge with orders to get customer info and dates
    merged = orders_df.merge(returns, on='order_id')
    # Add product_id via first item per order
    first_items = order_items_df.sort_values('order_item_id').groupby('order_id').first()['product_id']
    merged = merged.merge(first_items, on='order_id', how='left')
    merged['return_id'] = 'RET_' + merged['order_id'].astype(str)
    purchase_ts = pd.to_datetime(merged['order_purchase_timestamp'])
    merged['return_date'] = purchase_ts + pd.Timedelta(days=30)  # heuristic
    merged['refund_amount'] = np.nan
    merged['return_status'] = 'Approved'
    merged['quantity_returned'] = 1
    merged['days_to_return'] = (merged['return_date'] - purchase_ts).dt.days
    merged['return_shipping_address'] = np.nan
    merged['is_fraud'] = np.nan

    return merged[['return_id', 'order_id', 'customer_id', 'product_id', 'return_date', 'return_reason', 'quantity_returned', 'refund_amount', 'return_status', 'is_fraud', 'days_to_return', 'return_shipping_address']]

def main():
    parser = argparse.ArgumentParser(description="Preprocess Olist dataset for fraud detection pipeline")
    parser.add_argument('--data-dir', required=True, help='Directory containing raw Olist CSVs')
    parser.add_argument('--out-dir', default='.', help='Output directory for unified CSVs')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_raw_frames(data_dir)
    print("Loaded raw Olist tables.")

    customers = build_customers(frames['customers'])
    print(f"Customers: {len(customers):,}")

    # Compute average price per product from order_items and fill into products
    products = build_products(frames['products'])
    avg_price = frames['order_items'].groupby('product_id')['price'].mean().rename('avg_price')
    products = products.merge(avg_price, on='product_id', how='left')
    products['price'] = products['avg_price']
    products.drop(columns=['avg_price'], inplace=True)
    print(f"Products: {len(products):,}")

    orders = build_orders_orders_items(frames['orders'], frames['order_items'])
    print(f"Orders: {len(orders):,}")

    returns = build_returns(frames['orders'], frames['reviews'], frames['order_items'])
    print(f"Returns proxy records: {len(returns):,}")

    # Save
    customers.to_csv(out_dir / 'customers.csv', index=False)
    products.to_csv(out_dir / 'products.csv', index=False)
    orders.to_csv(out_dir / 'orders.csv', index=False)
    returns.to_csv(out_dir / 'returns.csv', index=False)

    print("Unified CSVs written to", out_dir.resolve())

if __name__ == '__main__':
    main()
