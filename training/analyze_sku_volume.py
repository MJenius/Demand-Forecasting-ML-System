"""Analyze SKU volume distribution for segmentation"""
import pandas as pd
import numpy as np

# Load features
df = pd.read_parquet('features/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])

# Calculate mean daily sales per item-store
mean_sales = df.groupby(['item_id', 'store_id'])['target'].mean().reset_index()
mean_sales.columns = ['item_id', 'store_id', 'mean_sales']

print('SKU Sales Volume Statistics:')
print(f'Total SKUs: {len(mean_sales):,}')
print(f'\nPercentiles:')
for pct in [10, 25, 50, 75, 90]:
    val = mean_sales['mean_sales'].quantile(pct/100)
    print(f'  {pct}th: {val:.2f}')

print(f'\nMean sales < 1.5 (Low-volume):')
low_vol = len(mean_sales[mean_sales['mean_sales'] < 1.5])
print(f'  Count: {low_vol:,} SKUs ({low_vol/len(mean_sales)*100:.1f}%)')

print(f'\nMean sales >= 1.5 (Normal-volume):')
normal_vol = len(mean_sales[mean_sales['mean_sales'] >= 1.5])
print(f'  Count: {normal_vol:,} SKUs ({normal_vol/len(mean_sales)*100:.1f}%)')

# Check sales distribution within each segment
low = mean_sales[mean_sales['mean_sales'] < 1.5]
normal = mean_sales[mean_sales['mean_sales'] >= 1.5]

print(f'\n--- Low-volume SKU sales distribution ---')
print(f'  Min: {low["mean_sales"].min():.2f}, Max: {low["mean_sales"].max():.2f}')
print(f'  Mean: {low["mean_sales"].mean():.2f}')

print(f'\n--- Normal-volume SKU sales distribution ---')
print(f'  Min: {normal["mean_sales"].min():.2f}, Max: {normal["mean_sales"].max():.2f}')
print(f'  Mean: {normal["mean_sales"].mean():.2f}')

# Save segmentation for use in training
mean_sales.to_csv('training/sku_segments.csv', index=False)
print(f'\nSaved SKU segments to: training/sku_segments.csv')
