import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged dataset
df = pd.read_csv('merged_overwatch2_data.csv')

# Inspect the columns to understand available metrics
print("Columns in merged dataset:")
print(df.columns)

# Create a unified Date column from the 'Year' and 'Month' columns.
# We'll assume that the first day of the month represents the period.
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')

# Filter data from August 2023 to September 2024
start_date = pd.to_datetime('2023-08-01')
end_date = pd.to_datetime('2024-09-30')
df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].reset_index(drop=True)

print(f"Number of records in filtered dataset: {len(df_filtered)}")
print("Filtered data sample:")
print(df_filtered.head())

# --- EDA: Summary Statistics ---
print("\nSummary Statistics:")
print(df_filtered.describe())

print("\nMissing Data Counts:")
print(df_filtered.isnull().sum())

# --- Visualization 1: Steam Player Trends ---
plt.figure(figsize=(12, 6))
# Check if columns 'Avg. Players' and 'Peak Players' exist; adjust names if needed.
if 'Avg. Players' in df_filtered.columns and 'Peak Players' in df_filtered.columns:
    plt.plot(df_filtered['Date'], df_filtered['Avg. Players'], marker='o', linestyle='-', label='Avg. Players')
    plt.plot(df_filtered['Date'], df_filtered['Peak Players'], marker='o', linestyle='--', label='Peak Players')
    plt.xlabel('Date')
    plt.ylabel('Player Count')
    plt.title('Overwatch 2 Steam Player Trends (Aug 2023 - Sep 2024)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Columns 'Avg. Players' or 'Peak Players' not found in dataset.")

# --- Visualization 2: Twitch Viewership Trends ---
plt.figure(figsize=(12, 6))
if 'Hours_watched' in df_filtered.columns:
    plt.plot(df_filtered['Date'], df_filtered['Hours_watched'], marker='o', linestyle='-', color='orange', label='Hours Watched')
    plt.xlabel('Date')
    plt.ylabel('Hours Watched')
    plt.title('Overwatch 2 Twitch Hours Watched (Aug 2023 - Sep 2024)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'Hours_watched' not found in dataset.")

# --- Visualization 3: Scatter Plot (Steam vs. Twitch) ---
if 'Avg. Players' in df_filtered.columns and 'Hours_watched' in df_filtered.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_filtered, x='Avg. Players', y='Hours_watched')
    plt.title('Correlation: Avg. Players vs. Hours Watched (Overwatch 2)')
    plt.xlabel('Average Players (Steam)')
    plt.ylabel('Hours Watched (Twitch)')
    plt.tight_layout()
    plt.show()
else:
    print("Required columns for scatter plot not found.")

# --- Visualization 4: Correlation Matrix ---
# Select numeric columns for correlation analysis.
numeric_cols = df_filtered.select_dtypes(include='number').columns
if len(numeric_cols) > 0:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_filtered[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix for Overwatch 2 Metrics')
    plt.tight_layout()
    plt.show()
else:
    print("No numeric columns available for correlation analysis.")
