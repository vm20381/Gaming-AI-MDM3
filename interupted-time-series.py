import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# ------------------------------
# 1. Load and Prepare the Data
# ------------------------------
path = os.path.join(os.path.dirname(__file__), r"datasets\game_data.csv")
df = pd.read_csv(path)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')

game_id_to_use = 730 
df = df[df['Game_Id'] == game_id_to_use]
df = df.sort_values('Month')
df.set_index('Month', inplace=True)

# Ignore the first year of data to remove early fluctuations
df = df[df.index >= df.index.min() + pd.DateOffset(years=1)]

# Convert Avg. Players to percentage change (in percentage points)
ts = df['Avg. Players'].pct_change() * 100
ts = ts.dropna()

# # Plot original transformed data
# plt.figure(figsize=(12, 6))
# ts.plot(title="Original Percentage Change in Player Base", ylabel="% Change", xlabel="Date")
# plt.show()

# ------------------------------
# 2. Apply Detrending Transformations
# ------------------------------

# Step 1: Remove slow-moving trends using a rolling mean
rolling_window = 24  # 12-month rolling window for smoothing
rolling_mean = ts.rolling(rolling_window, center=True).mean()
ts_detrended = ts - rolling_mean
ts_detrended = ts_detrended.dropna()

# # Plot after rolling mean subtraction
# plt.figure(figsize=(12, 6))
# ts_detrended.plot(title="After Rolling Mean Subtraction", ylabel="% Change", xlabel="Date")
# plt.show()

# Step 2: Remove seasonality using seasonal decomposition
decomposition = seasonal_decompose(ts_detrended, model='additive', period=12)  # Assuming monthly data
ts_detrended = ts_detrended - decomposition.trend - decomposition.seasonal
ts_detrended = ts_detrended.dropna()

plt.figure(figsize=(12, 6))
ts_detrended.plot(title="Detrended Data", ylabel="% Change", xlabel="Date")
plt.show()

# ------------------------------
# 3. Extract Update Dates from JSON Files and Filter
# ------------------------------
patch_folder = os.path.join(os.path.dirname(__file__), "patch_notes")
pattern = os.path.join(patch_folder, f"{game_id_to_use}_patch_notes.json")
extracted_events = []
for file in glob.glob(pattern):
    with open(file, "r", encoding="utf-8") as f:
        patch_data = json.load(f)
        for note in patch_data:
            update_date = pd.to_datetime(note["date"], unit="s")
            extracted_events.append({"date": update_date.strftime("%Y-%m-%d"), "type": 1})

earliest_date = df.index.min()
events = [e for e in extracted_events if pd.to_datetime(e["date"]) >= earliest_date]
events = sorted(events, key=lambda x: pd.to_datetime(x["date"]))

# Window size: 12 months before and after the event
window_size = 3

# --------------------------------------------
# 4. Helper Functions for ITS Analysis
# --------------------------------------------
def get_window_data(ts, event_date, window_size):
    event_date = pd.to_datetime(event_date)
    start_date = event_date - pd.DateOffset(months=window_size)
    end_date = event_date + pd.DateOffset(months=window_size)
    window_data = ts[start_date:end_date].copy().reset_index()
    window_data['time'] = (window_data['Month'] - event_date).dt.days / 30.0
    return window_data

def its_regression(window_data):
    data = window_data.copy()
    
    if data.empty:
        print("window_data is empty, skipping regression.")
        return None  # Skip regression if no data

    data['D'] = (data['time'] >= 0).astype(int)
    data['time_after'] = data['time'] * data['D']
    X = data[['time', 'D', 'time_after']]
    y = data.iloc[:, 1]  # percentage change column

    # # Debug prints
    # print(f"window_data:\n{window_data.head()}\n")
    # print(f"X (Predictors):\n{X.head()}\n")
    # print(f"y (Target variable):\n{y.head()}\n")

    if X.empty or y.empty:
        print(f"Error: X or y is empty for event date {window_data.iloc[0, 0] if not window_data.empty else 'Unknown'}")
        return None  # Skip regression

    X = sm.add_constant(X).dropna()
    y = y.dropna()

    if X.empty or y.empty:
        print("Error after dropping NaNs: X or y is still empty.")
        return None

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    return model


# --------------------------------------------
# 5. Loop Through Events and Run ITS Analysis
# --------------------------------------------
results = {}
cum_effect_list = []

for event in events:
    event_date = event['date']
    event_type = event['type']

    window_data = get_window_data(ts_detrended, event_date, window_size)

    if window_data.empty:
        continue

    its_model = its_regression(window_data)

    if its_model is None:
        continue  # Skip empty results

    effect = its_model.params['D']
    p_value = its_model.pvalues['D']

    results[event_date] = {'ITS': its_model}

    cum_effect_list.append({
        'event_date': pd.to_datetime(event_date),
        'effect': effect,
        'p_value': p_value,
        'event_type': event_type
    })


# --------------------------------------------
# 6. Aggregate ITS Results for Category Testing
# --------------------------------------------
cum_df = pd.DataFrame(cum_effect_list)
for etype in cum_df['event_type'].unique():
    subset = cum_df[cum_df['event_type'] == etype]
    effects = subset['effect'].dropna()
    n = len(effects)
    if n > 1:
        t_stat, p_val = stats.ttest_1samp(effects, popmean=0)
        print(f"\nCategory {etype}: n = {n}, Mean Effect = {effects.mean():.2f}%, t-stat = {t_stat:.2f}, p-value = {p_val:.7f}")
    else:
        print(f"\nCategory {etype}: n = {n} (not enough samples for t-test)")

# Sort by event date
cum_df = cum_df.sort_values('event_date')

# Plot statistical significance (p-values) over time
plt.figure(figsize=(12, 6))
plt.plot(cum_df['event_date'], cum_df['p_value'], marker='o', linestyle='-', label='p-value')
plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold (0.05)')
plt.axhline(y=0.01, color='g', linestyle='--', label='Highly Significant (0.01)')
plt.axhline(y=0.1, color='b', linestyle='--', label='Weak Significance (0.1)')
plt.xlabel("Event Date")
plt.ylabel("p-value")
plt.title("Statistical Significance (p-values) of ITS Effects")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Plot effect sizes over time
plt.figure(figsize=(12, 6))
plt.plot(cum_df['event_date'], cum_df['effect'], marker='o', linestyle='-', label='Effect Size (%)')
plt.axhline(y=0, color='k', linestyle='--', label='No Effect')
plt.xlabel("Event Date")
plt.ylabel("Effect Size (%)")
plt.title("Interrupted Time Series Effect Sizes Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()
