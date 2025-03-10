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
from scipy.stats import shapiro
import random

# ------------------------------
# 1. Load and Prepare the Data
# ------------------------------
path = os.path.join(os.path.dirname(__file__), r"datasets\game_data.csv")
df = pd.read_csv(path)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')

game_id_to_use = 578080 
df = df[df['Game_Id'] == game_id_to_use]
df = df.sort_values('Month')
df.set_index('Month', inplace=True)

# Ignore the first year of data to remove early fluctuations
df = df[df.index >= df.index.min() + pd.DateOffset(years=1)]

ts = df['Avg. Players'].pct_change() * 100
ts.name = 'value'
ts = ts.dropna()

# ------------------------------
# 2. Apply Detrending Transformations
# ------------------------------
rolling_window = 24  # 24-month rolling window for smoothing
rolling_mean = ts.rolling(rolling_window, center=True).mean()
ts_detrended = ts - rolling_mean
ts_detrended = ts_detrended.dropna()

decomposition = seasonal_decompose(ts_detrended, model='additive', period=12)
ts_detrended = ts_detrended - decomposition.trend - decomposition.seasonal
ts_detrended = ts_detrended.dropna()

print(ts_detrended.describe())
stat, p_value = shapiro(ts_detrended)
print(f"Shapiro-Wilk Test: p-value = {p_value:.4f}")

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
            # Initially assign a temporary type (0)
            extracted_events.append({"date": update_date.strftime("%Y-%m-%d"), "type": 0})

# Randomly assign only 3 events to type 1, and all others to type 2.
if len(extracted_events) >= 3:
    type1_indices = set(random.sample(range(len(extracted_events)), 3))
else:
    type1_indices = set(range(len(extracted_events)))  # if less than 3, all become type 1

for i, event in enumerate(extracted_events):
    if i in type1_indices:
        event['type'] = 1
    else:
        event['type'] = 2

# Only keep events that occur after the start of our analysis period
detrended_start = ts_detrended.index.min()
events = [e for e in extracted_events if pd.to_datetime(e["date"]) >= detrended_start]
events = sorted(events, key=lambda x: pd.to_datetime(x["date"]))

# --------------------------------------------
# 4. Aggregate Post-Event Windows by Event Type
# --------------------------------------------
window_months = 3  # post-event window in months
aggregated_post = {}  # key = event type, value = set of indices in ts_detrended

# Initialize dictionary for event types
for event in events:
    etype = event['type']
    if etype not in aggregated_post:
        aggregated_post[etype] = set()

# Populate the aggregated post indices for each event type
for event in events:
    event_date = pd.to_datetime(event['date'])
    etype = event['type']
    post_window = ts_detrended.loc[event_date : event_date + pd.DateOffset(months=window_months)].index
    aggregated_post[etype].update(post_window)

# Now, for each event type, define the post-event sample and the baseline
results_by_type = {}
for etype, post_indices in aggregated_post.items():
    post_indices = sorted(list(post_indices))
    post_values = ts_detrended.loc[post_indices] if len(post_indices) > 0 else pd.Series([], dtype=ts_detrended.dtype)
    baseline_indices = ts_detrended.index.difference(post_indices)
    baseline_values = ts_detrended.loc[baseline_indices]
    
    if len(post_values) < 2 or len(baseline_values) < 2:
        print(f"Skipping event type {etype}: insufficient data (post: {len(post_values)}, baseline: {len(baseline_values)})")
        continue
    
    n1, n2 = len(post_values), len(baseline_values)
    s1 = post_values.std(ddof=1)
    s2 = baseline_values.std(ddof=1)
    sp = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    se_diff = sp * np.sqrt(1/n1 + 1/n2)
    diff = post_values.mean() - baseline_values.mean()
    df_val = n1 + n2 - 2
    t_stat, p_val = stats.ttest_ind(post_values, baseline_values, equal_var=True)
    t_crit = stats.t.ppf(1-0.025, df_val)
    ci_lower = diff - t_crit * se_diff
    ci_upper = diff + t_crit * se_diff

    results_by_type[etype] = {
        'n_post': n1,
        'n_baseline': n2,
        'mean_post': post_values.mean(),
        'mean_baseline': baseline_values.mean(),
        'diff': diff,
        'se_diff': se_diff,
        't_stat': t_stat,
        'df': df_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_val': p_val
    }
    
    print(f"Event Type {etype}: n_post = {n1}, n_baseline = {n2}, "
          f"Mean Difference = {diff:.2f} (95% CI: {ci_lower:.2f}, {ci_upper:.2f}), "
          f"t-stat = {t_stat:.2f}, p-value = {p_val:.7f}")

# --------------------------------------------
# 5. Optionally, Plot the Results for Each Event Type
# --------------------------------------------
if results_by_type:
    types = list(results_by_type.keys())
    diffs = [results_by_type[t]['diff'] for t in types]
    ci_lowers = [results_by_type[t]['ci_lower'] for t in types]
    ci_uppers = [results_by_type[t]['ci_upper'] for t in types]
    errors = [abs(d - ci_lowers[i]) for i, d in enumerate(diffs)]
    
    plt.figure(figsize=(10, 6))
    bar_colors = ['green' if d > 0 else 'red' for d in diffs]
    plt.bar([str(t) for t in types], diffs, yerr=errors, capsize=5, color=bar_colors)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel("Event Type")
    plt.ylabel("Mean Difference (%)")
    plt.title("Aggregated Post-Event vs. Baseline Differences by Event Type")
    plt.show()

# # Window size for ITS regression (used earlier)
# window_size = 3

# # --------------------------------------------
# # 4. Helper Functions for ITS Analysis
# # --------------------------------------------
# def get_window_data(ts, event_date, window_size):
#     event_date = pd.to_datetime(event_date)
#     start_date = event_date - pd.DateOffset(months=window_size)
#     end_date = event_date + pd.DateOffset(months=window_size)
#     window_data = ts[start_date:end_date].copy().reset_index()
#     window_data.rename(columns={'index': 'Month', ts.name: 'value'}, inplace=True)
#     window_data['time'] = (window_data['Month'] - event_date).dt.days / 30.0
#     return window_data

# def its_regression(window_data):
#     data = window_data.copy()
#     if data.empty:
#         print("window_data is empty, skipping regression.")
#         return None  # Skip regression if no data
#     data['D'] = (data['time'] >= 0).astype(int)
#     data['time_after'] = data['time'] * data['D']
#     X = data[['time', 'D', 'time_after']]
#     y = data['value']
#     if X.empty or y.empty:
#         print("Error: X or y is empty for regression.")
#         return None
#     X = sm.add_constant(X).dropna()
#     y = y.dropna()
#     if X.empty or y.empty:
#         print("Error after dropping NaNs: X or y is still empty.")
#         return None
#     model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
#     return model

# # ------------------------------
# # 5. Loop Through Events and Run ITS Analysis
# # ------------------------------
# results = {}
# cum_effect_list = []

# for event in events:
#     event_date = event['date']
#     event_type = event['type']
#     window_data = get_window_data(ts_detrended, event_date, window_size)
#     if window_data.empty:
#         continue
#     its_model = its_regression(window_data)
#     if its_model is None:
#         continue  # Skip empty results
#     effect = its_model.params['D']
#     p_val = its_model.pvalues['D']
#     results[event_date] = {'ITS': its_model}
#     cum_effect_list.append({
#         'event_date': pd.to_datetime(event_date),
#         'effect': effect,
#         'p_value': p_val,
#         'event_type': event_type
#     })

# # ------------------------------
# # 6. Aggregate ITS Results for Category Testing
# # ------------------------------
# cum_df = pd.DataFrame(cum_effect_list)
# for etype in cum_df['event_type'].unique():
#     subset = cum_df[cum_df['event_type'] == etype]
#     effects = subset['effect'].dropna()
#     n = len(effects)
#     if n > 1:
#         t_stat, p_val = stats.ttest_1samp(effects, popmean=0)
#         print(f"\nCategory {etype}: n = {n}, Mean Effect = {effects.mean():.2f}%, t-stat = {t_stat:.2f}, p-value = {p_val:.7f}")
#     else:
#         print(f"\nCategory {etype}: n = {n} (not enough samples for t-test)")
# cum_df = cum_df.sort_values('event_date')

# plt.figure(figsize=(12, 6))
# plt.plot(cum_df['event_date'], cum_df['p_value'], marker='o', linestyle='-', label='p-value')
# plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold (0.05)')
# plt.axhline(y=0.01, color='g', linestyle='--', label='Highly Significant (0.01)')
# plt.axhline(y=0.1, color='b', linestyle='--', label='Weak Significance (0.1)')
# plt.xlabel("Event Date")
# plt.ylabel("p-value")
# plt.title("Statistical Significance (p-values) of ITS Effects")
# plt.legend()
# plt.xticks(rotation=45)
# plt.grid()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(cum_df['event_date'], cum_df['effect'], marker='o', linestyle='-', label='Effect Size (%)')
# plt.axhline(y=0, color='k', linestyle='--', label='No Effect')
# plt.xlabel("Event Date")
# plt.ylabel("Effect Size (%)")
# plt.title("Interrupted Time Series Effect Sizes Over Time")
# plt.legend()
# plt.xticks(rotation=45)
# plt.grid()
# plt.show()