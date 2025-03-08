import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# Load data
script_dir = os.path.dirname(__file__)

relative_path = r"..\datasets\merged_game_twitch_data.csv"
game_data_path = os.path.join(script_dir, relative_path)
game_data = pd.read_csv(game_data_path)

relative_path = r"apex_patch_notes.xlsx"
patch_data_path = os.path.join(script_dir, relative_path)
patch_data = pd.read_excel(patch_data_path)

# Clean up column names
game_data.rename(columns=lambda x: x.strip(), inplace=True)
patch_data.rename(columns=lambda x: x.strip(), inplace=True)

# Convert date columns using the expected format (e.g., "Mar-25" becomes Mar 2025)
game_data['Month'] = pd.to_datetime(game_data['Date'], format='%b-%y', errors='coerce')
patch_data['Date'] = pd.to_datetime(patch_data['Date'])

game_data['Date'] = game_data['Date'].astype('datetime64[ns]')
patch_data['Date'] = patch_data['Date'].astype('datetime64[ns]')

# Print out the number of conversion failures
print("Game data NaT count:", game_data['Date'].isna().sum())
print("Patch data NaT count:", patch_data['Date'].isna().sum())

# Drop rows with invalid dates
game_data = game_data.dropna(subset=['Date'])
patch_data = patch_data.dropna(subset=['Date'])

# Filter Apex Legends data
apex_data = game_data[game_data['Game_Name'] == 'Apex Legends']

# Merge datasets on Date
merged_data = pd.merge(apex_data, patch_data, on='Date', how='left')
merged_data = merged_data.dropna(subset=['Date'])

# Compute percentage change for player base
merged_data['Player Base % Change'] = merged_data['Avg. Players'].pct_change() * 100

# Rename column to avoid issues in the regression formula
merged_data.rename(columns={"Player Base % Change": "PlayerBaseChange"}, inplace=True)

# Create a Patch column: 0 = No Patch, 1 = Patch (using Major and Minor columns)
merged_data['Patch'] = merged_data['Major'].fillna(0) + merged_data['Minor'].fillna(0)

# Define the time window (in days) for analysis before and after a patch event
days_before = 7
days_after = 7

# Identify patch days
patch_dates = merged_data[merged_data['Patch'] > 0]['Date']

# Create before/after patch samples for PlayerBaseChange
before_patch = []
after_patch = []

for patch_date in patch_dates:
    before = merged_data[(merged_data['Date'] >= patch_date - pd.Timedelta(days=days_before)) &
                         (merged_data['Date'] < patch_date)]['PlayerBaseChange']
    after = merged_data[(merged_data['Date'] > patch_date) &
                        (merged_data['Date'] <= patch_date + pd.Timedelta(days=days_after))]['PlayerBaseChange']
    
    if len(before) > 0 and len(after) > 0:  # Only if both samples are non-empty
        before_patch.append(before.mean())
        after_patch.append(after.mean())

before_patch = np.array(before_patch)
after_patch = np.array(after_patch)

# Run statistical tests only if there are enough patch events
if len(before_patch) > 1 and len(after_patch) > 1:
    # A. Paired T-Test for PlayerBaseChange
    t_stat, p_value_ttest = ttest_rel(before_patch, after_patch, nan_policy='omit')
    print("Paired T-Test - Player Base Change Before vs. After Patch:")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value_ttest:.4f}")

    # B. Variance Test (Levene’s test)
    stat_levene, p_value_levene = levene(before_patch, after_patch, center='mean')
    print("Levene’s Test for Variance - Before vs. After Patch:")
    print(f"Statistic: {stat_levene:.4f}, p-value: {p_value_levene:.4f}")
else:
    print("Not enough patch events for statistical tests.")

# C. Regression Analysis: Does Patch Type Impact Player Base Change?
# Create a categorical Patch_Type column
merged_data['Patch_Type'] = np.where(merged_data['Major'] == 1, 'Major',
                                       np.where(merged_data['Minor'] == 1, 'Minor', 'None'))

# Run OLS regression with the simplified column name
model = ols("PlayerBaseChange ~ C(Patch_Type)", data=merged_data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nANOVA Results: Patch Type Impact on Player Base Change")
print(anova_table)

# Plotting the trends
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

# Plot PlayerBaseChange
ax1.plot(merged_data['Date'], merged_data['PlayerBaseChange'], color='blue', linewidth=2, label='Player Base % Change')
# Plot Twitch viewership % change (assuming 'Hours_watched' exists)
ax2.plot(merged_data['Date'], merged_data['Hours_watched'].pct_change() * 100, color='red', linewidth=2, linestyle='solid', label='Twitch Viewership % Change')

# Remove anomalous data in 2022
merged_data = merged_data[~((merged_data['Date'].dt.year == 2022) & (merged_data['Twitch Viewership % Change'].abs() > 500))]
ax2.plot(merged_data['Date'], merged_data['Twitch Viewership % Change'], color='red', linewidth=2, linestyle='solid', label='Twitch Viewership % Change')

# Add vertical lines for patch events from patch_data
for index, row in patch_data.iterrows():
    if pd.isna(row['Date']):
        continue  # Skip invalid dates
    if row.get('Major', 0) == 1:
        ax1.axvline(row['Date'], color='darkgrey', linestyle='dotted', alpha=0.8,
                    label='Major Patch' if index == 0 else "")
    elif row.get('Minor', 0) == 1:
        ax1.axvline(row['Date'], color='lightgrey', linestyle='dotted', alpha=0.6,
                    label='Minor Patch' if index == 0 else "")

ax1.set_xlabel('Date')
ax1.set_ylabel('Player Base % Change', color='blue')
ax2.set_ylabel('Twitch Viewership % Change', color='red')
ax1.set_title('Apex Legends Player Base % Change & Twitch Viewership % Change with Patch Releases')
ax1.legend(loc='upper left', title='Player Base & Twitch Viewership')
plt.show()
