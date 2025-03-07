import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

# Load datasets
game_data = pd.read_csv(r"C:\Users\szymc\Desktop\PythonProjects\Game Analysis\Gaming-AI-MDM3\datasets\merged_game_twitch_data.csv")
patch_data = pd.read_excel(r"C:\Users\szymc\Desktop\PythonProjects\Game Analysis\apex_patch_notes.xlsx")

# Ensure column names are stripped of spaces
game_data.rename(columns=lambda x: x.strip(), inplace=True)
patch_data.rename(columns=lambda x: x.strip(), inplace=True)

# Convert date columns to datetime format
game_data['Date'] = pd.to_datetime(game_data['Date'])
patch_data['Date'] = pd.to_datetime(patch_data['Date'])

# Filter data for Apex Legends only
apex_data = game_data[game_data['Game_Name'] == 'Apex Legends']

# Merge datasets on date to align patches with playerbase data
merged_data = pd.merge(apex_data, patch_data, on='Date', how='left')

# Plot playerbase and Twitch viewership trends
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

merged_data['Player Base % Change'] = merged_data['Avg. Players'].pct_change() * 100
ax1.plot(merged_data['Date'], merged_data['Player Base % Change'], color='blue', linewidth=2, label='Player Base % Change')
merged_data['Twitch Viewership % Change'] = merged_data['Hours_watched'].pct_change() * 100

# Remove anomalous data in 2022
merged_data = merged_data[~((merged_data['Date'].dt.year == 2022) & (merged_data['Twitch Viewership % Change'].abs() > 500))]
ax2.plot(merged_data['Date'], merged_data['Twitch Viewership % Change'], color='red', linewidth=2, linestyle='solid', label='Twitch Viewership % Change')

# Add vertical lines for patches with different colors for Minor and Major patches
for index, row in patch_data.iterrows():
    if row['Major'] == 1:
        ax1.axvline(row['Date'], color='darkgrey', linestyle='dotted', alpha=0.8, label='Major Patch' if index == 0 else "")
    elif row['Minor'] == 1:
        ax1.axvline(row['Date'], color='lightgrey', linestyle='dotted', alpha=0.6, label='Minor Patch' if index == 0 else "")

# Labels and legend
ax1.set_xlabel('Date')
ax1.set_ylabel('Player Base % Change', color='blue')
ax2.set_ylabel('Twitch Viewership % Change', color='red')
ax1.set_title('Apex Legends Player Base % Change & Twitch Viewership % Change with Patch Releases')
ax1.legend(loc='upper left', title='Player Base & Twitch Viewership')

patch_legend = [
    Line2D([0], [0], color='darkgrey', linestyle='dotted', alpha=0.8, label='Major Patch'),
    Line2D([0], [0], color='lightgrey', linestyle='dotted', alpha=0.6, label='Minor Patch')
]

ax2.legend(handles=[Line2D([0], [0], color='blue', linewidth=2, label='Player Base'),
                        Line2D([0], [0], color='red', linewidth=2, linestyle='solid', label='Twitch Viewership')], 
          loc='upper left', title='Player Base & Twitch Viewership')

ax1.legend(handles=patch_legend, loc='lower right', title='Patch Types', ncol=1)

plt.show()
