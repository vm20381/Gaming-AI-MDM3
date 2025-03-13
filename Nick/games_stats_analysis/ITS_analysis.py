#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset

def main():
    # 1. Load and inspect your CSV
    csv_path = "merged_game_twitch_data.csv"  # Adjust if needed
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    df.columns = df.columns.str.strip()

    # Inspect the first few 'Month' values
    print("\n--- Debug: First few 'Month' values ---")
    print(df['Month'].head(10))

    # 2. Parse 'Month' with format '%b-%y' (e.g. 'Sep-24' => 2024-09-01)
    df['Date'] = pd.to_datetime(df['Month'], format='%b-%y', errors='coerce')

    # Check the overall date range
    print("\nOverall date range:", df['Date'].min(), "to", df['Date'].max())

    # 3. Filter for Counter-Strike 2
    # Make sure the game name matches exactly what's in your CSV
    df_csgo = df[df["Game_Name"].str.lower() == "counter-strike 2"].copy()
    df_csgo = df_csgo.sort_values('Date').reset_index(drop=True)
    print("CSGO date range:", df_csgo['Date'].min(), "to", df_csgo['Date'].max())

    # Check that 'Avg. Players' is numeric
    if "Avg. Players" not in df_csgo.columns:
        print("ERROR: 'Avg. Players' column not found in df_csgo.")
        return

    df_csgo["Avg. Players"] = pd.to_numeric(df_csgo["Avg. Players"], errors='coerce')
    # Drop rows missing 'Avg. Players' or 'Date'
    df_csgo = df_csgo.dropna(subset=["Avg. Players", "Date"])
    if df_csgo.empty:
        print("\nNo valid CSGO data after dropping NaNs. Exiting.")
        return

    print("\n--- Debug: 'Avg. Players' describe() ---")
    print(df_csgo["Avg. Players"].describe())

    # 4. Create fake patch events
    fake_patch_dates = [
        pd.Timestamp("2017-05-01"),
        pd.Timestamp("2019-08-01"),
        pd.Timestamp("2020-11-01"),
        pd.Timestamp("2021-05-01"),
        pd.Timestamp("2022-08-01"),
        pd.Timestamp("2023-11-01")
    ]
    intervention_window = 3  # months

    def is_post_patch(date):
        for patch_date in fake_patch_dates:
            if patch_date <= date < (patch_date + DateOffset(months=intervention_window)):
                return 1
        return 0

    df_csgo['Post_Patch'] = df_csgo['Date'].apply(is_post_patch)

    # 5. Create time trend and interaction term
    df_csgo = df_csgo.sort_values('Date').reset_index(drop=True)
    df_csgo['Time'] = range(len(df_csgo))
    df_csgo['Interaction'] = df_csgo['Time'] * df_csgo['Post_Patch']

    # 6. ITS Regression
    X = df_csgo[['Time', 'Post_Patch', 'Interaction']]
    X = sm.add_constant(X)
    y = df_csgo["Avg. Players"]

    its_model = sm.OLS(y, X).fit(cov_type='HC3')
    print("\n--- ITS Model Summary ---")
    print(its_model.summary())

    # 7. Plot the time series
    plt.figure(figsize=(14, 6))
    plt.plot(df_csgo['Date'], df_csgo['Avg. Players'], marker='o', linestyle='-', label='Avg. Players')

    for i, patch_date in enumerate(fake_patch_dates):
        plt.axvline(patch_date, color='red', linestyle='--', alpha=0.8,
                    label="Patch Event" if i == 0 else "")
        plt.axvspan(patch_date,
                    patch_date + DateOffset(months=intervention_window),
                    color='red', alpha=0.2,
                    label="Post-Patch Window" if i == 0 else "")

    plt.title("CSGO: Avg. Players Over Time with Fake Patch Interventions")
    plt.xlabel("Date")
    plt.ylabel("Avg. Players")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
