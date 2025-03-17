import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime


# -------------------------------
# Load and Parse Patch Notes JSON
# -------------------------------
def load_patch_notes_json(patch_folder):
    patch_data = []

    for filename in os.listdir(patch_folder):
        if filename.endswith('_patch_notes.json'):
            appid = int(filename.split('_')[0])
            file_path = os.path.join(patch_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for entry in json_data:
                    patch_timestamp = entry.get('date', None)
                    if patch_timestamp:
                        patch_date = datetime.utcfromtimestamp(patch_timestamp)
                        patch_data.append({'appid': appid, 'patch_date': patch_date})

    patches_df = pd.DataFrame(patch_data)
    patches_df['patch_date'] = pd.to_datetime(patches_df['patch_date'])  # Keep exact dates
    print(f"Loaded {len(patches_df)} patches from {len(os.listdir(patch_folder))} files.")
    return patches_df


# -------------------------------
# ITSA Per Game Function
# -------------------------------
def itsa_per_game(df, patch_df, metric='Avg. Players'):
    df['Month'] = pd.to_datetime(df['Month'], format='%b-%y', errors='coerce')
    all_results = []

    for game_id in df['Game_Id'].unique():
        game_df = df[df['Game_Id'] == game_id].copy()
        game_name = game_df['Game_Name'].iloc[0]
        topic = game_df['LDA_Topic'].iloc[0]

        if game_df.empty:
            continue

        monthly_df = game_df.groupby('Month')[metric].mean().reset_index()
        patches = patch_df[patch_df['appid'] == game_id]

        # Filter patches to only those within player data time range
        min_month, max_month = monthly_df['Month'].min(), monthly_df['Month'].max()
        patches = patches[(patches['patch_date'] >= min_month) & (patches['patch_date'] <= max_month)]

        if patches.empty:
            print(f"⚠️ No relevant patches for {game_name} within playerbase timeline. Skipping.")
            continue

        print(f"\nAnalyzing Game: {game_name} (AppID: {game_id}) with {len(patches)} patches.")

        monthly_df['time'] = np.arange(len(monthly_df))

        monthly_df = monthly_df.dropna(subset=[metric])
        if monthly_df.empty:
            print(f"⚠️ No valid playerbase data for {game_name}. Skipping.")
            continue

        X = monthly_df[['time']]
        X = sm.add_constant(X)
        y = monthly_df[metric]

        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

        result = {
            'Game_Id': game_id,
            'Game_Name': game_name,
            'Topic': topic,
            'R_squared': model.rsquared
        }
        all_results.append(result)

        print(model.summary())

        plt.figure(figsize=(12, 6))
        plt.plot(monthly_df['Month'], y, label='Actual', marker='o')
        plt.plot(monthly_df['Month'], model.fittedvalues, label='Fitted', linestyle='--')
        # Plot patches as vertical lines at exact patch dates
        for patch_date in patches['patch_date']:
            plt.axvline(patch_date, color='red', linestyle='--', alpha=0.7, label='Patch' if patch_date == patches['patch_date'].iloc[0] else "")
        plt.xlabel('Month')
        plt.ylabel(metric)
        plt.title(f"ITSA for {game_name} ({metric})")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()

    results_df = pd.DataFrame(all_results)
    return results_df


# -------------------------------
# Main Driver Code
# -------------------------------
if __name__ == "__main__":
    merged_df = pd.read_csv(r"C:\\Users\\szymc\\Desktop\\PythonProjects\\Game Analysis\\categorized_games_with_topics.csv")
    patch_df = load_patch_notes_json(r"C:\\Users\\szymc\\Desktop\\PythonProjects\\Game Analysis\\patch_notes")

    itsa_results_df = itsa_per_game(merged_df, patch_df, metric='Avg. Players')

    if not itsa_results_df.empty:
        itsa_results_df.to_csv(r"C:\\Users\\szymc\\Desktop\\PythonProjects\\Game Analysis\\itsa_results_players.csv", index=False)
        print("\n✅ All analysis completed and results saved.")
    else:
        print("\n⚠️ No valid ITSA results to analyze or summarize. Please check data availability.")
