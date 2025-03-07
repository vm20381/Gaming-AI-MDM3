# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:25:19 2025

@author: ashru
"""
#import datetime
import pandas as pd
from prophet import Prophet
#import matplotlib


#import dataframes
playerData = pd.read_csv("C:/Users/ashru/Downloads/game_data.csv")
apexPatches = pd.read_excel(open("C:/Users/ashru/Downloads/apex_patch_notes.xlsx",'rb'))

playerData = playerData[playerData.Game_Name == 'Apex Legends']
#get apex patches starting when game data does
apexPatches = apexPatches.iloc[:161]


playerData.Month = pd.to_datetime(playerData["Month"], format="%b-%y")

apexPatches.Date = pd.to_datetime(apexPatches.Date, format="%B %d, %Y")#.dt.strftime("%Y-%m-%d")
print(apexPatches.columns)

apexPatches["year"] = apexPatches["Date"].dt.year
apexPatches["month"] = apexPatches["Date"].dt.month

playerData["year"] = playerData["Month"].dt.year
playerData["month"] = playerData["Month"].dt.month

result = apexPatches.merge(playerData, left_on=["year", "month"], right_on=["year", "month"], how="left")
result = result[['Date','Avg. Players']]
result.Date = result.Date.dt.strftime("%Y-%m-%d")
result.columns = ['ds','y']

m = Prophet()
m.fit(result)

future = m.make_future_dataframe(periods=365)
future.tail()

# Python
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


fig1 = m.plot(forecast)

