import pandas as pd
import numpy as np

filepath = "data.csv"


data = pd.DataFrame(pd.read_csv(filepath))
type(data)

data = data.iloc[2:, 19:]

data.head()

feature_names = [
    "T1G1-creativity",
    "T1G1-improvement",
    "T1G2-creativity",
    "T1G2-improvement",
    "T1G3-creativity",
    "T1G3-improvement",
    "T1G4-creativity",
    "T1G4-improvement",
    "T1G5-creativity",
    "T1G5-improvement",
    "T1G6-creativity",
    "T1G6-improvement",
    "T2G1-creativity",
    "T2G1-improvement",
    "T2G2-creativity",
    "T2G2-improvement",
    "T2G3-creativity",
    "T2G3-improvement",
    "T2G4-creativity",
    "T2G4-improvement",
    "T2G5-creativity",
    "T2G5-improvement",
    "T2G6-creativity",
    "T2G6-improvement",
]

len(feature_names)

data.columns.values[:24] = feature_names
data


taskone = data.iloc[:, :12]
taskone


tasktwo = data.iloc[:, 12:24]
tasktwo


# Task One
taskone = taskone.dropna()
taskone = taskone.apply(pd.to_numeric)


innovation_cols_one = {}
for i in range(1, 7):
    creativity_col = f"T1G{i}-creativity"
    improvement_col = f"T1G{i}-improvement"
    innovation_col = f"T1G{i}-innovation"
    innovation_cols_one[innovation_col] = (
        taskone[f"T1G{i}-creativity"] + taskone[f"T1G{i}-improvement"]
    ) / 2

innovation_df_one = pd.DataFrame(innovation_cols_one)

innovation_df_one = innovation_df_one.fillna(innovation_df_one.mean())
innovation_df_one.max

print(innovation_df_one)


# Conditions for T-Tests
# Independence of Observations
# Homogeity of Variance
# Normality

tasktwo = tasktwo.apply(pd.to_numeric)


innovation_cols_two = {}
for i in range(1, 7):
    creativity_col = f"T2G{i}-creativity"
    improvement_col = f"T2G{i}-imrpovement"
    innovation_col = f"T2G{i}-innovation"
    innovation_cols_two[innovation_col] = (
        tasktwo[f"T2G{i}-creativity"] + tasktwo[f"T2G{i}-improvement"]
    ) / 2

innovation_df_two = pd.DataFrame(innovation_cols_two)
print(innovation_df_two)


innovation_df_two = innovation_df_two.fillna(innovation_df_two.mean())
innovation_df_two.max
