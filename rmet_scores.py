import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("RMET.csv")
df["name"] = df["name"].str.lower()

df.head()

# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x="social_sensitivity", y="score", data=df)
plt.title("Comparison of (RMET scores) by category")
plt.xlabel("Category based on RMET")
plt.ylabel("Scores")
plt.show()


df.columns
group_df = df.groupby("social_sensitivity")["score"].describe()
