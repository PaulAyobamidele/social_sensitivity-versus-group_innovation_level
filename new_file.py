# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

# %%
df = pd.read_csv(
    "C:/Users/user/Desktop/SCI Class Folder/Semester Two/Cogitive Science 2/final_paper/newdata.csv"
)

# %%
# Set pandas display options to show all columns and rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_colwidth", None)  # Show full column width
pd.set_option("display.expand_frame_repr", False)  # Prevent line wrapping

# %%
df = df.iloc[2:]

# %%
df

# %%
df.columns

# %%
df.drop(
    columns=[
        "StartDate",
        "EndDate",
        "Status",
        "IPAddress",
        "Progress",
        "Finished",
        "RecordedDate",
        "ResponseId",
        "RecipientLastName",
        "RecipientFirstName",
        "RecipientEmail",
        "ExternalReference",
        "LocationLatitude",
        "LocationLongitude",
        "UserLanguage",
        "consent form",
        "Introduction",
        "End of Survey",
    ],
    inplace=True,
)
df.columns

# %%

df

# %%

# Change the data types of the specified columns to float
df["Duration (in seconds)"] = df["Duration (in seconds)"].astype("float64")
df.info()

# %%
df = df[df["Duration (in seconds)"] > 60.0]

# %%
df = df[df["DistributionChannel"] == "anonymous"]

# %%
df

# %%
df.info()

# %%
task1_password = [
    "Duration (in seconds)",
    "Task1-grp1_1",
    "Task1-grp1_2",
    "Task1-grp2_1",
    "Task1-grp2_2",
    "Task1-grp3_1",
    "Task1-grp3_2",
    "Task1-grp4_1",
    "Task1-grp4_2",
    "Task1-grp5_1",
    "Task1-grp5_2",
    "Task1-grp6_1",
    "Task1-grp6_2",
]


df_pass = df[task1_password].copy()

# %%
df_pass

# %%
df_pass.info()


# %%
df_pass.dropna(inplace=True)

# %%
df_pass

# %%
task2_alarm = [
    "Duration (in seconds)",
    "Task2-grp1_1",
    "Task2-grp1_2",
    "Task2-grp2_1",
    "Task2-grp2_2",
    "Task2-grp3_1",
    "Task2-grp3_2",
    "Task2-grp4_1",
    "Task2-grp4_2",
    "Task2-grp5_1",
    "Task2-grp5_2",
    "Task2-grp6_1",
    "Task2-grp6_2",
]


df_alarm = df[task2_alarm].copy()

# %%
df_alarm.info()

# %%
df_alarm.dropna(inplace=True)

# %%
print("df pass info", df_pass.info())
print("df alarm info", df_alarm.info())


# %%
def innov_score(df, col1, col2, new_col_name):
    df.loc[:, new_col_name] = df[[col1, col2]].astype("float64").mean(axis=1)
    return df


# %%
df_pass = innov_score(df_pass, "Task1-grp1_1", "Task1-grp1_2", "inov_grp1_task_1")
df_pass = innov_score(df_pass, "Task1-grp2_1", "Task1-grp2_2", "inov_grp2_task_1")
df_pass = innov_score(df_pass, "Task1-grp3_1", "Task1-grp3_2", "inov_grp3_task_1")
df_pass = innov_score(df_pass, "Task1-grp4_1", "Task1-grp4_2", "inov_grp4_task_1")
df_pass = innov_score(df_pass, "Task1-grp5_1", "Task1-grp5_2", "inov_grp5_task_1")
df_pass = innov_score(df_pass, "Task1-grp6_1", "Task1-grp6_2", "inov_grp6_task_1")

# %%
df_pass.head()

# %%
df_alarm = innov_score(df_alarm, "Task2-grp1_1", "Task2-grp1_2", "inov_grp1_task_2")
df_alarm = innov_score(df_alarm, "Task2-grp2_1", "Task2-grp2_2", "inov_grp2_task_2")
df_alarm = innov_score(df_alarm, "Task2-grp3_1", "Task2-grp3_2", "inov_grp3_task_2")
df_alarm = innov_score(df_alarm, "Task2-grp4_1", "Task2-grp4_2", "inov_grp4_task_2")
df_alarm = innov_score(df_alarm, "Task2-grp5_1", "Task2-grp5_2", "inov_grp5_task_2")
df_alarm = innov_score(df_alarm, "Task2-grp6_1", "Task2-grp6_2", "inov_grp6_task_2")

# %%
df_alarm

# %%


# %%
df_innov_task2 = df_alarm[
    [
        "inov_grp1_task_2",
        "inov_grp2_task_2",
        "inov_grp3_task_2",
        "inov_grp4_task_2",
        "inov_grp5_task_2",
        "inov_grp6_task_2",
    ]
]

# %%
df_innov_task2

# %%


# %%
df_innov_task1 = df_pass[
    [
        "inov_grp1_task_1",
        "inov_grp2_task_1",
        "inov_grp3_task_1",
        "inov_grp4_task_1",
        "inov_grp5_task_1",
        "inov_grp6_task_1",
    ]
]

# %%
df_innov_task2.info()
df_innov_task1.info()

# %%
df_innov_task1.reset_index(drop=True, inplace=True)
df_innov_task2.reset_index(drop=True, inplace=True)


avg_innov_task1 = df_innov_task1.mean()

avg_innov_task1.describe()
# Calculate average innovation scores for each group in Task 2
avg_innov_task2 = df_innov_task2.mean()


avg_innov_task2.describe()
# Display the results
print("Average Innovation Scores for Task 1 (Password Innovation Task):")
print(avg_innov_task1)

print("\nAverage Innovation Scores for Task 2 (Alarm Clock Innovation Task):")
print(avg_innov_task2)

custom_labels_task1 = ["Group 1", "Group 2", "Group 3", "Group 4", "Group 5", "Group 6"]
custom_labels_task2 = ["Group 1", "Group 2", "Group 3", "Group 4", "Group 5", "Group 6"]
# Bar plot for Task 1
plt.figure(figsize=(6, 4))
avg_innov_task1.plot(kind="bar", color="#F15A24")
plt.bar(
    avg_innov_task1.index,
    avg_innov_task1.values,
    color="#F15A24",
    tick_label=custom_labels_task1,
)
plt.title("Average Innovation Scores for Task 1 (Password Innovation)")
plt.xlabel("Groups")
plt.ylabel("Average Innovation Score")
plt.ylim(
    0, max(avg_innov_task1.max(), avg_innov_task2.max()) + 1
)  # Adjust y-axis for better comparison
plt.show()

# Bar plot for Task 2
plt.figure(figsize=(6, 4))
plt.bar(
    avg_innov_task2.index,
    avg_innov_task2.values,
    color="#F15A24",
    tick_label=custom_labels_task2,
)
plt.title("Average Innovation Scores for Task 2 (Alarm Clock Innovation)")
plt.xlabel("Groups")
plt.ylabel("Average Innovation Score")
plt.ylim(
    0, max(avg_innov_task1.max(), avg_innov_task2.max()) + 1
)  # Adjust y-axis for better comparison
plt.show()


# %%
df_innov = pd.concat([df_innov_task1, df_innov_task2], axis=1)

# %%
df_innov.info()
df_innov.describe()

# %%
df_innov.fillna(df_innov.mean(), inplace=True)


# %%
df_innov.describe()

# %%
df_mean_inov = pd.DataFrame()

# %%
df_mean_inov["grp1"] = df_innov["inov_grp1_task_1"] + df_innov["inov_grp1_task_2"]
df_mean_inov["grp2"] = df_innov["inov_grp2_task_1"] + df_innov["inov_grp2_task_2"]
df_mean_inov["grp3"] = df_innov["inov_grp3_task_1"] + df_innov["inov_grp3_task_2"]
df_mean_inov["grp4"] = df_innov["inov_grp4_task_1"] + df_innov["inov_grp4_task_2"]
df_mean_inov["grp5"] = df_innov["inov_grp5_task_1"] + df_innov["inov_grp5_task_2"]
df_mean_inov["grp6"] = df_innov["inov_grp6_task_1"] + df_innov["inov_grp6_task_2"]

# %%


# %%
df_mean_inov

# %%
df_mean_inov.describe()


plt.style.use("default")

# Create the boxplot with additional customizations
boxprops = dict(linestyle="-", linewidth=2)
medianprops = dict(linestyle="-", linewidth=2, color="red")
meanprops = dict(marker="o", markerfacecolor="green", markeredgecolor="black")

fig, ax = plt.subplots()
bp = ax.boxplot(
    [df_mean_inov["grp1"], df_mean_inov["grp6"]],
    labels=["Group 1", "Group 6"],
    boxprops=boxprops,
    medianprops=medianprops,
    meanprops=meanprops,
    showmeans=True,
)

# Add title and labels
plt.title("Box Plot of Group 1 and Group 6")
plt.xlabel("Groups")
plt.ylabel("Innovative Score")

# Custom legend
import matplotlib.lines as mlines

# Create custom legend handles
box_handle = mlines.Line2D([], [], linestyle="-", linewidth=2, label="Box")
median_handle = mlines.Line2D(
    [], [], color="red", linestyle="-", linewidth=2, label="Median"
)
mean_handle = plt.Line2D(
    [], [], color="black", marker="o", markerfacecolor="green", label="Mean"
)

# Add the legend to the plot
ax.legend(handles=[median_handle, mean_handle], loc="upper right")

# Display the plot
plt.show()
# %%
df_mean_inov.to_csv("innovation.csv", index=False)


df_mean_inov.describe()

# %%
from scipy import stats

# Test for Normality
# Shapiro Wilk Test

shapiro_grp1 = stats.shapiro(df_mean_inov["grp1"])
shapiro_grp6 = stats.shapiro(df_mean_inov["grp6"])

t_stat1, p_value1 = stats.ttest_ind(df_mean_inov["grp1"], df_mean_inov["grp6"])


#
# Output the results
print(f"T-statistic: {t_stat1}, P-value: {p_value1}")

# %%
# Test for Equal Variance
# Levene's Test:

# Perform Levene's test for equal variances
levene_test = stats.levene(df_mean_inov["grp1"], df_mean_inov["grp6"])

# Output the results
print(
    f"Levene's Test for equal variances: Statistic={levene_test.statistic}, P-value={levene_test.pvalue}"
)


# t test

# Perform t-test (equal variances or not based on Levene's test)
if levene_test.pvalue > 0.05:
    # If p-value > 0.05, assume equal variances
    t_stat, p_value = stats.ttest_ind(
        df_mean_inov["grp1"], df_mean_inov["grp6"], equal_var=True
    )
else:
    # If p-value <= 0.05, assume unequal variances
    t_stat, p_value = stats.ttest_ind(
        df_mean_inov["grp1"], df_mean_inov["grp6"], equal_var=False
    )

# Output the results
print(f"T-statistic: {t_stat}, P-value: {p_value}")


# %%

t_stat = 1.8170732074176326
p_value = 0.07170135807821133
df = len(df_mean_inov["grp1"]) + len(df_mean_inov["grp6"]) - 2  # degrees of freedom

# Generate x values for the t-distribution
x = np.linspace(-4, 4, 1000)
t_dist = stats.t(df)

# Plot the t-distribution
plt.plot(x, t_dist.pdf(x), label=f"t-distribution (df={df})")

# Mark the critical t-values for a two-tailed test at alpha = 0.05
alpha = 0.05
crit_val_left = t_dist.ppf(alpha / 2)
crit_val_right = t_dist.ppf(1 - alpha / 2)
plt.axvline(
    crit_val_left,
    color="red",
    linestyle="dashed",
    label=f"Critical value ({alpha/2} tail)",
)
plt.axvline(crit_val_right, color="red", linestyle="dashed")

# Mark the t-statistic
plt.axvline(
    t_stat, color="blue", linestyle="solid", label=f"T-statistic = {t_stat:.2f}"
)

# Fill the rejection regions
plt.fill_between(
    x,
    0,
    t_dist.pdf(x),
    where=(x < crit_val_left) | (x > crit_val_right),
    color="red",
    alpha=0.2,
)

# Add labels and legend
plt.xlabel("T value")
plt.ylabel("Probability Density")
plt.title("T-Distribution and T-Test Result")
plt.legend()
plt.show()
# %%
import seaborn as sns


# Plot
plt.figure(figsize=(12, 8))
sns.histplot(
    data=df_mean_inov["grp1"], bins=15, kde=True, multiple="stack", palette="tab10"
)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Histogram of Group Values")
plt.legend(title="Group")
plt.grid(True)


# Plot
plt.figure(figsize=(12, 8))
sns.histplot(
    data=df_mean_inov["grp2"], bins=15, kde=True, multiple="stack", palette="tab10"
)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Histogram of Group Values")
plt.legend(title="Group")
plt.grid(True)


# Plot
plt.figure(figsize=(12, 8))
sns.histplot(
    data=df_mean_inov["grp3"], bins=15, kde=True, multiple="stack", palette="tab10"
)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Histogram of Group Values")
plt.legend(title="Group")
plt.grid(True)


# Plot
plt.figure(figsize=(12, 8))
sns.histplot(
    data=df_mean_inov["grp4"], bins=15, kde=True, multiple="stack", palette="tab10"
)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Histogram of Group Values")
plt.legend(title="Group")
plt.grid(True)


# Plot
plt.figure(figsize=(12, 8))
sns.histplot(
    data=df_mean_inov["grp5"], bins=15, kde=True, multiple="stack", palette="tab10"
)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Histogram of Group Values")
plt.legend(title="Group")
plt.grid(True)

# Plot
plt.figure(figsize=(12, 8))
sns.histplot(
    data=df_mean_inov["grp6"], bins=15, kde=True, multiple="stack", palette="tab10"
)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Histogram of Group Values")
plt.legend(title="Group")
plt.grid(True)

plt.show()

# %%
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# One-way ANOVA
anova_result = stats.f_oneway(
    df_mean_inov["grp1"],
    df_mean_inov["grp2"],
    df_mean_inov["grp3"],
    df_mean_inov["grp4"],
    df_mean_inov["grp5"],
    df_mean_inov["grp6"],
)
print(f"ANOVA F-statistic: {anova_result.statistic}, P-value: {anova_result.pvalue}")


df_between = len(df_mean_inov.columns) - 1  # Between-groups degrees of freedom
df_within = len(df_mean_inov.values.flatten()) - len(
    df_mean_inov.columns
)  # Within-groups degrees of freedom

# Calculate the critical F-value
alpha = 0.05  # significance level
f_critical = stats.f.ppf(1 - alpha, df_between, df_within)

# Generate the F-distribution values
x = np.linspace(0, 5, 1000)
y = stats.f.pdf(x, df_between, df_within)

# Plot the F-distribution
plt.plot(x, y, label=f"F-distribution df=({df_between}, {df_within})")
plt.axvline(
    f_critical, color="red", linestyle="--", label=f"Critical F-value: {f_critical:.2f}"
)
plt.axvline(
    anova_result.statistic,
    color="blue",
    linestyle="-",
    label=f"Observed F-statistic: {anova_result.statistic:.2f}",
)
plt.fill_between(x, y, where=x > f_critical, color="red", alpha=0.2)
plt.title("F-Distribution with Critical Value and Observed F-Statistic (ANOVA)")
plt.xlabel("F value")
plt.ylabel("Probability Density")
plt.legend()
plt.show()

# If the ANOVA is significant, perform post-hoc analysis
if anova_result.pvalue < 0.05:
    # Melt the DataFrame for post-hoc analysis
    df_melted = df.melt(var_name="Group", value_name="InnovationLevel")

    # Post-hoc Tukey HSD test
    mc = sm.stats.multicomp.MultiComparison(
        df_melted["InnovationLevel"], df_melted["Group"]
    )
    tukey_result = mc.tukeyhsd()
    print(tukey_result)

    # Post-hoc Dunn's test with Bonferroni correction
    dunn_result = sp.posthoc_dunn(df, p_adjust="bonferroni")
    print(dunn_result)
else:
    print("No significant differences found between the groups based on ANOVA.")


# %%
# Descriptive Statistics
descriptive_stats = df_mean_inov.describe()
print(descriptive_stats)

# Boxplot for visualizing distribution of each group
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_mean_inov)
plt.title("Boxplot of Innovation Levels by Group")
plt.xlabel("Group")
plt.ylabel("Innovation Level")
plt.show()

# Histogram for each group
df_mean_inov.hist(bins=20, figsize=(14, 10))
plt.suptitle("Histograms of Innovation Levels by Group")
plt.show()

# Pairplot for visualizing relationships
sns.pairplot(df_mean_inov)
plt.suptitle("Pairplot of Innovation Levels by Group", y=1.02)
plt.show()


# %%


# %%


# Calculate the correlation matrix for task 1
correlation_matrix_task1 = df_pass.corr()
print("Correlation Matrix - Task 1:\n", correlation_matrix_task1)

# Calculate the correlation matrix for task 2
correlation_matrix_task2 = df_alarm.corr()
print("Correlation Matrix - Task 2:\n", correlation_matrix_task2)

# Combined correlation matrix for both tasks
correlation_matrix_combined = df_innov.corr()
print("Correlation Matrix - Combined:\n", correlation_matrix_combined)


# %%
def calculate_ave(df, indicators):
    # Convert columns to numeric, coercing errors to NaNs
    df[indicators] = df[indicators].apply(pd.to_numeric, errors="coerce")
    # Drop rows with NaNs in the relevant columns
    df = df.dropna(subset=indicators)
    # Calculate the average variance extracted (AVE)
    return df[indicators].var().mean()


# Example for group 1
ave_grp1_task1 = calculate_ave(df_pass, ["Task1-grp1_1", "Task1-grp1_2"])
ave_grp1_task2 = calculate_ave(df_alarm, ["Task2-grp1_1", "Task2-grp1_2"])

# Square root of AVE
sqrt_ave_grp1_task1 = np.sqrt(ave_grp1_task1)
sqrt_ave_grp1_task2 = np.sqrt(ave_grp1_task2)

print("Square root of AVE - Task 1 Group 1:", sqrt_ave_grp1_task1)
print("Square root of AVE - Task 2 Group 1:", sqrt_ave_grp1_task2)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap for combined correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_combined, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap")
plt.show()
# %%


# demographic data

df.columns
# %%
demographic_columns = ["Age", "Gender", "Education_level"]
# %%

# Create a DataFrame with demographic data
df_demographics = df[demographic_columns].copy()

# Display first few rows of demographic data
df_demographics.head()
# %%
# Descriptive statistics for demographic data
df_demographics.describe(include="all")
# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Age distribution
df_demographics["Age"] = pd.to_numeric(df_demographics["Age"], errors="coerce")
age_bins = [0, 18, 30, 45, 60, 100]
age_labels = ["<18", "18-29", "30-44", "45-59", "60+"]
df_demographics["AgeGroup"] = pd.cut(
    df_demographics["Age"], bins=age_bins, labels=age_labels, right=False
)

plt.figure(figsize=(10, 6))
sns.histplot(df_demographics["Age"].dropna(), bins=50, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Gender distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="Gender", data=df_demographics)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Education distribution
plt.figure(figsize=(12, 6))
sns.countplot(x="Education_level", data=df_demographics)
plt.title("Education Level Distribution")
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# Ensure Age column is numeric
df_demographics["Age"] = pd.to_numeric(df_demographics["Age"], errors="coerce")

# Plot histogram for Age with fewer bins
plt.figure(figsize=(10, 6))
sns.histplot(df_demographics["Age"].dropna(), bins=10, kde=True)  # Reduced bins to 10
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# %%
# Define age bins and labels
age_bins = [0, 18, 30, 45, 60, 100]
age_labels = ["<18", "18-29", "30-44", "45-59", "60+"]
df_demographics["AgeGroup"] = pd.cut(
    df_demographics["Age"], bins=age_bins, labels=age_labels, right=False
)

# Plot count of each AgeGroup
plt.figure(figsize=(10, 6))
sns.countplot(x="AgeGroup", data=df_demographics)
plt.title("Age Group Distribution")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.show()


# Merge innovation data with demographic data
df_merged = pd.concat([df_innov, df_demographics], axis=1)

# Boxplot of innovation scores by age group
plt.figure(figsize=(12, 6))
sns.boxplot(x="AgeGroup", y="inov_grp6_task_1", data=df_merged)
plt.title("Innovation Score Group 1 Task 1 by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Innovation Score")
plt.show()


# %%
# Expert

expert_taskone = {
    "group1_creativity": [5],
    "group1_improvement": [2],
    "group2_creativity": [6],
    "group2_improvement": [2],
    "group3_creativity": [4],
    "group3_improvement": [4],
    "group4_creativity": [6],
    "group4_improvement": [5],
    "group5_creativity": [5],
    "group5_improvement": [6],
    "group6_creativity": [3],
    "group6_improvement": [1],
}

df_expert_alarm = pd.DataFrame(expert_taskone)


expert_tasktwo = {
    "group1_creativity": [5],
    "group1_improvement": [5],
    "group2_creativity": [4],
    "group2_improvement": [4],
    "group3_creativity": [2],
    "group3_improvement": [2],
    "group4_creativity": [2],
    "group4_improvement": [2],
    "group5_creativity": [3],
    "group5_improvement": [3],
    "group6_creativity": [5],
    "group6_improvement": [3],
}

df_expert_password = pd.DataFrame(expert_tasktwo)


def expert_innov_score(df, col1, col2, new_col_name):
    df[new_col_name] = df[[col1, col2]].astype("float64").mean(axis=1)
    return df


# Apply the function to calculate innovation levels for each group in task one (alarm)
df_expert_alarm = expert_innov_score(
    df_expert_alarm,
    "group1_creativity",
    "group1_improvement",
    "group1_innovation_level",
)
df_expert_alarm = expert_innov_score(
    df_expert_alarm,
    "group2_creativity",
    "group2_improvement",
    "group2_innovation_level",
)
df_expert_alarm = expert_innov_score(
    df_expert_alarm,
    "group3_creativity",
    "group3_improvement",
    "group3_innovation_level",
)
df_expert_alarm = expert_innov_score(
    df_expert_alarm,
    "group4_creativity",
    "group4_improvement",
    "group4_innovation_level",
)
df_expert_alarm = expert_innov_score(
    df_expert_alarm,
    "group5_creativity",
    "group5_improvement",
    "group5_innovation_level",
)
df_expert_alarm = expert_innov_score(
    df_expert_alarm,
    "group6_creativity",
    "group6_improvement",
    "group6_innovation_level",
)

# Apply the function to calculate innovation levels for each group in task two (password)
df_expert_password = expert_innov_score(
    df_expert_password,
    "group1_creativity",
    "group1_improvement",
    "group1_innovation_level",
)
df_expert_password = expert_innov_score(
    df_expert_password,
    "group2_creativity",
    "group2_improvement",
    "group2_innovation_level",
)
df_expert_password = expert_innov_score(
    df_expert_password,
    "group3_creativity",
    "group3_improvement",
    "group3_innovation_level",
)
df_expert_password = expert_innov_score(
    df_expert_password,
    "group4_creativity",
    "group4_improvement",
    "group4_innovation_level",
)
df_expert_password = expert_innov_score(
    df_expert_password,
    "group5_creativity",
    "group5_improvement",
    "group5_innovation_level",
)
df_expert_password = expert_innov_score(
    df_expert_password,
    "group6_creativity",
    "group6_improvement",
    "group6_innovation_level",
)

# Display the resulting dataframes
print("Expert Alarm Innovation Levels:\n", df_expert_alarm)
print("Expert Password Innovation Levels:\n", df_expert_password)


expert_mean_inov = pd.DataFrame()

expert_mean_inov["grp1"] = (
    df_expert_alarm["group1_innovation_level"]
    + df_expert_password["group1_innovation_level"]
)
expert_mean_inov["grp2"] = (
    df_expert_alarm["group2_innovation_level"]
    + df_expert_password["group2_innovation_level"]
)
expert_mean_inov["grp3"] = (
    df_expert_alarm["group3_innovation_level"]
    + df_expert_password["group3_innovation_level"]
)
expert_mean_inov["grp4"] = (
    df_expert_alarm["group4_innovation_level"]
    + df_expert_password["group4_innovation_level"]
)
expert_mean_inov["grp5"] = (
    df_expert_alarm["group4_innovation_level"]
    + df_expert_password["group4_innovation_level"]
)
expert_mean_inov["grp6"] = (
    df_expert_alarm["group6_innovation_level"]
    + df_expert_password["group6_innovation_level"]
)


expert_mean_inov.dropna()

expert_mean_inov
# %%


crowd_mean_inov = df_mean_inov.mean().to_frame().T
crowd_mean_inov.index = ["Crowd"]

# Expert mean innovation scores
expert_mean_inov = pd.DataFrame(
    {
        "grp1": [8.5],
        "grp2": [8.0],
        "grp3": [6.0],
        "grp4": [7.5],
        "grp5": [7.5],
        "grp6": [6.0],
    },
    index=["Expert"],
)

# Display the dataframes
print("Crowd Mean Innovation Scores:\n", crowd_mean_inov)
print("\nExpert Mean Innovation Scores:\n", expert_mean_inov)


combined_mean_inov = pd.concat([crowd_mean_inov, expert_mean_inov])
combined_mean_inov = combined_mean_inov.T  # Transpose for better visualization


# weighting the expert
crowd_weight = 0.3
expert_weight = 0.7

# Calculate combined score using weighted average
combined_mean_inov["combined_score"] = (combined_mean_inov["Crowd"] * crowd_weight) + (
    combined_mean_inov["Expert"] * expert_weight
)

# Display the DataFrame with combined scores
print(df)

# Display the combined dataframe
print("\nCombined Mean Innovation Scores:\n", combined_mean_inov)
# %%
# Bar plot to compare crowd and expert evaluations
plt.figure(figsize=(12, 6))
combined_mean_inov.plot(kind="bar", figsize=(12, 6))
plt.title("Comparison of Crowd and Expert Innovation Scores")
plt.xlabel("Groups")
plt.ylabel("Innovation Scores")
plt.xticks(rotation=0)
plt.legend(title="Evaluator")
plt.show()

# %%
# Calculate the difference between crowd and expert evaluations
difference = crowd_mean_inov.iloc[0] - expert_mean_inov.iloc[0]
difference = difference.to_frame(name="Difference")

# Display the differences
print("\nDifferences between Crowd and Expert Evaluations:\n", difference)

# Statistical summary
summary = combined_mean_inov.describe()
print("\nStatistical Summary:\n", summary)

# %%
# CORRELATION

crowd_means = df_mean_inov.mean()

# Convert expert evaluation to Series for easy comparison
expert_means = expert_mean_inov.iloc[0]

# Combine both evaluations into a single DataFrame for comparison
comparison_df = pd.DataFrame(
    {"Crowd Evaluation": crowd_means, "Expert Evaluation": expert_means}
)


plt.figure(figsize=(12, 6))
comparison_df.plot(kind="bar")
plt.title("Comparison of Crowd and Expert Evaluations")
plt.xlabel("Groups")
plt.ylabel("Innovation Score")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

# Display the comparison DataFrame and correlation
print(comparison_df)
print(f"Correlation between Crowd and Expert Evaluations: {correlation:.2f}")

correlation = comparison_df.corr().iloc[0, 1]
# %%
# Correlation Coefficient:

# The correlation coefficient quantifies the degree to which the crowd's and expert's evaluations are linearly related.
# A value close to 1 implies a strong positive correlation, meaning the crowd's and expert's evaluations tend to agree.
# A value close to 0 implies little to no linear relationship, indicating that the crowd's and expert's evaluations do not align well.
# A value close to -1 implies a strong negative correlation, meaning the crowd's and expert's evaluations tend to disagree.
# Explanation of Innovation Level:

# A high positive correlation suggests that the crowd can reliably evaluate innovation levels similarly to experts, indicating that the crowd's judgment can be trusted.
# A low or negative correlation suggests that the crowd's evaluations diverge from the expert's, which could imply differences in understanding, criteria, or evaluation standards between the two groups.


survey = pd.read_csv("rmet_survey.csv")

# Create a DataFrame
survey = pd.DataFrame(survey)

# Clean the data: Selecting the relevant columns (assuming the dataset starts from the 4th row and 19th column)

survey_cleaned_data = survey.iloc[3:, 18:]

survey_cleaned_data = survey_cleaned_data.dropna()

survey_cleaned_data.describe()

len(survey_cleaned_data)


# Rename the columns appropriately
survey_cleaned_data.columns = [
    "positive_atmosphere",
    "level_communications",
    "value_contributions",
    "group_attentiveness",
    "mode_aggregation",
    "group_label",
    "gender",
]

# Convert the necessary columns to numeric
cols_to_convert = [
    "positive_atmosphere",
    "level_communications",
    "value_contributions",
    "group_attentiveness",
]

survey_cleaned_data.groupby("mode_aggregation").size()

unique_modes = survey_cleaned_data["mode_aggregation"].unique()
print(unique_modes)

# Create a dictionary for mapping
mode_mapping = {
    "By a dominant individual": "Dominant Individual",
    "By voting": "Voting",
    "Through consensus building and discussion": "Discussion",
    "randomly": "Randomly",
}

# Replace mode_aggregation values with new labels
survey_cleaned_data["mode_aggregation"] = survey_cleaned_data["mode_aggregation"].map(
    mode_mapping
)

# Plotting the distribution of mode_aggregation
plt.figure(figsize=(10, 6))
sns.countplot(x="mode_aggregation", data=survey_cleaned_data, palette="viridis")
plt.title("Distribution of the Mode of Aggregation")
plt.xlabel("Mode of Aggregation")
plt.ylabel("Count")
plt.xticks(rotation=0, ha="right")
plt.tight_layout()
plt.show()

for col in cols_to_convert:
    survey_cleaned_data[col] = pd.to_numeric(survey_cleaned_data[col], errors="coerce")

# Drop rows with missing values
survey_cleaned_data = survey_cleaned_data.dropna()

# Display the first few rows of the cleaned data
print(survey_cleaned_data.head())


# Group by gender and calculate the mean of each numerical column
gender_grouped = survey_cleaned_data.groupby("gender").mean()
print(gender_grouped)

# Plotting: Boxplots for numerical features by gender individually
for col in survey_cleaned_data.columns[:-2]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="gender", y=col, data=survey_cleaned_data)
    plt.title(f"Boxplot of {col} by Gender")
    plt.show()

# Group by group label and gender, then calculate the mean
grouped_data = (
    survey_cleaned_data.groupby(["group_label", "gender"]).mean().reset_index()
)

# Plotting: Mean positive atmosphere by group label and gender
plt.figure(figsize=(10, 6))
sns.barplot(x="group_label", y="positive_atmosphere", hue="gender", data=grouped_data)
plt.title("Mean Positive Atmosphere by Group Label and Gender")
plt.xlabel("Group Label")
plt.ylabel("Mean Positive Atmosphere")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# %%
label_grouped = survey_cleaned_data.groupby("group_label").mean()

print(label_grouped)
# Plotting: Boxplots for numerical features by gender individually


# %%
filtered_groups = label_grouped.loc[["group 1", "group 2", "group 5", "group 6"]]

mode_aggregation = filtered_groups.groupby(["mode_aggregation", "group_label"]).mean()
print(mode_aggregation)

# Print the filtered data
print(filtered_groups)

# Plotting: Bar plots for the selected groups
plt.figure(figsize=(10, 6))
sns.barplot(x=filtered_groups.index, y=filtered_groups["positive_atmosphere"])
plt.title("A barplot of Cohension Level per group")
plt.xlabel("Group Label")
plt.ylabel("Mean Positive Atmosphere")
plt.show()

# Plot for level of communications
plt.figure(figsize=(10, 6))
sns.barplot(x=filtered_groups.index, y=filtered_groups["level_communications"])
plt.title("A barplot of Communication Level per group")
plt.xlabel("Group Label")
plt.ylabel("Mean Level of Communications")
plt.show()

# Plot for value of contributions
plt.figure(figsize=(10, 6))
sns.barplot(x=filtered_groups.index, y=filtered_groups["value_contributions"])
plt.title("A barplot of Perceived Contribution Value per group")
plt.xlabel("Group Label")
plt.ylabel("Mean Value of Contributions")
plt.show()

# Plot for group attentiveness
plt.figure(figsize=(10, 6))
sns.barplot(x=filtered_groups.index, y=filtered_groups["group_attentiveness"])
plt.title("A barplot of Responsiveness Level per group")
plt.xlabel("Group Label")
plt.ylabel("Mean Group Attentiveness")
plt.show()


filtered_groups = survey_cleaned_data[
    survey_cleaned_data["group_label"].isin(
        ["group 1", "group 2", "group 3", "group 4", "group 5", "group 6"]
    )
]

# Calculate the distribution of mode of aggregation for each group
mode_aggregation_distribution = (
    filtered_groups.groupby(["group_label", "mode_aggregation"])
    .size()
    .reset_index(name="count")
)

# Print the distribution as a table
print("Distribution of Mode of Aggregation for Filtered Groups:")
print(mode_aggregation_distribution)

# Plotting: Bar plots for the selected groups
plt.figure(figsize=(12, 8))
sns.barplot(
    x="group_label",
    y="count",
    hue="mode_aggregation",
    data=mode_aggregation_distribution,
    palette="viridis",
)
plt.title("Distribution of Mode of Aggregation for Filtered Groups")
plt.xlabel("Group Label")
plt.ylabel("Count")
plt.legend(title="Mode of Aggregation")
plt.tight_layout()
plt.show()
