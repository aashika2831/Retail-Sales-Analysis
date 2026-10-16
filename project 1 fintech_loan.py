import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#task 1 load loan data and check memory
df=pd.read_csv("Fintech_Loan_Risk_Expert_Project.csv")
df.head()
df.info(memory_usage="deep")
memory = df.memory_usage(deep=True).sum()
print("Total memory used:", memory, "bytes")
print("Total memory used:", memory / 1024, "KB")
print("Total memory used:", memory / (1024 ** 2), "MB")

#task 2 reorder columns
print(df.columns.tolist())
new_order = [
    "Loan_ID",
    "Age",
    "Annual_Income",
    "Credit_Score",
    "Employment_Years",
    "Home_Ownership",
    "Loan_Amount",
    "Loan_Term_Months",
    "Interest_Rate",
    "Monthly_EMI",
    "Debt_to_Income_Ratio",
    "Loan_Purpose",
    "Loan_Status"
]
df=df[new_order]
df.head()
print(df.columns.tolist)

#task 3 change columns to correct types
df["Age"]=df["Age"].astype(int)
df["Loan_Term_Months"]=df["Loan_Term_Months"].astype(int)
df["Debt_to_Income_Ratio"]=df["Debt_to_Income_Ratio"].astype(float)

#task 4 check for bad or blank values
print("Missing values:")
print(df.isnull().sum())

print("blank values:")
print(df.isnull().sum())

print("Infinite values")
print(df.isnull().sum())

#task 5 validate busines rules
invalid_income=df[df["Annual_Income"]<= 0]
print("Invalid_income records:",len(invalid_income))

Invalid_Credit=df[
    (df["Credit_Score"]<300)|
    (df["Credit_Score"]>850)]
print("Invalid_Credit score records:",len(Invalid_Credit))

#task 6 key percentage milestones
percentiles = df["Loan_Amount"].quantile(
    [0.25,0.50,0.75,0.90])
print(percentiles)

#task 7 calculate avg income manually
income=df["Annual_Income"].to_numpy()
total_income=np.sum(income)
number_of_people=len(income)
manual_mean = total_income / number_of_people
print("Manual Mean:", manual_mean)
python_mean = df["Annual_Income"].mean()
print("Python Mean:", python_mean)
print("Difference:", manual_mean - python_mean)

#task 8 Calculate variance
mean_income=np.sum(income)
difference=income-mean_income
squared_difference=difference**2
sum_squared_difference=np.sum(squared_difference)
manual_variance = sum_squared_difference / len(income)
print("Manual Variance:", manual_variance)
python_variance = np.var(df["Annual_Income"])
print("Python Variance:", python_variance)

#Task 9 Standard Deviation
manual_std=np.sqrt(manual_variance)
print("Manual Standrd Deviation:",manual_std)
python_std = np.std(df["Annual_Income"])
print("Python Standard Deviation:", python_std)
print("Difference:", manual_std - python_std)

#Task 10 Compare two types of Standard Deviation
population_std = df["Credit_Score"].std(ddof=0)
print("Population Standard Deviation:", population_std)
sample_std = df["Credit_Score"].std(ddof=1)
print("Sample Standard Deviation:", sample_std)
print("Difference:", sample_std - population_std)

#Task 11 Find Middle Credit score
credit_scores = sorted(df["Credit_Score"].tolist())
n = len(credit_scores)
print("Number of scores:", n)
middle1 = credit_scores[n // 2 - 1]
middle2 = credit_scores[n // 2]
manual_median = (middle1 + middle2) / 2
print("Manual Median:", manual_median)
python_median = df["Credit_Score"].median()
print("Python Median:", python_median)

#Task 12 Calculate Standard error 
income_std = np.std(df["Annual_Income"], ddof=1)
n = len(df["Annual_Income"])
standard_error = income_std / np.sqrt(n)
print("Standard Error:", standard_error)

#Task 13 Find Unusually High Debt Ratios
q1 = df["Debt_to_Income_Ratio"].quantile(0.25)
q3 = df["Debt_to_Income_Ratio"].quantile(0.75)
iqr = q3 - q1
print("IQR:", iqr)
upper_limit = q3 + 1.5 * iqr
print("Upper Limit:", upper_limit)
extreme_debt = df[
    df["Debt_to_Income_Ratio"] > upper_limit]
extreme_debt

#Task 14 EXtract Loan Numbers into Arrays
loan_amount = df["Loan_Amount"].to_numpy()
interest_rate = df["Interest_Rate"].to_numpy()
loan_term = df["Loan_Term_Months"].to_numpy()
print(loan_amount[:5])
print(interest_rate[:5])
print(loan_term[:5])

#Task 15 Convert Annual Interest to Monthly
monthly_interest_rate = interest_rate / 100 / 12
print(monthly_interest_rate[:10])

#Task 16 Calculate Monthly EMI
P = loan_amount
r = monthly_interest_rate
n = loan_term
calculated_emi = (
    P * r * (1 + r) ** n
) / (
    (1 + r) ** n - 1
)
print(calculated_emi[:10])

#Task 17 Verify Calculated Payments
comparison = pd.DataFrame({
    "Dataset_EMI": df["Monthly_EMI"],
    "Calculated_EMI": calculated_emi
})
comparison["Difference"] = (
    comparison["Dataset_EMI"] -
    comparison["Calculated_EMI"]
)
print(
    "Maximum Difference:",
    comparison["Difference"].abs().max()
)

#Task 18 Combine Loan Features into Matrix
loan_matrix = df[
    ["Loan_Amount", "Annual_Income", "Monthly_EMI"]
].to_numpy()
print("Shape:", loan_matrix.shape)
column_averages = np.mean(loan_matrix, axis=0)

print("Average Loan Amount:", column_averages[0])
print("Average Annual Income:", column_averages[1])
print("Average Monthly EMI:", column_averages[2])

#Task 19 Calculate Yearly Debt Burden
debt_burden = (
    df["Monthly_EMI"] * 12
) / df["Annual_Income"]
df["Debt_Burden"] = debt_burden
df[[
    "Annual_Income",
    "Monthly_EMI",
    "Debt_Burden"
]].head()

#Task 20 Find Safest and Riskiest Borrows
safest_index = df["Debt_Burden"].idxmin()
print("Safest Row:", safest_index)
print(df.loc[safest_index])
riskiest_index = df["Debt_Burden"].idxmax()
print("Riskiest Row:", riskiest_index)
print(df.loc[riskiest_index])

#Task 21 Create Risk Levels
conditions = [
    (df["Credit_Score"] < 600) |
    (df["Debt_to_Income_Ratio"] > 0.45),

    (df["Credit_Score"] >= 750)]
choices = [
    "High Risk",
    "Low Risk"
] 
df["Risk_Category"] = np.select(
    conditions,
    choices,
    default="Medium Risk"
)

#Task 22 Group loans by Size
df["Loan_Size"] = pd.qcut(
    df["Loan_Amount"],
    q=4,
    labels=[
        "Small",
        "Medium",
        "Large",
        "Very Large"
    ]
)

#Task 23 Default Count by Risk Category
default_count = pd.crosstab(
    df["Risk_Category"],
    df["Loan_Status"]
)
default_count

#Task 24 Default Percentage by Loan Purpose
print(df["Loan_Status"].unique())
df["Default_Flag"] = (
    df["Loan_Status"] == "Defaulted"
).astype(int)
default_percentage = (
    df.groupby("Loan_Purpose")["Default_Flag"]
    .mean() * 100
)
print(default_percentage)

#task 25 Filter High-Risk Defaulters
high_risk_defaulters = df[
    (df["Risk_Category"] == "High Risk") &
    (df["Loan_Status"] == "Defaulted")
]
high_risk_defaulters
print(
    "High-risk defaulters:",
    len(high_risk_defaulters)
)

#Task 26 Calculate total Active Loan Exposure
print(df["Loan_Status"].unique())
active_loans = df[
    df["Loan_Status"] == "Active"
].copy()
active_loans = active_loans.sort_values(
    by="Interest_Rate",
    ascending=False
)
active_loans["Running_Loan_Exposure"] = (
    active_loans["Loan_Amount"].cumsum()
)
active_loans[
    ["Loan_ID",
     "Interest_Rate",
     "Loan_Amount",
     "Running_Loan_Exposure"]
].head(10)
total_active_exposure = active_loans["Loan_Amount"].sum()

print(
    "Total Active Loan Exposure:",
    total_active_exposure
)

#Task 27 Compare Renters vs Owners
print(df["Home_Ownership"].unique())
renters_owners = df[
    df["Home_Ownership"].isin(["Rent", "Own"])
]
default_rates = (
    renters_owners
    .groupby("Home_Ownership")["Default_Flag"]
    .mean() * 100
)
print(default_rates)

#Task 28 Correlation heatmap
correlation_data = df[
    [
        "Annual_Income",
        "Credit_Score",
        "Loan_Amount",
        "Monthly_EMI"
    ]
]
correlation_matrix = correlation_data.corr()
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Loan Risk Correlation Heatmap")
plt.show()

#Task 29 Histogram of Credits Scores
mean_credit = df["Credit_Score"].mean()
median_credit = df["Credit_Score"].median()
plt.figure(figsize=(9, 5))
plt.hist(
    df["Credit_Score"],
    bins=20,
    edgecolor="black"
)
plt.axvline(
    mean_credit,
    linestyle="--",
    label="Mean"
)
plt.axvline(
    median_credit,
    linestyle=":",
    label="Median"
)
plt.xlabel("Credit Score")
plt.ylabel("Number of Borrowers")
plt.title("Distribution of Credit Scores")
plt.legend()
plt.show()

#Task 30 Interest Rates Across Risk Levels
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="Risk_Category",
    y="Interest_Rate"
)
plt.xlabel("Risk Category")
plt.ylabel("Interest Rate")
plt.title("Interest Rates Across Risk Levels")
plt.show()

#Task 31 Income vs Loan amount
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Loan_Amount",
    hue="Loan_Status"
)
plt.xlabel("Annual Income")
plt.ylabel("Loan Amount")
plt.title("Income vs Loan Amount")
plt.legend(title="Loan Status")
plt.show()

#Task 32 Debt Ratio Density curves
plt.figure(figsize=(9, 6))

sns.kdeplot(
    data=df[df["Loan_Status"] == "Defaulted"],
    x="Debt_to_Income_Ratio",
    label="Defaulted",
    fill=True
)

sns.kdeplot(
    data=df[df["Loan_Status"] == "Fully Paid"],
    x="Debt_to_Income_Ratio",
    label="Fully Paid",
    fill=True
)
plt.xlabel("Debt-to-Income Ratio")
plt.ylabel("Density")
plt.title("Debt Ratio Distribution: Defaulted vs Fully Paid")
plt.legend()
plt.show()

#Task 33 horizontal Bar Chart of losses by purpose
df["Unpaid_Loan_Amount"] = np.where(
    df["Loan_Status"] == "Defaulted",
    df["Loan_Amount"],
    0
)
loss_by_purpose = (
    df.groupby("Loan_Purpose")["Unpaid_Loan_Amount"]
    .mean()
    .sort_values()
)
plt.figure(figsize=(9, 6))
plt.barh(
    loss_by_purpose.index,
    loss_by_purpose.values
)
plt.xlabel("Average Unpaid Loan Amount")
plt.ylabel("Loan Purpose")
plt.title("Average Unpaid Loan Amount by Loan Purpose")
plt.tight_layout()
plt.show()

#Task 34 Build a 4-chart risk dashboard
fig, axes = plt.subplots(
    2, 2,
    figsize=(15, 10)
)
#Credit Score Histogram
axes[0, 0].hist(
    df["Credit_Score"],
    bins=20,
    edgecolor="black"
)
axes[0, 0].set_title("Credit Score Distribution")
axes[0, 0].set_xlabel("Credit Score")
axes[0, 0].set_ylabel("Borrowers")
#Debt boxplot
sns.boxplot(
    data=df,
    x="Risk_Category",
    y="Debt_to_Income_Ratio",
    ax=axes[0, 1]
)
axes[0, 1].set_title("Debt Ratio by Risk Level")
#Risk level pie chart
risk_counts = df["Risk_Category"].value_counts()
axes[1, 0].pie(
    risk_counts.values,
    labels=risk_counts.index,
    autopct="%1.1f%%"
)
axes[1, 0].set_title("Risk Category Distribution")
#Income vs Loan
sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Loan_Amount",
    hue="Loan_Status",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Income vs Loan Amount")
plt.tight_layout()
plt.show()

#task 35 save clean data and final summary
df.to_csv(
    "Fintech_Loan_Risk_Final.csv",
    index=False
)
df.to_excel(
    "Fintech_Loan_Risk_Final.xlsx",
    index=False
)
print("Final CSV and Excel files saved successfully.")
#finding 1
print(df["Risk_Category"].value_counts())
#finding 2
overall_default_rate = df["Default_Flag"].mean() * 100
print(
    "Overall Default Rate:",
    overall_default_rate,
    "%"
)
#finding 3
purpose_default_rate = (
    df.groupby("Loan_Purpose")["Default_Flag"]
    .mean() * 100
)
print(
    purpose_default_rate.sort_values(
        ascending=False
    )
)



