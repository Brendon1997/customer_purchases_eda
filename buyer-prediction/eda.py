import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Function to perform EDA
def perform_eda(df):
    print("Starting EDA...")
    check_missing_values(df)
    show_data_overview(df)
    plot_distributions(df)
    analyze_correlations(df)
    print("EDA Completed!")


# Function to check missing values
def check_missing_values(df):
    print("Checking for missing values...")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_info = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage})
    missing_info = missing_info[missing_info['Missing Values'] > 0]
    if missing_info.empty:
        print("No missing values found.")
    else:
        print("Columns with missing values:")
        print(missing_info)
    print()


# Function to show basic data overview
def show_data_overview(df):
    print("Data Overview:")
    print("First 5 rows of the data:")
    print(df.head())
    print("\nBasic Statistics:")
    print(df.describe(include='all'))
    print("\nData Types:")
    print(df.dtypes)
    print()


# Function to plot data distributions
def plot_customer_transaction_distribution(df):
    # Number of transactions per customer
    customer_transaction_counts = df['customer_id'].value_counts()

    # Plot the distribution of transaction counts
    sns.histplot(customer_transaction_counts, bins=20, kde=True)
    plt.title('Distribution of Number of Transactions per Customer')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Number of Customers')
    plt.show()


def plot_distributions(df):
    print("Plotting distributions...")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    # Plot numerical features
    for col in numerical_cols:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

    plot_customer_transaction_distribution(df)

    columns_to_drop = ['invoice_no', 'customer_id', 'invoice_date']
    categorical_cols = categorical_cols.drop(columns=columns_to_drop).columns

    # Plot categorical features
    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Count of {col}')
        plt.show()
    print()


# Function to analyze correlations between numerical features
def analyze_correlations(df):
    print("Analyzing correlations...")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()
    else:
        print("Not enough numerical features to compute correlations.")
    print()


# Sample data loading and EDA execution
customer_data_path = 'data/customer_shopping_data.csv'
df = pd.read_csv(customer_data_path)

# Perform EDA
perform_eda(df)