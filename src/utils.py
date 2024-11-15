import ast
from numpy import split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_dict_column(column):
    """Parses a column containing dictionary-like strings and extracts the values."""
    try:
        # Safely evaluate the string to a Python dictionary
        parsed_col = ast.literal_eval(column)
        # Join the values if the evaluation returns a dictionary
        if isinstance(parsed_col, dict):
            return '; '.join(parsed_col.values())
    except (ValueError, SyntaxError):
        return column  # Return the original if parsing fails


def clean_language_format(language_data):
    # Function to clean up the language format safely
    if pd.isna(language_data):
        return language_data
    try:
        # Parse the string to a dictionary
        language_dict = ast.literal_eval(language_data)
        # Clean up the keys and values
        return {key.replace(" Language", "").strip(): value.replace(" Language", "").strip() for key, value in
                language_dict.items()}
    except (ValueError, SyntaxError):
        # If parsing fails, handle it as a raw string
        return language_data.replace(" Language", "").strip()


def proportion_missing_values(data, feature, group, adaptation=False):
    return data[data['movie_is_adaptation'] == adaptation].groupby(group)[feature].apply(lambda x: x.isna().sum())


def plot_missing_revenues_budget(y1, y2, y3, y4):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(y1.index, y1, label='Missing Revenue Adapted', color='red')
    ax1.plot(y2.index, y2, label='Missing Budget Adapted', color='blue')
    ax1.set_title('Adapted Movies')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of missing values')
    ax1.set_xlim([1910, 2010])
    ax1.set_ylim([0, 70])
    ax1.grid(True)
    ax1.legend()

    ax2.plot(y3.index, y3, label='Missing Revenue', color='red')
    ax2.plot(y4.index, y4, label='Missing Budget', color='blue')
    ax2.set_title('Non-Adapted Movies')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number of missing values')
    ax2.set_xlim([1910, 2010])
    ax2.set_ylim([0, 2000])
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    plt.show()


def heatmap_missing_values(data):
    # Calculate missing values for movie adaptations
    missing_values_adaptations = data[data['movie_is_adaptation'] == True].isna().sum() / len \
        (data[data['movie_is_adaptation'] == True]) * 100

    # Calculate missing values for non-adaptations
    missing_values_non_adaptations = data[data['movie_is_adaptation'] == False].isna().sum() / len \
        (data[data['movie_is_adaptation'] == False]) * 100

    # Prepare data for heatmap
    missing_values_df = pd.DataFrame \
        ({'Adaptations': missing_values_adaptations, 'Non-Adaptations': missing_values_non_adaptations}).T

    plt.figure(figsize=(14, 6))
    sns.heatmap(missing_values_df, annot=False, fmt=".1f", cmap='Reds',
                cbar_kws={'label': 'Percentage of Missing Values (%)'})
    plt.title('Heatmap of Missing Values for Adaptations and Non-Adaptations')
    plt.xlabel('Columns')
    plt.ylabel('Movie Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.show()

def proportion_missing_values_all_dataset(data, features, adaptation=False):
    """
    Create a dictionary to store the count of missing values for each feature.
    If adaptation is True, use only the subset where 'movie_is_adaptation' is True.
    """
    if adaptation:
        data = data[data['movie_is_adaptation'] == True]
        print(f"Count of adapted movies: {len(data)}")
    
    missing_values_dict = {feature: data[feature].isna().sum() for feature in features}
    return missing_values_dict

def print_missing_values_summary(data, features, adaptation=False):
    """
    Prints the proportion of missing values for each feature in a formatted way.
    If adaptation is True, uses only the subset where 'movie_is_adaptation' is True.
    """
    if adaptation:
        subset_data = data[data['movie_is_adaptation'] == True]
        total_count = len(subset_data)
    else:
        subset_data = data
        total_count = len(data)
    
    missing_values_dict = proportion_missing_values_all_dataset(data, features, adaptation)
    print("\nMissing Values Summary:")
    for feature, missing_count in missing_values_dict.items():
        proportion = (missing_count / total_count) * 100 if total_count > 0 else 0
        print(f"{feature}: {missing_count} missing values ({proportion:.2f}%)")

    # Print the overall proportion of missing values for adaptations and non-adaptations
    adapted_missing_counts = {feature: data[data['movie_is_adaptation'] == True][feature].isna().sum() for feature in features}
    non_adapted_missing_counts = {feature: data[data['movie_is_adaptation'] == False][feature].isna().sum() for feature in features}

    print("\nProportion of missing values for adaptations:")
    for feature, count in adapted_missing_counts.items():
        proportion = (count / len(data[data['movie_is_adaptation'] == True])) * 100 if len(data[data['movie_is_adaptation'] == True]) > 0 else 0
        print(f"{feature}: {proportion:.2f}%")

    print("\nProportion of missing values for the entire dataset (including non-adaptations):")
    for feature, count in missing_values_dict.items():
        proportion = (count / len(data)) * 100 if len(data) > 0 else 0
        print(f"{feature}: {proportion:.2f}%")

def proportion_of_dataset(dataset_original, dataset_reduced) : 
    return (len(dataset_reduced)/len(dataset_original)*100)

def plot_histograms(df, columns):
    """
    Plot histograms for given columns in a DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    columns (list): List of column names for which to plot histograms.
    """
    for column in columns:
        if column in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column].dropna(), kde=True, bins=30, color='blue')

            

            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--', linewidth=0.5)
            plt.show()
        else:
            print(f"Column '{column}' not found in the DataFrame.")

