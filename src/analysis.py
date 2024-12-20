import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import gridspec
from tqdm import tqdm
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy import stats
import lifelines
from lifelines import KaplanMeierFitter
from statsmodels.stats.descriptivestats import describe
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def histogram_book_genre(data):
    genre_columns = ['book_children', 'book_historical', 'book_drama',
                     'book_anime', 'book_fantasy', 'book_science_fiction', 'book_horror',
                     'book_thriller', 'book_detective', 'book_satire', 'book_comedy']

    genre_counts = data[genre_columns].sum()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=genre_counts.index.str.replace('book_', '').str.replace('_', ' ').str.title(),
                y=genre_counts.values, hue = genre_counts.index.str.replace('book_', '').str.replace('_', ' ').str.title(), legend=False, palette='viridis')
    plt.title("Distribution of Book Genres")
    plt.ylabel("Count")
    plt.xlabel("Genre")
    plt.xticks(rotation=45)
    plt.show()


def plot_pie_chart_2(df, column_pos, column_neg):
    pos_counts = df[column_pos].sum()
    neg_counts = df[column_neg].sum()
    counts = [pos_counts, neg_counts]
    labels = [column_pos, column_neg]
    # Plot the donut chart
    plt.figure(figsize=(4, 4))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#8B0000', '#6a737b'], startangle=90,
            wedgeprops=dict(width=0.3))
    plt.title("Percentage of {} vs {}".format(column_pos, column_neg))
    plt.show()


def plot_pie_chart_1(df, column):
    counts = df[column].value_counts()
    labels = ["non_{}".format(column), column]
    # Plot the donut chart
    plt.figure(figsize=(4, 4))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#8B0000', '#6a737b'], startangle=90,
            wedgeprops=dict(width=0.3))
    plt.title("Percentage of {} vs {}".format(labels[0], labels[1]))
    plt.show()


def get_similarity(propensity_score1, propensity_score2):
    """Calculate similarity for instances with given propensity scores"""
    return 1 - np.abs(propensity_score1 - propensity_score2)


def matching_revenues(dataset):
    # select only entries with budget and revenue data available
    df = (dataset
          .query('movie_budget.notnull() & movie_revenue.notnull()')
          .reset_index(drop=True)
          )

    # select relevant columns for trial
    genre_cols = [col for col in df.columns if col.startswith('movie_genre_')]
    country_cols = [col for col in df.columns if col.startswith('movie_country_')]
    other_cols = ['movie_release', 'movie_runtime', 'movie_budget']
    relevant_cols = genre_cols + country_cols + other_cols

    X_df = df[relevant_cols]
    y_df = df['movie_is_adaptation']

    # logisitic regression to compute propensity scores
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression())
    ])
    pipe.fit(X_df, y_df)
    df['propensity_score'] = pipe.predict_proba(X_df)[:, 1]

    # match treatment and control groups
    treatment_df = df.query('movie_is_adaptation')
    control_df = df.query('~movie_is_adaptation')

    G = nx.Graph()
    for control_id, control_row in tqdm(control_df.iterrows(), total=len(control_df), desc='Building Graph'):
        for treatment_id, treatment_row in treatment_df.iterrows():
            # only match movies with same release year to make optimization faster
            if control_row['movie_release'] != treatment_row['movie_release']:
                continue
            similarity = get_similarity(control_row['propensity_score'],
                                        treatment_row['propensity_score'])
            if similarity < 0.975:
                continue

            G.add_weighted_edges_from([(control_id, treatment_id, similarity)])

    matching = nx.max_weight_matching(G)
    matched = [i[0] for i in list(matching)] + [i[1] for i in list(matching)]
    balanced_df = df.iloc[matched]
    return balanced_df


def plot_revenue_histogram(df):
    df = df.assign(
        label=lambda x: x.movie_is_adaptation.map(
            {True: "Adapted", False: "Original"}
        ).astype("category"),
    )

    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    sns.histplot(
        data=df,
        x="movie_revenue",
        hue="label",
        ax=ax1,
        bins=50,
        palette=["#8B0000", "#6a737b"],
        log_scale=True,
    )
    sns.boxplot(
        data=df,
        x="movie_revenue_log",
        y="label",
        ax=ax0,
        palette=["#8B0000", "#6a737b"],
        fliersize=0,
    )

    ax0.set(xlabel="", ylabel="")
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.set(xticks=[], yticks=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_visible(False)

    ax1.set(xlabel='Box Office Revenue [$US]', ylabel="Number of Movies")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_legend().set_title("")
    ax1.get_legend()._legend_box.align = "left"
    plt.show()


def matching_rating(dataset):
    df = (dataset
          .query('movie_budget.notnull() & movie_revenue.notnull()')
          .reset_index(drop=True)
          )
    df = (df
          .query('imdb_rating.notnull()')
          .reset_index(drop=True)
          )
    genre_cols = [col for col in df.columns if col.startswith('movie_genre_')]
    country_cols = [col for col in df.columns if col.startswith('movie_country_')]
    other_cols = ['movie_release', 'movie_runtime', 'movie_budget']
    relevant_cols = genre_cols + country_cols + other_cols

    X_df = df[relevant_cols]
    y_df = df['movie_is_adaptation']

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression())
    ])
    pipe.fit(X_df, y_df)
    df['propensity_score'] = pipe.predict_proba(X_df)[:, 1]

    # match treatment and control groups
    treatment_df = df.query('movie_is_adaptation').reset_index(drop=False)
    control_df = df.query('~movie_is_adaptation').reset_index(drop=False)

    matching_df = (treatment_df
                   .merge(
        control_df,
        on=['movie_release'] + genre_cols[:15] + country_cols[:5],
        suffixes=['_treatment', '_control']
    )
                   .assign(
        similarity=lambda x: x.apply(lambda y: get_similarity(y.propensity_score_treatment, y.propensity_score_control),
                                     axis=1)
    )
                   .query('similarity > 0.975')
                   )

    G = nx.Graph()
    for _, row in tqdm(matching_df.iterrows(), total=len(matching_df), desc='Building Graph'):
        similarity = row.similarity
        index_control = row['index_control']
        index_treatment = row['index_treatment']
        G.add_weighted_edges_from([(index_control, index_treatment, similarity)])

    matching = nx.max_weight_matching(G)
    matched = [i[0] for i in list(matching)] + [i[1] for i in list(matching)]
    balanced_df = df.iloc[matched]
    return balanced_df


def plot_ratings_histogram(df):
    df = df.assign(
        label=lambda x: x.movie_is_adaptation.map(
            {True: "Adapted", False: "Original"}
        ).astype("category"),
    )

    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    sns.histplot(
        data=df,
        x="imdb_rating",
        hue="label",
        ax=ax1,
        bins=50,
        palette=["#8B0000", "#6a737b"]
    )
    sns.boxplot(
        data=df,
        x="imdb_rating",
        y="label",
        ax=ax0,
        palette=["#8B0000", "#6a737b"],
        fliersize=0,
    )

    ax0.set(xlabel="", ylabel="")
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.set(xticks=[], yticks=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_visible(False)

    ax1.set(xlabel='IMDB Rating', ylabel="Number of Movies")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_legend().set_title("")
    ax1.get_legend()._legend_box.align = "left"
    plt.show()


def make_book_histplot(df, col: str, x_label: str, log=False, ylog=False):
    if log:
        df = df.copy()  
        df.loc[:, "log"] = np.log(df[col])  
        col = "log"
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    if ylog == True:
        ax1.set_yscale('log')
    if log:
        sns.histplot(data=df, x=col, hue='label', ax=ax1, bins=50, palette=['#6a737b', '#8B0000'], log_scale=True,
                     multiple="stack")
        sns.boxplot(data=df, x="log", y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0, hue = 'label', legend=False)
    else:
        sns.histplot(data=df, x=col, hue='label', ax=ax1, bins=50, palette=['#6a737b', '#8B0000'], multiple="stack")
        sns.boxplot(data=df, x=col, y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0, hue = 'label', legend=False)

    ax0.set(xlabel="", ylabel="")
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.set(xticks=[], yticks=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_visible(False)

    ax1.set(xlabel=x_label, ylabel="Number of Books")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_legend().set_title("")
    ax1.get_legend()._legend_box.align = "left"

    plt.show()

def plot_award_proportion_barplot(df, x_col, y_col, x_label, x_lim, colors=['#6a737b', '#8B0000']):
    """
    Plots a barplot showing the proportion of award-winning books.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    x_col (str): The column name for the x-axis values (proportions).
    y_col (str): The column name for the y-axis categories (labels).
    x_label (str): The label for the x-axis.
    x_lim (tuple): The x-axis limit as a tuple (min, max).
    colors (list): The list of colors for the bars (default: ['#6a737b', '#8B0000']).
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create the barplot
    sns.barplot(data=df, y=y_col, x=x_col, palette=colors, edgecolor='.0', ax=ax, hue = y_col, legend=False)
    
    # Set labels and aesthetics
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.xlim(x_lim)
    
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Show the plot
    plt.show()

def plot_survival_curve(data, time_col, title, xlabel, ylabel, xlim=None):
    """
    Plots a Kaplan-Meier survival curve for the given dataset.

    Parameters:
    - data (pd.DataFrame): The dataset containing the time-to-event data.
    - time_col (str): The name of the column representing the time-to-event variable.
    - title (str): The title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - xlim (tuple, optional): Limit for the x-axis as (min, max).
    """
    # Initialize Kaplan-Meier fitter
    kmf = KaplanMeierFitter()

    # Fit the Kaplan-Meier estimator
    kmf.fit(data[time_col], event_observed=[1] * len(data))

    # Plot the survival curve
    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function()
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if xlim:
        plt.xlim(xlim)
    plt.show()


def plot_grouped_survival_curves(data, time_col, group_col, title, xlabel, ylabel, xlim=None):
    """
    Plots Kaplan-Meier survival curves for different groups within the dataset.

    Parameters:
    - data (pd.DataFrame): The dataset containing the time-to-event and grouping data.
    - time_col (str): The name of the column representing the time-to-event variable.
    - group_col (str): The name of the column to group by.
    - title (str): The title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - xlim (tuple, optional): Limit for the x-axis as (min, max).
    """
    # Initialize Kaplan-Meier fitter
    kmf = KaplanMeierFitter()

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Iterate through groups and plot survival curves
    for group, subset in data.groupby(group_col):
        kmf.fit(subset[time_col], event_observed=[1] * len(subset), label=f"{group_col}: {group}")
        kmf.plot_survival_function()

    # Add plot title and labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if xlim:
        plt.xlim(xlim)
    plt.legend(title=group_col, fontsize=10)
    plt.show()



def analyze_adaptation_timeline(df):
    """
    Enhanced analysis of book-to-movie adaptation timelines with advanced statistics
    and visualizations.
    """
    # Filter for actual adaptations and clean the data
    adaptations = df[df['movie_is_adaptation'] == True].copy()
    adaptations = adaptations.dropna(subset=['time_gap'])
    
    # Create decade groups for movies
    adaptations['movie_decade'] = (adaptations['movie_release'] // 10) * 10
    
    # Process genres
    adaptations['primary_genre'] = adaptations['movie_genres'].apply(
        lambda x: eval(x)[0] if isinstance(x, str) else x[0] if isinstance(x, list) else 'Unknown'
    )
    
    # Create time period categories
    adaptations['adaptation_speed'] = pd.cut(
        adaptations['time_gap'],
        bins=[-np.inf, 5, 15, 30, np.inf],
        labels=['Very Fast (≤5)', 'Fast (6-15)', 'Moderate (16-30)', 'Slow (>30)']
    )
    
    # Comprehensive statistical analysis
    stats_analysis = {
        'basic_stats': describe(adaptations['time_gap']),
        'distribution_test': stats.normaltest(adaptations['time_gap']),
        'decade_trends': stats.spearmanr(
            adaptations['movie_decade'],
            adaptations['time_gap']
        )
    }
    
    # Genre analysis with statistical tests
    top_genres = adaptations['primary_genre'].value_counts().head(5).index
    genre_stats = {}
    for genre in top_genres:
        genre_data = adaptations[adaptations['primary_genre'] == genre]['time_gap']
        other_data = adaptations[adaptations['primary_genre'] != genre]['time_gap']
        genre_stats[genre] = {
            'mean': genre_data.mean(),
            'median': genre_data.median(),
            'count': len(genre_data),
            'mann_whitney': mannwhitneyu(genre_data, other_data)
        }
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 12))
    
    # Enhanced Timeline Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data=adaptations, x='time_gap', bins=50, kde=True)
    plt.axvline(adaptations['time_gap'].median(), color='r', linestyle='--', 
                label=f'Median: {adaptations["time_gap"].median():.1f} years')
    plt.axvline(adaptations['time_gap'].mean(), color='g', linestyle='--', 
                label=f'Mean: {adaptations["time_gap"].mean():.1f} years')
    plt.title('Distribution of Adaptation Timelines')
    plt.xlabel('Years between Book and Movie')
    plt.legend()
    
    # Decade Analysis with error bars
    plt.subplot(2, 2, 2)
    decade_stats = adaptations.groupby('movie_decade').agg({
        'time_gap': ['mean', 'count', 'std']
    }).reset_index()
    decade_stats.columns = ['decade', 'mean', 'count', 'std']
    decade_stats['se'] = decade_stats['std'] / np.sqrt(decade_stats['count'])
    
    # Create bar plot
    bars = plt.bar(range(len(decade_stats)), decade_stats['mean'])
    plt.errorbar(range(len(decade_stats)), decade_stats['mean'], 
                 yerr=decade_stats['se'], fmt='none', color='black', capsize=5)
    
    # Customize x-axis
    plt.xticks(range(len(decade_stats)), decade_stats['decade'], rotation=45)
    plt.title('Average Adaptation Timeline by Decade\nwith Standard Error Bars')
    plt.xlabel('Movie Release Decade')
    plt.ylabel('Years to Adaptation')
    
    # Add sample sizes to bars
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'n={int(decade_stats.iloc[idx]["count"])}',
                ha='center', va='bottom')
    
    # Genre Analysis
    plt.subplot(2, 2, 3)
    genre_data = adaptations.groupby('primary_genre')['time_gap'].agg(['mean', 'count'])
    genre_data = genre_data.sort_values('count', ascending=False).head(10)
    
    sns.barplot(x=genre_data.index, y='mean', data=genre_data.reset_index())
    plt.title('Average Adaptation Timeline by Genre\n(Top 10 Most Common Genres)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Years to Adaptation')
    
    # Adaptation Speed Distribution
    plt.subplot(2, 2, 4)
    speed_dist = adaptations['adaptation_speed'].value_counts()
    plt.pie(speed_dist, labels=speed_dist.index, autopct='%1.1f%%')
    plt.title('Distribution of Adaptation Speeds')
    
    # Prepare summary statistics
    summary_stats = {
        'timeline_stats': stats_analysis['basic_stats'],
        'genre_analysis': genre_stats,
        'decade_correlation': stats_analysis['decade_trends'],
        'speed_distribution': speed_dist
    }
    
    return {
        'summary_stats': summary_stats,
        'adaptations_data': adaptations,
        'decade_trends': decade_stats,
        'genre_trends': genre_data
    }



def linear_regression_analysis(data, categorical_features, numeric_features, target_col):
    """
    Perform linear regression analysis using Statsmodels and report significant features.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing features and target variable.
    - categorical_features (list): List of binary categorical feature columns.
    - numeric_features (list): List of continuous numeric feature columns.
    - target_col (str): Name of the target variable column.
    
    Returns:
    - model_summary (str): Statsmodels regression summary.
    - significant_features (dict): Features with p-values < 0.05.
    """
    # Extract features and target
    X_cat = data[categorical_features]
    X_num = data[numeric_features]
    y = data[target_col]

    # Scale numeric features
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns, index=X_num.index)

    # Combine categorical and scaled numeric features
    X = pd.concat([X_cat, X_num_scaled], axis=1)

    # Drop rows with missing values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Add constant for statsmodels
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Extract significant features
    significant_features = {
        feature: {
            'coefficient': model.params[feature],
            'p_value': model.pvalues[feature]
        }
        for feature in model.pvalues.index if model.pvalues[feature] < 0.05
    }

    # Return the model summary and significant features
    return model.summary(), significant_features



def regression_analysis(data, features, target, test_size=0.2, random_state=42):
    """
    Performs regression analysis using Lasso, Ridge, and Random Forest models.

    Parameters:
    - data (pd.DataFrame): The dataset containing features and target.
    - features (list): List of feature column names.
    - target (str): The name of the target column.
    - test_size (float): Proportion of data to use as the test set.
    - random_state (int): Seed for reproducibility.

    Returns:
    - results (dict): RMSE and R2 scores for each model.
    - weights_summary (dict): Coefficients or feature importances for each model.
    """
    # Prepare features (X) and target (y)
    X = data[features]
    y = data[target]

    # Drop rows with missing values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize models
    lasso = Lasso(alpha=0.01, random_state=random_state)
    ridge = Ridge(alpha=1.0)
    random_forest = RandomForestRegressor(n_estimators=100, random_state=random_state)

    models = {'Lasso': lasso, 'Ridge': ridge, 'Random Forest': random_forest}
    results = {}

    # Train and evaluate models
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R2': r2}

    # Extract feature weights
    lasso_weights = dict(zip(X_train.columns, lasso.coef_))
    ridge_weights = dict(zip(X_train.columns, ridge.coef_))
    rf_importances = dict(zip(X_train.columns, random_forest.feature_importances_))

    weights_summary = {
        'Lasso Coefficients': lasso_weights,
        'Ridge Coefficients': ridge_weights,
        'Random Forest Importances': rf_importances
    }

    return results, weights_summary



def visualize_timegap_genres(data, genres, time_col='time_gap', title_prefix='Time Gap for'):
    """
    Visualizes the distribution of time gaps for each genre using histograms and box plots.

    Parameters:
    - data (pd.DataFrame): The dataset containing genres and time gap information.
    - genres (list): List of genre column names (binary features).
    - time_col (str): The column name representing the time gap (default: 'time_gap').
    - title_prefix (str): Prefix for the title of each plot (default: 'Time Gap for').
    """
    # Subplot grid for histograms
    plt.figure(figsize=(15, 10))
    
    for i, genre in enumerate(genres, 1):
        plt.subplot(5, 3, i)  # Create subplot grid (5 rows, 3 columns)
        
        # Filter data for the current genre and plot histogram
        sns.histplot(data=data[data[genre] == True], x=time_col, bins=30)
        
        # Add title and labels
        plt.title(f'Distribution of {title_prefix} {genre.replace("book_", "").title()}')
        plt.xlabel('Time Gap (years)')
        plt.ylabel('Count')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
    # Box plot to compare distributions across genres
    plt.figure(figsize=(12, 6))
    genre_data = []
    genre_labels = []

    for genre in genres:
        genre_data.append(data[data[genre] == True][time_col])  # Append data for box plot
        genre_labels.append(genre.replace("book_", "").title())  # Create labels for box plot

    # Create box plot
    plt.boxplot(genre_data, labels=genre_labels)
    plt.title('Time Gap Distribution Comparison Across Genres')
    plt.ylabel('Time Gap (years)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()



def print_enhanced_insights(results):
    """
    Prints detailed statistical insights from the analysis.
    """
    print("📊 ADAPTATION TIMELINE ANALYSIS\n")
    
    print("1. BASIC STATISTICS")
    print(results['summary_stats']['timeline_stats'])
    
    print("\n2. GENRE INSIGHTS")
    for genre, stats in results['summary_stats']['genre_analysis'].items():
        print(f"\n{genre}:")
        print(f"  Mean: {stats['mean']:.1f} years")
        print(f"  Median: {stats['median']:.1f} years")
        print(f"  Sample size: {stats['count']}")
        print(f"  Statistical significance: p={stats['mann_whitney'].pvalue:.4f}")
    
    print("\n3. TEMPORAL TRENDS")
    corr = results['summary_stats']['decade_correlation']
    print(f"Decade-Timeline Correlation: rho={corr[0]:.3f}, p={corr[1]:.4f}")
    
    print("\n4. ADAPTATION SPEED DISTRIBUTION")
    speed_dist = results['summary_stats']['speed_distribution']
    for category, percentage in (speed_dist / len(speed_dist) * 100).items():
        print(f"{category}: {percentage:.1f}%")

    plt.tight_layout()

def matching_books(dataset, fiction_cols, genre_cols):
    # match treatment and control groups
    df = (dataset
          .assign(
        book_release_bin=dataset.book_release // 5 * 5
    )
          .query('book_rating.notna() & book_ratings_count > 10')
          )

    treatment_df = df.query('book_adapted == 1').reset_index(drop=False)
    control_df = df.query('book_adapted == 0').reset_index(drop=False)

    # exact matching on book release year, fiction and genre
    matching_df = (treatment_df
    .merge(
        control_df,
        on=['book_release_bin'] + fiction_cols + genre_cols,
        suffixes=['_treatment', '_control'],
        how='inner'
    )
    )

    # build graph for matching
    G = nx.Graph()
    for _, row in tqdm(matching_df.iterrows(), total=len(matching_df), desc='Building Graph'):
        index_control = row['index_control']
        index_treatment = row['index_treatment']
        G.add_edges_from([(index_control, index_treatment)])

    matching = nx.maximal_matching(G)
    matched = [i[0] for i in list(matching)] + [i[1] for i in list(matching)]
    balanced = df.loc[matched]
    return balanced


def make_book_histplot_2(df, col: str, x_label: str, log=False, ylog=False):
    if log:
        df["log"] = np.log(df[col])
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    if ylog == True:
        ax1.set_yscale('log')
    if log:
        sns.histplot(data=df, x=col, hue='label', ax=ax1, bins=50, palette=['#6a737b', '#8B0000'], log_scale=True)
        sns.boxplot(data=df, x="log", y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0, hue = 'label', legend=False)
    else:
        sns.histplot(data=df, x=col, hue='label', ax=ax1, bins=50, palette=['#6a737b', '#8B0000'])
        sns.boxplot(data=df, x=col, y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0, hue = 'label', legend=False)

    ax0.set(xlabel="", ylabel="")
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.set(xticks=[], yticks=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_visible(False)

    ax1.set(xlabel=x_label, ylabel="Number of Books")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_legend().set_title("")
    ax1.get_legend()._legend_box.align = "left"

    plt.show()


def clean_country_list(countries):
    if pd.isna(countries):
        return []
    return [country.strip() for country in countries.strip("[]").replace("'", "").split(",")]

def create_country_pairs(countries):
    if len(countries) > 1:
        return [(countries[i], countries[j]) for i in range(len(countries)) for j in range(i + 1, len(countries))]
    return []

def extract_main_genre(genre_list):
    if pd.notnull(genre_list):  # Check if the genre_list is not null
        genres = genre_list.split(',')
        if len(genres) > 1:  # Check if the list contains more than one genre
            return f"{genres[0].strip()}]"  # Format the first genre with square brackets
        else:
            return genre_list.strip()  # Keep the single genre as is
    return None