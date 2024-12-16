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


def histogram_book_genre(data):
    genre_columns = ['book_children', 'book_historical', 'book_drama',
                     'book_anime', 'book_fantasy', 'book_science_fiction', 'book_horror',
                     'book_thriller', 'book_detective', 'book_satire', 'book_comedy']

    genre_counts = data[genre_columns].sum()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=genre_counts.index.str.replace('book_', '').str.replace('_', ' ').str.title(),
                y=genre_counts.values, palette='viridis')
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
        df["log"] = np.log(df[col])
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    if ylog == True:
        ax1.set_yscale('log')
    if log:
        sns.histplot(data=df, x=col, hue='label', ax=ax1, bins=50, palette=['#6a737b', '#8B0000'], log_scale=True,
                     multiple="stack")
        sns.boxplot(data=df, x="log", y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0)
    else:
        sns.histplot(data=df, x=col, hue='label', ax=ax1, bins=50, palette=['#6a737b', '#8B0000'], multiple="stack")
        sns.boxplot(data=df, x=col, y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0)

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
        sns.boxplot(data=df, x="log", y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0)
    else:
        sns.histplot(data=df, x=col, hue='label', ax=ax1, bins=50, palette=['#6a737b', '#8B0000'])
        sns.boxplot(data=df, x=col, y='label', ax=ax0, palette=['#6a737b', '#8B0000'], fliersize=0)

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
