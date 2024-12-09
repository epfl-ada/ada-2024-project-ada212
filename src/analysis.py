import ast
from numpy import split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
