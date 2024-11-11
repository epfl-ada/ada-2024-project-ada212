# From Page to Screen – The Journey and Success of Book Adaptations

Team : Ghalia, Rania, Valentin, Omar and Leïla 

# Abstract


# Research Questions
In this analysis, we would like to answer the following questions:

- What traits in books make them attractive for adaptation?

- How long does it take for a popular book to become a movie, and what influences this timeline?

- What elements predict the success of a movie adaptation?

- How do movie adaptations perform internationally, and are adaptations from certain countries more successful?

- Is there a direct correlation between the budget of an adaptation and its financial and critical success?


# Methods 

In order to find answers to these questions, we will use the following data analysis pipeline.

## Part 0 : Creating dataset and cleaning the data

## Part 1 : Looking for similarities between adapted books
- Step 1 : Use correlation analysis to identify book features (e.g., ratings, page count) that correlate with being chosen for adaptation
- Step 2 : Use a Random Forest Regressor to determine which features of the book (e.g., page count, genre, ratings) are most important in predicting movie success.
- Step 3 : Implement a logistic regression to predict whether a book will result in a successful movie adaptation (using a threshold, such as revenue above a median value or IMDb rating above a certain score).

## Part 2 : 
- Step 1 : Use Kaplan-Meier estimators to visualize the “survival” rate of books over time i.e., how likely they are to be adapted within certain time frames.

## Part 3 : Results of book-to-movie transformation
- Step 1 : Model the relationship between book characteristics (e.g., genre, ratings) and movie success metrics (e.g., box office revenue, IMDb rating). For instance, using correlational analysis between Goodreads ratings and movie performance metrics.
- Step 2 : Compare movie adaptations with original movies to see which type tends to perform better.

## Part 4 : International renown of adaptations 
- Step 1 : Create maps showing the performance of adaptations by country.
- Step 2 : Use audience review data to assess how adaptations are received in different parts of the world.

## Part 5 : Focusing on budget allocated to adaptations
- Step 1 : Use scatter plots and regression models to show if a larger budget leads to higher revenue and better ratings.
- Step 2 : Calculate ROI and visualize the distribution to identify budget “sweet spots” where adaptations achieve the best returns.

## Part 6 : Create Datastory

# Timeline 

## Until 15/11/2024 :   
- Creating dataset 
- Data-Handling and Preprocessing 
- Initial Exploratory Data Analysis 

## 15/11 to 30/11 
- Homework 2
- Parts 1 - 3

## 30/11 to 8/12
- Parts 4 - 5
- Start Part 6 : How would we like to visualize our findings ? 
- Start implementation

## 8/12 to 22/12 
- Cleaning repository 
- Finalize/Refine our host page for visualizing our data story
