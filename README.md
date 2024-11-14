# From Page to Screen – The Journey and Success of Book Adaptations

Team : Ghalia, Rania, Valentin, Omar and Leïla 

# Abstract
This project explores the impact of book adaptations on a movie’s success, examining which factors contribute to their financial and critical outcomes. By leveraging the CMU Movie Summary Corpus, enriched with metadata from IMDb and Wikidata, the research investigates whether being adapted from a book, including genre and source type (single novel or series), correlates with box office revenue and audience reception. The analysis aims to uncover patterns in adaptation strategies, comparing profitability across different genres and production scales. This study not only highlights the key elements behind the successful transition from page to screen but also offers insights into broader audience preferences and the storytelling elements that resonate with viewers.

## Storytelling Framework
The analysis is framed around the storytelling theme. This narrative guides us through the following research questions, building a cohesive story that contrasts book adaptations and original movies.

# Research Questions
1. **The Origins: What Books Capture the Attention of Filmmakers?**
   - What traits in books make them attractive for adaptation?

2. **The Path to Adaptation: What Factors Influence Adaptation Time?**
   - How long does it take for a popular book to become a movie, and what influences this timeline?
   - Are there seasonal patterns in the release of book adaptations?

3. **The Grand Premiere: Measuring the Success of Adaptations**
   - What elements predict the success of a movie adaptation?
   - How has the popularity and financial success of book adaptations evolved over time?

4. **The Global Reception: Do Adaptations Translate Well Across Borders?**
   - How do movie adaptations perform internationally, and are adaptations from certain countries more successful?
   - Are adaptations from certain countries more successful, and do they align with the nationality of the book's author?

5. **The Budget and Investment Dilemma: Does Spending More Pay Off?**
   - Is there a direct correlation between the budget of an adaptation and its financial and critical success?

6. **Plot Twist: Unexpected Success Stories**
   - Which book adaptations outperformed expectations despite low initial ratings or budgets?



# Methods 
In order to find answers to these questions, we will use the following data analysis pipeline.

## Part 0 : Creating dataset and cleaning the data
- **Approach**:
  - Remove duplicates and handle missing values appropriately.
  - Clean categorical variables (e.g., genres, languages) and standardize numerical data (e.g., revenue, budget) for analysis.
  - Normalize or transform data as needed (e.g., log transformations for skewed distributions).
    
## Part 1 : Looking for similarities between adapted books
- Step 1 : Use correlation analysis to identify book features (e.g., ratings, page count) that correlate with being chosen for adaptation
- Step 2 : Use a Random Forest Regressor to determine which features of the book (e.g., page count, genre, ratings) are most important in predicting movie success.
- Step 3 : Implement a logistic regression to predict whether a book will result in a successful movie adaptation (using a threshold, such as revenue above a median value or IMDb rating above a certain score).

## Part 2 : 
- Step 1 : **Kaplan-Meier Estimators**: Visualize the “survival” rate of books over time, showing the likelihood of adaptation within different time frames.
- Step 2: **Seasonal Analysis**: Analyze the release patterns of adaptations to identify if certain seasons align with higher success rates using time series plots.

## Part 3 : Results of book-to-movie transformation
- Step 1 : Model the relationship between book characteristics (e.g., genre, ratings) and movie success metrics (e.g., box office revenue, IMDb rating). For instance, using correlational analysis between Goodreads ratings and movie performance metrics.
- Step 2 : Compare movie adaptations with original movies to see which type tends to perform better.

## Part 4 : International renown of adaptations 
- Step 1 : Create maps showing the performance of adaptations by country.
- Step 2 : Use audience review data to assess how adaptations are received in different parts of the world.

## Part 5 : Focusing on budget allocated to adaptations
- Step 1 : Use scatter plots and regression models to show if a larger budget leads to higher revenue and better ratings.
- Step 2 : Calculate ROI and visualize the distribution to identify budget “sweet spots” where adaptations achieve the best returns.

## Part 6 : Plot twist: unexpected story successes



# Timeline 

## Until 15/11/2024 (Deadline Milestone P2) :   
- Creating dataset (Rania and Ghalia)
- Data-Handling and Preprocessing (Rania and Ghalia)
- Initial Exploratory Data Analysis (Omar, Leïla and Valentin)

## 15/11 to 30/11 :
- Homework 2 (all 5 members)
- Parts 1 - 3 (Leïla, Rania and Valentin)

## 30/11 to 8/12 :
- Parts 4 - 5 (Ghalia and Omar)
- Start Part 6 : How would we like to visualize our findings ? (Valentin)
- Start implementation(Leïla and Rania)

## 8/12 to 22/12 (Deadline Milestone P3) : 
- Cleaning repository 
- Finalize/Refine our host page for visualizing our data story
