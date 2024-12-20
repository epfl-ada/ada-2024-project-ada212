# From Page to Screen – The Journey and Success of Book Adaptations

Team : Ghalia, Rania, Valentin, Omar and Leïla 

# Abstract
This project explores the impact of book adaptations on a movie’s success, examining which factors contribute to their financial and critical outcomes. By leveraging the CMU Movie Summary Corpus, enriched with metadata from IMDb and Wikidata, the research investigates whether being adapted from a book, including genre and source type (single novel or series), correlates with box office revenue and audience reception. The analysis aims to uncover patterns in adaptation strategies, comparing profitability across different genres and production scales. This study not only highlights the key elements behind the successful transition from page to screen but also offers insights into broader audience preferences and the storytelling elements that resonate with viewers.

# Data Story
Our Data Story is presented in the following link: https://raniahtr.github.io/ADA212_page_to_screens/

# Research Questions
## Storytelling Framework
The analysis is framed around the storytelling theme. This narrative guides us through the following research questions, building a cohesive story that contrasts book adaptations and original movies.

1. **The Origins: What Books Capture the Attention of Filmmakers?**
   - What traits in books make them attractive for adaptation?

2. **The Path to Adaptation: What Factors Influence Adaptation Time?**
   - How long does it take for a popular book to become a movie, and what influences this timeline?

3. **The Grand Premiere: Measuring the Success of Adaptations**
   - How book adaptations succeed compared to original movies?

4. **The Global Reception: Do Adaptations Translate Well Across Borders?**
   - How do movie adaptations perform internationally, and are adaptations from certain countries more successful?

5. **The Budget and Investment Dilemma: Does Spending More Pay Off?**
   - Is there a direct correlation between the budget of an adaptation and its financial and critical success?

6. **Plot Twist: Unexpected Success Stories**
   - Which book adaptations outperformed expectations despite low initial ratings or budgets?

# Data
- **Base Dataset**: The primary dataset, the CMU Movie Summary Corpus, contains information on approximately 80,000 films, including details like release dates, duration, genre, and cast. For this study, we augmented this dataset with data from several additional sources.
- **Additional Datasets**:
   - **IMDb Ratings:** To measure audience reception, we enhanced our dataset with IMDb scores.
   - **TMDB:** Many entries in the CMU dataset lacked revenue information. We filled these gaps using data from The Movie Database (TMDB).
   - **Wikidata:** We fetched some more data about movies' revenue and budget from Wikidata. Moreover, to identify films that are based on books, we queried the Wikidata Graph Database with SPARQL. We also obtained metadata about these books, such as publication dates, page count, genre, and author, enabling comparisons between the books' characteristics and those of their corresponding films.
   - **Goodreads:** To further enrich our book data, we incorporated information scraped from Goodreads, including user ratings and page counts for each book.
   - **Consumer Price Index (CPI):** Since our revenue and budget data spans multiple decades, we used the U.S. Consumer Price Index to normalize financial data, making revenue and budget figures comparable over time.
- **Data extraction and first preprocessing:**: In the notebook data_extraction.ipynb, we loaded, preprocessed, and combined these datasets.
   - **How to merge ?** 
      - CMU-Wikidata Link: Each movie in the CMU dataset has a unique Wikipedia ID, which does not match the corresponding Wikidata IDs. We used the wikimapper package to align Wikidata and Wikipedia IDs accurately.
      - Wikidata-IMDb-TMDB Link: Since Wikidata includes IMDb IDs for most movies, we leveraged these identifiers to link records across the IMDb and TMDB datasets.
      - Wikidata-Goodreads Link: Merging book data from Wikidata with Goodreads posed difficulties due to inconsistent Goodreads IDs in Wikidata. Instead, we matched the records using a combination of (book_author, book_title), assuming this pair would uniquely identify most books.
   - **First Preprocessing:**
      - We standardized title and author formats across datasets.
      - We fetched the US Consumer Price Index (CPI) and computed the inflation adjustment factor for each year so we can later on normalize financial data.
- **Final Dataset:**  data_extraction.ipynb notebook creates 3 .csv files:
   - book_adapation.csv which contains metadata for movies and metadata for books for which movies were identified as adaptations from Wikidata.
   - inflation_adjustment.csv which contains the inflation adjustment rate for multiple years, it will be used later in the results.ipynb to normalize budgets and revenues.
   - book_adapation_extanded.csv which contains some late fetched revenue and budget infomations from Wikidata.

# Methods 
In order to find answers to these questions, we will use the following data analysis pipeline.

## Table of Contents
1. [Creating the Dataset and Cleaning the Data](#part-0-creating-the-dataset-and-cleaning-the-data)
2. [Identifying Traits That Make Books Attractive for Adaptation](#part-1-identifying-traits-that-make-books-attractive-for-adaptation)
3. [Mapping the Timeline of Adaptations](#part-2-mapping-the-timeline-of-adaptations)
4. [Measuring the Success of Book-to-Movie Transformations](#part-3-measuring-the-success-of-book-to-movie-transformations)
5. [Analyzing the International Success of Adaptations](#part-4-analyzing-the-international-success-of-adaptations)
6. [Examining the Role of Budget in Adaptations](#part-5-examining-the-role-of-budget-in-adaptations)
7. [Unveiling Unexpected Success Stories](#part-6-unveiling-unexpected-success-stories)

---

## Part 0: Creating the Dataset and Cleaning the Data
### Objective
Prepare a reliable dataset for analysis by addressing issues like missing values and inconsistencies.

### Approach
- Remove duplicates and handle missing values effectively.
- Standardize categorical variables (e.g., genres, languages) and numerical data (e.g., revenue, budget).
- Apply normalization or transformations (e.g., log transformations) to address skewed distributions.

---

## Part 1: Identifying Traits That Make Books Attractive for Adaptation
### Research Question
What traits in books make them attractive for adaptation?

### Analysis
- Explore common characteristics of adapted books (e.g., genre, length, ratings).
- Use correlation analysis to identify book features (e.g., ratings, page count) associated with adaptation likelihood.
- Compare adapted and non-adapted books using propensity score matching to highlight distinct traits.

---

## Part 2: Mapping the Timeline of Adaptations
### Research Question
How long does it take for a popular book to become a movie, and what influences this timeline?

### Analysis
- Visualize adaptation timelines with Kaplan-Meier estimators, showing the probability of adaptation over time.
- Apply Lasso and Ridge Regression to pinpoint factors influencing adaptation speed.

---

## Part 3: Measuring the Success of Book-to-Movie Transformations
### Research Question
How book adaptations succeed compared to original movies?

### Analysis
- Model relationships between book traits (e.g., genre, ratings) and movie success metrics (e.g., box office revenue, IMDb ratings).
- Conduct propensity score matching to compare success rates of book adaptations and original movies.

---

## Part 4: Analyzing the International Success of Adaptations
### Research Question
How do movie adaptations perform internationally, and are adaptations from certain countries more successful?

### Analysis
- Use maps to visualize the performance of adaptations by country.
- Leverage audience review data to assess reception across different regions.
- Investigate trends between multinational productions and single-country productions.

---

## Part 5: Examining the Role of Budget in Adaptations
### Research Question
Is there a direct correlation between the budget of an adaptation and its financial and critical success?

### Analysis
- Use scatter plots and regression models to analyze relationships between budget size, revenue, and success metrics.
- Calculate ROI using the formula:
  ROI = $\frac{\text{Revenue} - \text{Budget}}{\text{Budget}}$

  and visualize the distribution with kernel density plots.
- Identify budget ranges associated with optimal returns on investment.

---

## Part 6: Unveiling Unexpected Success Stories
### Research Question
Which book adaptations outperformed expectations despite low initial ratings or budgets?

### Analysis
- Identify outliers where adaptations exceeded expectations despite low initial ratings or budgets.
- Examine common attributes among these adaptations to uncover trends in unexpected success.

# 🗓️ Timeline

## ⏳ Until 15/11/2024 (Deadline Milestone P2)
- **Creating Dataset**: *Rania and Ghalia*
- **Data Handling and Preprocessing**: *Rania and Ghalia*
- **Initial Exploratory Data Analysis**: *Omar, Leïla, and Valentin*

## 📅 15/11 to 30/11
- **Homework 2**: *All 5 members*
- **Parts 1 - 3 Completion**: *Leïla, Rania, and Valentin*

## 📆 30/11 to 8/12
- **Parts 4 - 6 Completion**: *Ghalia and Omar*
- **Start - Visualization Planning**: *Valentin*
- **Begin Implementation**: *Leïla and Rania*

## 🗓️ 8/12 to 22/12 (Deadline Milestone P3)
- **Repository Cleaning**: *All team members*
- **Finalize/Refine Host Page for Data Visualization**: *All team members*

# 🗓️ Organization within the team
- **Ghalia**: Web Scraping/ Data Cleaning / Research Question 1/ Research Question 3/ Research Question 4/ GitHub Organization
- **Rania**: Web Scraping/ Data Cleaning/ Research Question 2/ Research Question 4/ Data Story/ Website Layout
- **Leila**: Data Preprocessing/ Preliminary Analyses/ Research Question 4/ Research Question 6/ GitHub Organization
- **Omar**: Web Scrapping/ Preliminary Analyses/ Research Question 1/ Research Question 5/ Research Question 6
- **Valentin**: Preliminary Analyses/ Research Question 6

```
ada-2024-project-ada212/
│
├── data/                                  # Directory for all data files used in the project
│   ├── book_adaptation.csv                # Dataset containing movies and their potential book adaptation information
│   ├── book_adaptation_expanded.csv       # Expanded dataset with additional revenues and budgets for movies
│   ├── cleaned_dataset.csv                # Cleaned version of the dataset after preprocessing
│   └── inflation_adjustment.csv           # Data file for adjusting monetary values for inflation
│
├── data_utils/                            # Directory for one file used by data_extraction
│   └── flat-ui__data-ThuNov142024.csv     # U.S. Consumer Price Index 
│
├── src/                                   # Source code directory for project scripts and notebooks
│   ├── utils.py                           # Utility functions used during the data cleaning and preliminary analysis phases
│   └── analysis.py                        # Utility functions used across the analysis phase
│
├── README.md                              # Project documentation
├── framework.ipynb                        # Notebook outlining the framework for analysis
├── results.ipynb                          # Notebook summarizing and presenting the results
├── data_extraction.ipynb                  # Jupyter notebook for data extraction and initial exploration
├── requirements.txt                       # List of dependencies for reproducing the project environment
```
