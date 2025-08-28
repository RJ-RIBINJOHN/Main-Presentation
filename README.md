## Online Gaming Behavior Analysis and Player Segmentation

## Project Description
This project analyzes online gaming behavior data to understand player engagement, predict churn risk, and segment players based on their behavior. The analysis includes data preprocessing, exploratory data analysis, building and evaluating churn prediction models, and performing player segmentation using K-Means clustering.

## How to Run the Code
1.  **Open the Notebook**: Open the provided Colab notebook in Google Colaboratory.
2.  **Run Cells**: Execute each code cell sequentially. The notebook is structured in a step-by-step manner, covering data loading, preprocessing, model building, evaluation, and clustering.
3.  **Data File**: Ensure the `online_gaming_behavior_dataset.csv` file is available in the Colab environment. The notebook assumes this file is present in the `/content/` directory.

## Project Structure and Key Steps
The notebook is organized into the following main sections:

1.  **Data Loading and Initial Exploration**: Loads the dataset and performs initial checks for missing values and data types.
2.  **Data Preprocessing**: Handles missing values and encodes categorical features.
3.  **Data Visualization (Exploratory Data Analysis)**: Visualizes the distribution of features and relationships between variables.
4.  **Player Churn Prediction (Model 1: Based on Engagement Level)**:
    *   Defines churn based on the 'EngagementLevel' column.
    *   Sets up features and target.
    *   Builds and trains an XGBoost classifier with SMOTE for handling class imbalance.
    *   Evaluates the model using classification report and ROC-AUC.
    *   Identifies churn risk levels based on predicted probabilities.
    *   Performs hyperparameter tuning using RandomizedSearchCV.
5.  **Player Churn Prediction (Model 2: Based on Sessions per Week)**:
    *   Defines churn based on 'SessionsPerWeek'.
    *   Preprocesses data, excluding 'SessionsPerWeek' from features to avoid data leakage.
    *   Builds and trains a Random Forest classifier with SMOTE.
    *   Evaluates the model and plots the ROC curve.
    *   Analyzes feature importance.
    *   Performs hyperparameter tuning using RandomizedSearchCV.
6.  **Synthetic Skill Rating Prediction (Model 3)**:
    *   Creates a synthetic 'SkillRating' column based on 'PlayerLevel', 'AchievementsUnlocked', and 'GameDifficulty'.
    *   Prepares data for regression.
    *   Trains and evaluates Linear Regression, Random Forest Regressor, and XGBoost Regressor models.
    *   Optimizes these models using hyperparameter tuning (GridSearchCV).
7.  **Player Segmentation using K-means clustering**:
    *   Selects features for clustering.
    *   Preprocesses features for clustering.
    *   Determines the optimal number of clusters using the elbow method and silhouette score.
    *   Applies the K-Means algorithm.
    *   Visualizes clusters using PCA.
    *   Analyzes and interprets the characteristics of each cluster.

## Key Findings
*   Initial data exploration revealed insights into the distribution of player demographics, gaming habits, and engagement levels.
*   Churn prediction models were built and evaluated to identify players at risk of churning.
*   The synthetic skill rating prediction models demonstrated high performance.
*   K-Means clustering identified distinct player segments based on their gaming behavior, including "Spenders," "Casual Players," and "Highly Engaged Players."

## Dependencies
The project requires the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- plotly
- xgboost

These libraries are imported in the first code cell of the notebook.

## Conclusion
This project provides a comprehensive analysis of online gaming behavior, offering valuable insights for understanding player dynamics, predicting churn, and tailoring strategies for different player segments.l Report
