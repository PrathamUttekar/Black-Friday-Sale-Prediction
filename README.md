
# Black Friday Sales Prediction

## Overview
This project aims to predict sales for Black Friday using various machine learning models. The dataset contains customer purchase information and various features related to the shopping behavior. The goal is to build and evaluate multiple regression models to predict the total purchase amount.

## Dataset
The dataset used in this project is BlackFridaySales.csv, which includes features such as:

- User_ID
- Product_ID
- Gender
- Age
- Occupation
- City_Category
- Stay_In_Current_City_Years
- Marital_Status
- Product_Category_1
- Product_Category_2
- Product_Category_3
- Purchase


1. ### Data Loading and Exploration
- pd.read_csv("BlackFridaySales.csv"): Loads the dataset into a DataFrame.
- df.info(): Provides a summary of the dataset including data types and non-null counts.
- df.isnull().sum(): Checks for missing values in each column.
- sns.histplot(df["Purchase"], color='g'): Plots the distribution of the target variable (Purchase) to visualize its range and skewness.
- df.corr(): Computes and displays the correlation matrix to understand relationships between features.

2. ### Data Preprocessing
- da = df.copy(): Creates a copy of the DataFrame to avoid modifying the original dataset.
- LabelEncoder(): Encodes categorical variables (Gender, Age, City_Category) into numerical values.
- fillna(0).astype('int64'): Replaces missing values in Product_Category_2 and Product_Category_3 with 0 and converts them to integer type.
- da.drop(["User_ID", "Product_ID"], axis=1): Removes columns that are not useful for modeling.
- X and y: Separates the features and target variable.

3. ### Train-Test Split
- train_test_split(): Splits the dataset into training and testing sets. test_size=0.2 means 20% of the data is used for testing, and random_state=42 ensures reproducibility.

4. ### Linear Regression
- LinearRegression(): Creates a Linear Regression model.
- fit(X_train, y_train): Trains the model on the training data.
- predict(X_test): Makes predictions on the test data.
- mean_absolute_error, mean_squared_error, r2_score: Calculate various evaluation metrics to assess model performance:
    - MAE: Average absolute difference between predicted and actual values.
    - MSE: Average squared difference.
    - RMSE: Square root of MSE, providing error in the same units as the target variable.
    - R-squared: Proportion of variance explained by the model.

5. ### Decision Tree Regressor

  - DecisionTreeRegressor(): Creates a Decision Tree model.
  - random_state=0: Ensures reproducibility.
  - fit(X_train, y_train): Trains the model.
  - predict(X_test): Makes predictions.
  - Evaluation metrics are calculated as before.
  
6. ### Random Forest Regressor
  - RandomForestRegressor(): Creates a Random Forest model.
  - fit(X_train, y_train): Trains the model.
  - predict(X_test): Makes predictions.
  - Evaluation metrics are calculated similarly.  

7. ### XGBoost Regressor
  - XGBRegressor(): Creates an XGBoost model with specified hyperparameters.
  - fit(X_train, y_train): Trains the model.
  - predict(X_test): Makes predictions.
  - Evaluation metrics are calculated similarly to previous models.


## Results
After running the code, you will get performance metrics for each model. This will help you compare their effectiveness in predicting Black Friday sales and choose the best model for your needs.

## Conclusion

This project demonstrates the use of various machine learning models for predicting sales based on customer data. By comparing the performance of Linear Regression, Decision Tree Regressor, Random Forest Regressor, and XGBoost Regressor, you can select the most effective model for your needs.

