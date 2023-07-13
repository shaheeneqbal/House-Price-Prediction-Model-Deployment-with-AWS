![image](https://github.com/shaheeneqbal/House-Price-Prediction-Model-Deployment-with-AWS/assets/67499556/f1a83716-9a21-4dd1-95f6-c801f17a6bd6)# House-Price-Prediction-Model-Deployment-with-AWS
### About
In this project, I am using "house_data" dataset for predicting House prices with advanced Regression Machine Learning algorithms and also creating a web page using HTML. Deploying this house price prediction machine learning model on an Amazon EC2 instance as web application with Flask which makes the model accessible and scalable for real-time prediction queries.
### Introduction

### Prerequisites
Before begin, ensure that we have the following prerequisite:
* VS Code
* Excel
* AWS Account
* WinSCP
* Putty
### Data Source
"house_data" dataset from kaggle https://www.kaggle.com/datasets/shivachandel/kc-house-data
### Dataset 
The dataset used for this project is a collection of house data, they are stored in a CSV (Comma Separated Values) file named "house_data.csv" . It contains 4600 rows and 18 columns.
* date: Date house was sold
* price: Price is prediction target
* bedrooms: Number of Bedrooms/House
* bathrooms: Number of bathrooms/bedrooms
* sqft_living: square footage of the home
* sqft_lot: square footage of the lot
* floors :Total floors (levels) in house
* waterfront :House which has a view to a waterfront
* view: Has been viewed
* condition :How good the condition is Overall
* sqft_above :square footage of house apart from basement
* sqft_basement: square footage of the basement
* yr_built :Built Year
* yr_renovated :Year when house was renovated
* street: street of city
* city: city of USA
* zipcode:zip code
* country: USA  
### Getting Started
##### Step 1: Dependencies
To run the code in this repository, need to have the following dependencies installed:
* Python 3.6+
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* HTML
* Flask
* Pickle File
##### Step 2: Explore the Dataset
Explore the dataset to understand its structure, identify any missing values, outliers, or inconsistencies.
##### Step 3: Data Cleaning and Preprocessing
Preprocessing the dataset involves handling missing values and outliers by either imputing them or removing the corresponding instances. Encode categorical variables using techniques such as one-hot encoding or label encoding. Scale numerical features if necessary to ensure all features have a similar range.
##### Step 4: Split the Dataset into Dependent and Independent Features
Spliting the preprocessed dataset into dependent and independent features. The dependent feature is the target variable, which in this case is the price and independent features are the remaining variables that will be used to predict the house price. 
##### Step 5: Feature Selection and Engineering
Feature selection and engineering are used to improve the model's accuracy and interpretability by including the most informative features and transforming them appropriately. It requires a combination of data analysis, statistical techniques, and domain knowledge to select and engineer relevant features effectively for prediction.
##### Step 6: Split the Dataset into Training and Testing
Spliting the preprocessed dataset into training and testing sets. The training set will be used to train the machine learning model, while the testing set will be used to evaluate its performance.
##### Step 7: Machine Learning Algorithms
The choice of algorithm depends on the characteristics of the dataset. For this dataset, using 3 different algorithms: LinearRegression, RandomForestRegressor, and GradientBoostingRegressor
##### Step 8: Train the Model
Train all the 3 machine learning models using the training data. The model will learn the patterns and relationships between the house features and their corresponding prices.
##### Step 9: Evaluate the Model
Evaluate the performance of the trained model using appropriate evaluation metrics such as mean absolute error (MAE), mean squared error (MSE), or root mean squared error (RMSE). These metrics provide insights into how well the model is predicting the house prices.
##### Step 10: Fine-tune the Model
If the model's performance is not satisfactory, consider fine-tuning the model by adjusting hyperparameters or trying different algorithms. This iterative process helps improve the model's accuracy.
##### Step 11: Make Predictions
Once the model is trained and evaluated, it is ready to make predictions on new, unseen house data. Provide the relevant features of a house as input to the model, and it will estimate the price based on the learned patterns.












### License
This project is licensed under the MIT License. Feel free to modify and use it as per your requirements.
### Conclusion



