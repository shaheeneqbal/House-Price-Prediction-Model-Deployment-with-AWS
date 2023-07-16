# House-Price-Prediction-Model-Deployment-with-AWS
### About
In this project, I am using "house_data" dataset for predicting House prices with advanced Regression Machine Learning algorithms and also creating a web page using HTML. Deploying this house price prediction machine learning model on an Amazon EC2 instance as web application with Flask which makes the model accessible and scalable for real-time prediction queries.
### Introduction
The House Price Prediction Model Deployment with AWS is an innovative solution that leverages the power of machine learning and cloud computing to predict house prices accurately. Using 3 different advanced Regression Machine Learning algorithms such as Linear Regression, Random Forest Regressor and Gradient Boosting Regressor. After doing model evaluation and optimization found out that Gradient Boosting Regressor providing good results. I choose Gradient Boosting Regressor for House Price Prediction deployment.

AWS (Amazon Web Services) provides a comprehensive suite of cloud computing tools and services that enable cost-effective and scalable deployment of machine learning models. The House Price Prediction Model can handle big datasets, produce fast predictions, and easily adjust to changing market conditions by leveraging AWS's infrastructure.
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
### Model Development
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
Once the model is trained and evaluated, it is ready to make predictions on new, unseen house data.
##### Step 12: Create Pickle file
Pickle file automatically generated when running the '.ipynb' file in VS Code.
### Model Deployment on AWS
##### Step 1: Create an AWS Account
If don't have an AWS account, sign up at https://aws.amazon.com/
##### Step 2: Download and Install WinSCP and PuTTY
In our System, download and install these two software: WinSCP and PuTTY
* WinSCP: WinSCP (Windows Secure Copy) is a popular graphical SFTP (SSH File Transfer Protocol) and SCP (Secure Copy) client for Windows. It allows you to securely transfer files between your local machine and a remote server, such as an EC2 instance on AWS.
* PuTTy: PuTTY is a free and open-source SSH and telnet client that allows you to securely connect to remote servers, such as AWS EC2 instances.
##### Step 3: Set Up an EC2 Instance
EC2 (Elastic Compute Cloud) is a virtual server in the AWS cloud. It will be used to host machine learning project. Follow these steps to set up an EC2 instance:
* Go to the EC2 dashboard.
* Click on "Launch Instance."
* Create security group by giving all the details with Inbound and Outbound rules.
* Choose an Amazon Machine Image (AMI) that suits project's requirements. For this project, I am going through with Ubuntu.
* Select an instance type based on computational needs, I am using here t2.micro as it is free tier eligible.
* Create a new key pair or use an existing one for secure access to the instance.
* Configure the instance details, such as network settings and storage.
* Launch the instance.
##### Step 4: Connect to the EC2 Instance
Connect to EC2 instance, need an SSH client and from there copy Public DNS link and check with username. Open WinSCP and paste this Public DNS link in hostname and give username, in password have to give that file path which is generated while creating key pair through 'Advanced' option and then click on Login. After Login, upload all the files to server through WinSCP.
##### Step 5: Set Up the Environment
Need to set up the environment by installing the required dependencies which in PuTTY using in my project such as python, flask, numpy and scikit-learn. 
##### Step 6: Deploy the Model
Once model is trained and ready for deployment, create a Flask web application to host our model and provide a user interface through HTML. Run the flask file in PuTTY and copy the port number, and also copy the Public DNS link from instance and paste them in the browser with colon such as http://ec2-13-53-197-170.us-east-2.compute.amazonaws.com:8080/ 

### Provide the relevant features of a house as input as shown below: 

![image](https://github.com/shaheeneqbal/House-Price-Prediction-Model-Deployment-with-AWS/assets/67499556/c17b1597-f753-4e9a-9e95-1c6df054e57d)

### Estimated the price based on the learned patterns as shown below:

![image](https://github.com/shaheeneqbal/House-Price-Prediction-Model-Deployment-with-AWS/assets/67499556/7b8189c0-0ae3-4062-9cca-a620ccf259f6)

### License
This project is licensed under the MIT License. Feel free to modify and use it as per your requirements.
### Conclusion
The Machine Learning model and cloud computing enables real estate professionals, homeowners, and potential purchasers to make well-informed decisions based on data-driven projections. With the exponential growth of data and advancements in machine learning algorithms, the need for accurate house price predictions has become crucial in the real estate industry. Due to their reliance on manual computations and few data sources, traditional valuation methods frequently fall short of delivering precise predictions. The House Price Prediction Model overcomes these constraints by relying on a powerful machine learning model capable of analysing massive volumes of data and finding significant patterns. This solution opens up new opportunities for increased efficiency, improved decision-making, and better understanding of the housing market dynamics.


