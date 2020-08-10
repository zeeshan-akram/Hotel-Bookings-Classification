# Hotel Bookings Cancel Classifier: Project Overview
* The model predicts whether the client will cancel booking or not with accuracy of 85.91%. I did this project to learn about classification problems and it may help hotels to check whether their client cancels the booking or not.
* Get data from Kaggle.com datasets [Hotel booking demand](https://www.kaggle.com/jessemostipak/hotel-booking-demand).
* Cleaned Data (Handled missing values).
* Perform exploratory data analysis to know about data.
* Perform feature engineering.
* Built model and optimize it.
* Built client facing API using flask.
## Code and Resources Used
**Python Version:** 3.7 <br>
**Packages:** numpy, pandas, seaborn, matplotlib, plotly, sklearn, lightgbm, xgboost, pickle, flask <br>
**Learn EDA From:** [Kaggle Notebook](https://www.kaggle.com/marcuswingen/eda-of-bookings-and-ml-to-predict-cancelations) <br>
**About Classification Problems:** [Article](https://towardsdatascience.com/machine-learning-classifiers-a5cc4e1b0623) <br>
**GitHub Markdown:** [Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) <br>
## Data Cleaning
The first thing i've done after getting data is data cleaning. It is very important after cleansing data we can perform EDA and later modeling.<br>
**Handled Missing Values:** 
* Handled missing categorical values by grouping related features.
* Handled some numerical categorical missing values.
## Exploratory Data Analysis
I checked bookings ratio, seasonal bookings, bookings from different country, markets and distrbutions channel.<br>
**Some Visuals:**<br>
![deleted or not found](https://github.com/zeeshan-akram/Hotel-Bookings-Classification/blob/master/perc-booking-cancel-booking.png)
![deleted or not found](https://github.com/zeeshan-akram/Hotel-Bookings-Classification/blob/master/change-in-price.png)
![deleted or not found](https://github.com/zeeshan-akram/Hotel-Bookings-Classification/blob/master/month-reservations.png)
![deleted or not found](https://github.com/zeeshan-akram/Hotel-Bookings-Classification/blob/master/market-segments.png)
![deleted or not found](https://github.com/zeeshan-akram/Hotel-Bookings-Classification/blob/master/daily-bookings.png)
![deleted or not found](https://github.com/zeeshan-akram/Hotel-Bookings-Classification/blob/master/country-bookings.png)
![deleted or not found](https://github.com/zeeshan-akram/Hotel-Bookings-Classification/blob/master/payments-market-segments.png)
![deleted or not found](https://github.com/zeeshan-akram/Hotel-Bookings-Classification/blob/master/rooms-price.png)
![deleted or not found](https://github.com/zeeshan-akram/Hotel-Bookings-Classification/blob/master/special-request.png)
![deleted or not found](https://github.com/zeeshan-akram/Hotel-Bookings-Classification/blob/master/non-stay-bookings.png)<br>
**For more visuals check source code**
## Feature Engineering
After EDA I perform feature engineering modeling. Incorrect or inconsistent data leads to false conclusions. And so, I selected only important features.
I perform following operations

* Check Pearson correlation.
  * Removed correlated predictors to reduce multicolinearity.
  * Removed features with very low correlation with target variable.
* Perform label and One hot encoding.
* Scale down data with Min Max Scaler.
## Model Building 
First I split data into train and test data then use Boosting and Bagging techniques.<br>
**Bagging** takes the advantage of ensemble learning wherein multiple weak learner outperform a single strong learner. It helps reduce variance and thus helps us avoid overfitting.<br>
**Boosting** algorithms seek to improve the prediction power by training a sequence of weak models, each compensating the weaknesses of its predecessors.<br>
## Model Performance
Random Forest outperforms of all models.<br>
* XGBoost: Accuracy = 83.81%
* Decision Tree: Accuracy = 81.71%
* LGBM : Accuracy = 82.90%
* Random Forest: Accuracy = 84.68 % <br>
I selected Random Forest<br>
**After Performing Hyperparameter Tuning:**<br>
Random Forest Accuracy = 85.91%.
