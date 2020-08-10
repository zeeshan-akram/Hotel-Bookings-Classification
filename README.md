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
![deleted or not found](https://github.com/zeeshan-akram/Hotel-Bookings-Classification/blob/master/non-stay-bookings.png)
**For more visuals check source code**
