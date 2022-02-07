
<div align="center">
<a href="https://mail.google.com/mail/u/?authuser=elenahoo@gmail.com">
  <img align="center" alt="elenahoo | Gmail" width="30px" src="https://edent.github.io/SuperTinyIcons/images/svg/gmail.svg" />
	<code>elenahoo@gmail.com</code>
</a>
	<span> ┃ </span>
	
<a href="https://t.me/elenahoo">
  <img align="center" alt="elenahoo | Telegram" width="30px" src="https://edent.github.io/SuperTinyIcons/images/svg/telegram.svg" />
	<code>@elenahoo</code>
</a>
	<span>┃</span>
  <a href="https://discordapp.com/channels/@me/E.Hu#6754/" style="margin-top: 12px;">
  <img  align="center" alt="Discord" width="30px" src="https://raw.githubusercontent.com/peterthehan/peterthehan/master/assets/discord.svg" />
	  <code>E.Hu#6754</code>
</a>
	<span>┃</span>
  <a href="https://twitter.com/messages/compose?text=DM%20text&recipient_id=794664237721329664" style="margin-top: 12px;">
  <img  align="center" alt="elenahoo | Twitter" width="30px" src="https://raw.githubusercontent.com/peterthehan/peterthehan/master/assets/twitter.svg" />
	  <code>@elenahoolu</code>
</a>
<br />

</div>
	
<div align="center">
  <strong>For questions, discussions and freelance work, please feel free to reach out! </strong>
</div>
<br />

# Art Blocks prediction models

A price and number of sales prediction for Art Blocks NFT. The summary results are published on Medium https://elenahoo.medium.com/art-blocks-sale-and-price-prediction-18b812259685

This notebook provides statistical & machine learning models that can predict whether a collection minted before August 1st 2021 will have a resale in August; and how much will the collection be sold for in August. 

Database used in the files is from [Flipside Crypto](https://www.flipsidecrypto.com)

## 1. Data Preparation

There are two types of predictions that can be done:

1.   **Collection level:** To predict which collection would be sold in August and how much would it be sold for. i.e. among all the collections / projects in Art Blocks, based on the different features of each collection such as duration of the mint, number of tokens minted or average sale price, which collection will be sold in August and for how much.
2.   **Token level:** To predict which specific token within the same collection would be sold in August and how much would it be sold for. i.e. within the Chromie Squiggle collection, based on the different traits and features, which one specific token will be sold in August and for how much.

Since each collection is different and will have different token level features, the prediction can only be done on either the collection level or the token level. 

This project will only focus on the collection level prediction. 

## 1.1 Data Structure

The collection data is transformed into two data structure formats, suitable for building time-series regression and non-time-series machine learning models. The models and methodologies are explained in more detail in section 2.

* Non-time series data: (`collection_data.csv` | *unique key*: `collection_name`): contains static non-time dependent information of the collection i.e. artist name, aspect ratio, curation status etc. This data structure is used for machine learning models i.e. Decision Tree, Random Forest, XGBoost.

* Time-series data (`collection_data_ts.csv` | *unique key*: `collection_name`, `year_month`): contains time-dependent information of the collection i.e. sale volume, price. This data structure is used for regression models i.e. OLS, poisson regression.

Collection Level Data

<img width="1817" alt="image" src="https://user-images.githubusercontent.com/36990254/137588282-38fe6f82-779a-4b5b-988b-1483e934c232.png">

Collection Level Time-Series Data

<img width="674" alt="image" src="https://user-images.githubusercontent.com/36990254/137588473-5babd95d-18c6-4c6d-a714-f7281ceec4ed.png">


### 1.2 Data Definition
 
Fields directly from the Flipside database without transformation i.e. artist, aspect_ratio, curation_status etc. are not included here. The definition of the fields created / calcualted from other fields are below:
* COUNT_TOKEN: total number of tokens of the collection
* DAYS_SINCE_MINT: number of days between mint date (`created_at_timestamp`) and September 1st 2021.
* FEATURE_NUMBER: number of features of the collection
* TRAITS_NUMBER: number of traits of the collection
* MINT_CURRENCY: same as tx_currency
* MINT_DURATION: number of minting days
* AUGUST_SALE_COUNT: number of sales in August 2021 from the collection
* AUGUST_SALE_PRICE: average sale price in August 2021 from the collection
* YEAR_MONTH: the year and month of `block_timestamp`
* SALE_COUNT: number of sales of the collection in the particular month
* PRICE_USD: same as `price_usd` in nft_events table
* PRICE_RANGE: difference between the minimum and maximum price_usd of the collection in the particular month

### 1.3 Data Formatting & Cleansing
The data types are transformed so they can be processed through models.

<img width="275" alt="image" src="https://user-images.githubusercontent.com/36990254/137588507-75b7ecbf-2cd2-43ee-8505-64a4163a02f5.png">

## 2. Methodology
### 2.1 Model options
There are two events to be predicted:

1.   The likelihood of a collection or a token being resold in August 2021. This can be formulated as:
```
Y(c) = f(w,x)
```
where collection (or token) c has an outcome Y = 1 if there is sale(s) in August and Y = 0 if there is no sale(s). f(w,x) represents a model with coefficients w and independent variables x. Since Y a binary event and is also categorical, a regression model and a classification model are tried:

- [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
- [Decision Tree Classifier](https://scikit-learn.org/0.15/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

2.   The sale price of the collection or token in August 2021. Since the past sale price is usually a good indicator of the future price, this can be interpreted as the prediction of a time-dependent event:
```
Y(c|tn) = f(w,x,Y(c|t0,...,tn-1))
```
where the formulation of the model is similar to the first event but only time t is added. Y is the sale price of collection (or token) c at time tn (August 2021) and x represents the non-time dependent independent variables;  Y from t0 to tn-1 represent the past prices that can be used to model the price of August 2021. 

We know the price of NFT has sky-rocketed since this year and there is surely a trend in the price. Given most of the time-series models require stationarity (no trend) in the data, so Y needs to be transformed into a percentage change in order to meet the model assumptions:
```
dY(c|t) = f(w,x,dY(c|t0,...,tn-1))
```
where dY is the percentage change in price of the collection (or token) c from month t-1 to t. 

 Some of the time-series models that can be used are:

- [Autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model)
- [ARMA](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model)
- [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

The sale price can also be predicted using a tree-based method from machine learning. In this case, the time-series variables such as historic sale price and sale volume can be used as additional features; and the predicted sale price can be modelled without the time dimension (non-time-series) as: 
```
dY(R|c) = f(w,x)
```
where dY(R|c) is the % price change category a collection (or token) c could fall into. x contains the static variables and the time-dependent variables. The tree-based machine learning models that can be used are:
- [Decision Tree Regression](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Gradient Boosting Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

### 2.2 Evaluation criteria
The most commonly used model performance evaluation metrics for classification problems are:
- [Accuracy Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html): number of correct predictions out of total number of predictions. Since the predicted class is highly unbalanced i.e. there are much more resales than no-resales, there is a tendency for classifiers to always predict the most frequent class. To avoid bias in the evaluation, [Blanced Accuracy Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) is used instead of accuracy score.
- [Log Loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html): measures how close the prediction is to the corresponding actual/true value.
- [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix): it's a table that visualises the performance of a classification model.

For regression models, the following evaulation metrics are used:
- [R-squared](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
- Scattor plot of the predicted vs. actual price

Other evaulation metrics that are not used in this notebook can be found here [Metrics & Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html).

## 3. Collection Level Prediction Models
### 3.1 Which collection will sell in August?
The following models predict which collection will have a resale in August 2021. The result of the prediction is a binary outcome (0 for no resale, 1 for resale). Logistic regression, Decision Tree and Random Forest are used. Decision Tree and Random Forest both predict with 100% accuracy, but Decision Tree is easier to interpretate and has a lower log loss. The most important features to predict if a collection will be resold in August is the number of tokens and the duration of the mint event.

Here is a summary of the model performance for each model:

<img width="917" alt="image" src="https://user-images.githubusercontent.com/36990254/137588719-56d1a2ca-7583-4925-bb42-928094a7ce51.png">

#### 3.1.1 Logistic Regression

The logistic regression correctly predicts 99% of the resales, 83% of the no resales and mis-predict 2 resales into non-resales, 1 non-resale into resale. The most important features are:
- mint duration 
- number of traits a collection has

**Accuracy & Log Loss**

The balanced accuracy for logistic regression is: 0.910

The log loss for logistic regression is: 0.030

Mis-predicted collections

<img width="695" alt="image" src="https://user-images.githubusercontent.com/36990254/137588912-a5879318-15a4-40e8-8b12-3f763d7987d4.png">

The confusion matrix shows the classifier correctly predicts 83% of the no resales and 99% of the resales. 

<img width="379" alt="image" src="https://user-images.githubusercontent.com/36990254/137588936-fce94edb-a1ba-4a37-989a-8732f44c8e3a.png">

The top 3 important features in the Decision Tree Classifier are:
- mint duration
- number traits of a collection
- aspect ratio

**Feature Importance**

<img width="410" alt="image" src="https://user-images.githubusercontent.com/36990254/137588962-1cc64875-99a0-44d5-b7e9-9577e77bbeb3.png">


#### 3.1.2 Decision Tree Classifier

The Decision Tree correctly predicts 100% of the resales and no resales when the tree depth reaches 4. The top 3 most important features are:
- number of tokens
- mint duration
- number of days since the mint
>
The tree path shows the fewer the tokens and the shorter the minting duration a collection has, the less likely the collection will be sold in August.
 
**Accuracy, Log Loss, Decision Tree Plot**

<img width="819" alt="image" src="https://user-images.githubusercontent.com/36990254/137589008-f759748e-a62d-4303-bcaf-6abc3cdd680e.png">

**Feature Importance**

<img width="822" alt="image" src="https://user-images.githubusercontent.com/36990254/137589017-66f774d6-36af-4897-a156-94ad8ffc6618.png">

#### 3.1.3 Random Forest Classifier

Random Forest is similar to Decision Tree, except that it uses ensemble method to create sub-samples to build many decision trees to train the model better. Random Forest also predicts quite well with 100% accuracy score, but with a higher log loss than Decision Tree. 
>
Since a simple decision tree already predict 100% accuracy, Random Forest is not going to improve model performance, hence it's only shown as an additional choice here. The most important features are the same as Decision Tree:
- mint duration
- number of tokens a collection has

**Accuracy, Log Loss & Confusion Matrix**

The balanced accuracy for Random Forest is: 1.000

The log loss for Random Forest is: 0.011

<img width="388" alt="image" src="https://user-images.githubusercontent.com/36990254/137589040-6bd255eb-b7a9-4b5b-856e-7a757939b4c7.png">

**Feature Importance**

<img width="807" alt="image" src="https://user-images.githubusercontent.com/36990254/137589108-17fadd48-65c2-4688-b9a1-adaffce4c651.png">

### 3.2 At what price the collection will sell in August?

As mentioned in section 2, the prediction can be based on time-series regressions or machine learning tree-based regression models.
>
**Why tree-based regression model is more suitable than time-series model in this case**
>
The issue here is that a time-series model usually predicts a price trend of a particular item (i.e. a single stock price), in this case a collection. We have over 100 unique collections, all with different price history, sale volumes and traits etc. So each collection will need its own model to fully incorporate the price trend in the prediction (which is a lot of models!). 
> 
Also, the time history is quite short for each collection. With less than 1 year of history and maximum 8 or 9 monthly data points for each collection, the time-series model will not likely to give reliable and robust results. Using daily data could increase the sample size, but not all collections have a sale everyday. A time-series model often require equal time intervals, which means the days with missing sale data due to no sale will have to be approximated with interpolation methods. 
>
Based on these reasons, and the fact that there are a lot of categorical variables in the data, tree-based regression models will be more suitable than time-series model. A summary of the 3 selected models and their performances are below. The most important feature to predict the August price turns out to be July's price from all 3 models.

**Model Performance**

<img width="525" alt="image" src="https://user-images.githubusercontent.com/36990254/137589133-050efa20-796d-4719-9638-85098abcd5ea.png">

#### 3.2.1 Data Preparation
**Transform time-series variables**
>
In order to to use the time-series data in a non-time-series tree-based model, the monthly average price in the collection level time-series data is pivoted to the collection level data, so each month's average sale price becomes an additional column, which will serve as features in the classification models. The number of sales from each month is also transformed in the same way.

<img width="1688" alt="image" src="https://user-images.githubusercontent.com/36990254/137589192-45ebbb7c-7c8b-4df2-a9e7-c6c382fe0827.png">


**Predict % change of sale price from July to August**
>
The price modelling data contains both the static features from `collection_level_data.csv` and the time-dependent features from `collection_level_data_ts.csv`.

Three different tree-based regression models are used here:
- [Decision Tree Regression](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html): an algorithm that can predict continous dependent variables in a tree structure.
- [Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html): a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
- [Gradient Boosting Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html): builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function.

#### 3.2.2 Decision Tree Regression
The initial Decision Tree Regression is without any hyper-parameter. When the tree depths reach 18, the R-squared is 100%. Since the tree is very deep and is likely to over-fit the data, different trials of tree depths vs. R-squared is plotted and the elbow point is the optimal tree depth, which is 5.

The most important features to predict the August sale price are:
- July's sale price 
- July's sale number
- December's sale number 
- curation status. 
>
The scatter plot shows how close the predicted price is to the actual. The perfect prediction will form a 45 degrees diagonal line. In the plot it shows most of the points are on the diagonal line, except for some low price predictions.

The R-squared for Decision Tree Regression is: 1.000

The tree depth without hyper-parameter tuning is: 18.000

<img width="420" alt="image" src="https://user-images.githubusercontent.com/36990254/137589225-33705fac-cda1-4b70-8cc2-88a3deec531d.png">


The R-squared for Decision Tree regression is: 0.997

The optimal tree depth after hyper-parameter tuning is: 5.000

<img width="806" alt="image" src="https://user-images.githubusercontent.com/36990254/137589243-dc6c0e20-a43c-4a5d-acfe-9367ec7adbdd.png">

**Decision Tree Plot**

https://colab.research.google.com/drive/1HQBG-J9fbNX_G6TWfWQeb9ngtTHC7154#scrollTo=5-ASxa8ZOOfu

<img width="633" alt="image" src="https://user-images.githubusercontent.com/36990254/137589310-6557cac0-3f39-470d-b8e7-42523f477f0d.png">

#### 3.2.2 Random Forest Regression
Random Forest Regression predicts slighly better than Decision Tree Regression, with R-squared of 97.3%. Since a simple decision tree already has R-squared of 99.7% and it's more easily interpretable, there is no need to explore other models to improve the performance. So Random Forest is shown as an extra choice here.

The top 3 most importance features are:
- July's sale price 
- June's sale price
- May's sale price
>

The predicted price vs. actual in the scatter plot shows some points are off away from the perfect 45 degrees diagonal line, indicating the prediction is less accurate than Decision Tree.

The R-squared for Random Forest Regression is: 0.973

<img width="800" alt="image" src="https://user-images.githubusercontent.com/36990254/137589326-ab8586fe-cdf0-4868-ad09-c8e82c987e01.png">

<img width="633" alt="image" src="https://user-images.githubusercontent.com/36990254/137589341-a3dfd152-860e-4565-9321-c68312a768c3.png">

#### 3.2.4 Gradient Boosting Regressor
Gradient Boosting Regressor is similar to Decision Tree, except that it uses ensemble method to learn from the previous step's error and build the next step in the decision trees in order to train the model better. Since a simple decision tree already has R-squared of 99.7% and it's easily interpretable, there is no need to explore other models to improve the performance. So Gradient Boosting is also shown as a different additional choice here.

Gradient Boosting Regressor as expected improves the performance to 99.9% R-squared due to its greedy search algorithm nature. The top 3 most importance features are:
- July's sale price 
- July's sale number
- March's sale price 
>

The predicted price vs. actual in the scatter plot shows almost a perfect line, except for a couple of underpredicted outliers when looking at the log-scale plot.

The R-squared for Gradient Boosting Regression is: 0.999

<img width="801" alt="image" src="https://user-images.githubusercontent.com/36990254/137589363-c943b128-8ff2-4850-ac76-ebdb94bd889b.png">

<img width="627" alt="image" src="https://user-images.githubusercontent.com/36990254/137589373-61e0c672-ac88-4d9f-9d15-ef20ce55ba8c.png">

## 4. Conclusion

(1) For the prediction of whether a collection will be resold in August, all the models show that the most important features are:
- the number of tokens a collection has: the fewer tokens, the less likely of resale
- the duration of the minting event: the shorter the duration, the less likely of resale

The smaller number of tokens and shorter minting duration lead to a lower chance of resale in August.

(2) For the prediction of the resale price in August, all 3 tree-based regression models have a high R-squared more than 97%. The most important feature is **July's sale price**.

Decision Tree is suitable for predicting both events - (1) and (2). The model performance is good with an accuracy score of 100% for event (1) and 99.7% for event (2).

## 5. Limitations & Future Improvements

Although the model performance is very well, this might be due to the fact that there are only very few collections that were not resold in August. In such an unbalanced class of too many resales, the model could simply predict the most frequent class to get a high accuracy score. There are also some other limitations of the models, which are summarised along with the unbalanced class issue below:
- Unbalanced class of too many resales, model tends to predict the most frequenty class to achieve high accuracy score.
- Not enough data in no resale category to do a K-fold cross validation or training/testing split; so the models cannot be tested on out of sample data.
- Not enough long sale history to build a time-series model.
- The set up of monthly data as features would mean the model needs to be recalibrated every month to incorporate new data.

The above limitations can all be mitigated with more data with longer history, which can be achieved as time goes.


