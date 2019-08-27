This is python sklearn implementation.

### Data
Kaggle avazu ctr prediction competition 

### Feature engineering
* Explotory data analysis notebook: an woe/iv style analysis suggest features such as banner_pos, C1x columns are important and this is consistent in model results.
* Date and timestamps: hour/morning/afternoon/evening,day, weekday, is_weekend and one-hot encoding
* Categorical columns: 
    * C20: a lot of -1, guess it may be an indicator(e.g. missing data); the rest seem like id sequence with 100000 as base number.
    * C1,C15,C16,C18: have limited unique values, treated as categorical and one-hot encoding
    * C17,C19,C21: have some outliers and more unique values, bucketize data.
    * We create feature in two ways
        * treat as categorical number  
        * bucketize data based on quantile
        * tried to convert large numbers to log scale, not much insight
* Encoded columns (site_info, app_info, device_info)
    * Convert hexadecimal digits to decimal numbers, and one-hot encoding
    * Create interactions using decimal columns
    * Simply one-hot encoding on string values
    
### Feature selection
* Check feature importance by running boosted tree model, and logistic regression with L1 regulation
* Remove feature columns with standarad deviation lower than 0.05
* Remove feature columns with correlation higher than 0.9
  
### Models
* Logistic Regression with L1 regulation using SGD(LR)
* Gradient Boosted Tree in sklearn
* Gradient Boosted Tree + LR: according to facebook research
* Neural Networks with tensorflow

###  Results
* LR, with feature engineering and regularization, has a pretty good logloss under 0.4 benchmark
* GBDT + LR significantly improved LR model
* Metrics

                  accuracy	   auc	   logloss	   recall
      lr	       0.848566	0.711747	0.389865	0.031035
      gbt	      0.848311	0.712800	0.389059	0.004611
      gbt_lr	   0.849801	0.720873	0.386660	0.031448

* ROC curve <br>
![alt text](https://github.com/ozhong/Data_Science_In_60Days/blob/master/project_classification_kaggle_CTRPrediction/codes_sklearn/Results_analysis/output_19_9.png "ROC curve")
![alt text](https://github.com/ozhong/Data_Science_In_60Days/blob/master/project_classification_kaggle_CTRPrediction/codes_sklearn/Results_analysis/output_19_20.png "ROC curve zoom in")
*More analysis [here](https://github.com/ozhong/Data_Science_In_60Days/blob/master/project_classification_kaggle_CTRPrediction/codes_sklearn/Results_analysis/Results_analysis.md)


### Summary
Two things improved the results most:
* Negative under-sampling: CTR dataset is usually very imblanced. Negative under-sampling reduced the size of training set and improved overall prediction accuracy. 
* Feature engineering: reading site/app/device info by digits and converting to numbers, effectively reduced number of columns compared to simple one-hot encoding. This also allows to create more interaction features to explore more information in site/app/device info.  
* GBDT + LR: LR is simple and fast to train. Adding GBDT improves AUC and logloss on 3rd decimal.

### Optional
* CTR prediction is an important in ads business, and more background readings can be found in homepage
