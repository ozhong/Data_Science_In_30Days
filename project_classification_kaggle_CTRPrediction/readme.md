# CTR Prediction
Click-through-rate(CTR) is a very important metric for evaluating ad performance. As a result, click prediction systems are essential and widely used for sponsored search and real-time bidding.

For this exercise, we are provided with 11 days worth of Avazu data, mostly categorical, to build and test prediction models for CTR.
We'll apply some machine learning techniques and see they fit into the context here.

## Data overview
This is a classification problem with following data attributes. 
- Imbalanced dataset given the low click-through-rate
- Very sparse features
- Massive amount of data

Therefore, the techniques in our toolbox include resampling (for imbalanced data), factorization or regularization(for sparsed features), parallel computing(for large amount of data), and of course, machine learning modelling.

## Models
- LR: Regression models usually are good baseline models, easy to implement and explain.
- GBDT + LR: The methdology stems from Facebook paper(2014) which used tree models for feature engineering and improves the baseline LR.
- FM, FFM: efficient models for sparsed matrices.
- Deep Learning: (to be updated)

Details about the competitions can be found 
https://www.kaggle.com/c/avazu-ctr-prediction/overview
