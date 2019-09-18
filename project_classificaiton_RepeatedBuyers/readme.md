
### Problem Definition
- Merchants sometimes run big promotions (e.g., discounts or cash coupons) on particular dates (e.g., Boxing-day Sales, "Black Friday" or "Double 11 (Nov 11th)”, in order to attract a large number of new buyers. It is important for merchants to identify who can be converted into repeated buyers. By targeting on these potential loyal customers, merchants can greatly reduce the promotion cost and enhance the return on investment (ROI).

### Data Description
- The data set contains anonymized users' shopping logs in the past 6 months before and on the "Double 11" day,and the label information indicating whether they are repeated buyers. 

### Solution:
- feature_engineering: process training data and create statistical features
- model_fit: tune parameters for LightGBM model
More notes on features
https://machinelearning100days.wordpress.com/2019/09/18/用户画像%ef%bc%88user-profile%ef%bc%89实例%ef%bc%9a天猫回购客户预测%ef%bc%88repeated-buyers-in-tmall/
=============================================================

### 介绍： 
- 通过长期的用户行为数据判断用户回购，从而提高市场营销的效率。

### 数据：
- 天池数据集。半年左右用户行为，包括用户，商家，商品（分类和品牌）的信息，用户行为包括点击，加入购物车，购买，加入心愿单等。
测试集大约有26万行，用户行为信息大约5000万行。

=============================================================
More Info
https://tianchi.aliyun.com/competition/entrance/231576/information
