### 项目介绍
拍拍贷“魔镜风控系统”从平均400个数据维度评估用户当前的信用状态，给每个借款人打出当前状态的信用分，在此基础上，再结合新发标的信息，打出对于每个标的6个月内逾期率的预测，为投资人提供了关键的决策依据

### 数据集
公开国内网络借贷行业的贷款风险数据，包括信用违约标签（因变量）、建模所需的基础与加工字段（自变量）、相关用户的网络行为原始数据。

初赛数据包括3万条训练集和2万条测试集。复赛会增加新的3万条数据，供参赛团队优化模型，并新增1万条数据作为测试集。

        • Master: 每一行代表一个样本（一笔成功成交借款），每个样本包含200多个各类字段。
                idx：每一笔贷款的unique key，可以与另外2个文件里的idx相匹配。
                UserInfo_*：借款人特征字段
                WeblogInfo_*：Info网络行为字段
                Education_Info*：学历学籍字段
                ThirdParty_Info_PeriodN_*：第三方数据时间段N字段
                SocialNetwork_*：社交网络字段
                LinstingInfo：借款成交时间
                Target：违约标签（1 = 贷款违约，0 = 正常还款）。测试集里不包含target字段。
        • Log_Info
                借款人的登陆信息。
                ListingInfo：借款成交时间
                LogInfo1：操作代码
                LogInfo2：操作类别
                LogInfo3：登陆时间
                idx：每一笔贷款的unique key
        • Userupdate_Info
                借款人修改信息
                ListingInfo1：借款成交时间
                UserupdateInfo1：修改内容
                UserupdateInfo2：修改时间
                idx：每一笔贷款的unique key

### 评判标准
模型评价标准 

定义：本次比赛采用AUC来评判模型的效果。AUC即以False Positive Rate为横轴，True Positive Rate为纵轴的ROC （Receiver Operating Characteristic）curve下方的面积的大小。 

    AUC = sum(S_i)/(M*N)
M 为正样本个数，N 为负样本个数，M * N 为正负样本对的个数。Si为第i个正负样本对的得分，定义如下： 

    when score_i_p > score_i_n, S_i = 1;
    when score_i_p = score_i_n, S_i = 0.5;
    when score_i_p < score_i_n, S_i = 0;
scorei-p为正负样本对中，模型给正样本的评分，scorei-n为正负样本对中，模型给负样本的评分。AUC值在[0,1]区间，越高越好。 

相关信息链接
https://www.ppdai.ai/mirror/showCompetitionRisk

