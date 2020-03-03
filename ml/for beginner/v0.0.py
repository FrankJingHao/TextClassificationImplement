"""
@简介：tfidf特征/ SVM模型
@成绩： 
"""
#导入所需要的软件包
import pandas as pd
from sklearn.svm import LinearSVC # LinearSV是一个线性SVM的分类器
from sklearn.feature_extraction.text import TfidfVectorizer  # 是构造文本的词频特征所要使用到的一个函数

print("开始...............")

#====================================================================================================================
# @代码功能简介：从硬盘上读取已下载好的数据，并进行简单处理
# @知识点定位：数据预处理   pandas包主要用于数据预处理，包括数据清清洗、数据选择等
#====================================================================================================================
df_train = pd.read_csv('../data/train_set.csv')
df_test = pd.read_csv('../data/test_set.csv')
df_train.drop(columns=['article', 'id'], inplace=True)  # drop丢掉article和id这两行，用不到
df_test.drop(columns=['article'], inplace=True)

#==========================================================
# @代码功能简介：将数据集中的字符文本转换成数字向量，以便计算机能够进行处理（一段文字 ---> 一个向量）
# @知识点定位：特征工程
#==========================================================
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9) #首先定一个类
vectorizer.fit(df_train['word_seg'])                                   #然后调用fit方法
x_train = vectorizer.transform(df_train['word_seg'])                   #文本--》向量
x_test = vectorizer.transform(df_test['word_seg'])
y_train = df_train['class']-1                                          #让label从0开始

#==========================================================
# @代码功能简介：训练一个分类器（根据数字向量学习一个SVM分类器）
# @知识点定位：传统监督学习算法之线性逻辑回归模型
#==========================================================
classifier = LinearSVC()                                               #初始化分类器
classifier.fit(x_train, y_train)                                       #学习

#根据上面训练好的分类器对测试集的每个样本进行预测
y_test = classifier.predict(x_test)                                    #预测       

#将测试集的预测结果保存至本地
df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('../results/beginner.csv', index=False)

print("完成...............")
