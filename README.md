# SOMS_NITC_AIMLBDA-project
Decision Tree for Insurance claims redflags

**Abstract**
The popularity of using machine learning (ML) and Big Data Analytics in business applications has increased as a result of technology developments and the realization of big data in various industries, with no exception to the insurance industry. One such attempt is made in the project work. The project aimed at keeping track of all the ambiguous healthcare insurance claims such as claims which are not payable due to any of the following reasons: duplicate bills, coverage, crossing buffer limit, age of claimant, violation of waiting period exclusion etc. The present work made an effort to develop a machine learning algorithm which will keep track of such claims. However, data can be fine-tuned to suit the ML environment. If ML is appropriately implemented, the results can save a lot of outgo and thus improve the profitability of the insurance companies and The Third Party Administrators (TPA’s). Further, it will also result in savings for the customers, since the mediclaim premium would not be increased due to such extra outgo. The proposed algorithm was able to identify various redflags, which was ultimately represented by total redflags feature. The algorithm can be further improved by considering the realtime data on insurance claims, and used by the third party administrator of insurance company network. 


**Objective of the research**
The project aim to understand how machine learning algorithms, primarily classification type, can help insurance companies to deduce patterns and identify various red flags in the insurance claims, such as capping red flags, duplicate claim red flag and deceases red flag. 

**Data Collection**
The design of a machine learning estimator begins with a dataset. Data was collected from the lake of data available at kaggle.com database. It was later cleaned and transformed based on the requirements. Initially there were 36 columns (variables/features) and 649 cases. 
**Data Preparation**
**Data Cleaning and Transformation**
Data Cleaning is the first step after the data is retrieved. It consists of detecting and removing inaccurate, false, incomplete, corrupt, or irrelevant records from the dataset. Data cleaning is also called “Data Pre-Processing”. One of the most common approaches for data cleaning is variable-by-variable cleaning. In this approach, illegal or misspelt features values are removed from the dataset based on certain factors such as the minimum and maximum value should not be outside the permissible range, the variance and standard deviation should not be more than the threshold value and there should not be any misspelt values in the dataset. Data values are either removed or manipulated depending on the coarseness of the data value. In case there are missing feature values, the feature was either eliminated if there were many missing values or the missing values were replaced by a dummy value (treating the missing value itself as a new value). Mean substitution is the most common approach for treating missing values. New features such as capping redflags, diseases redflags and duplicate redflags are added using the information in the present data. 
Dataset have a total of 646 cases with 270 males and 394 females. Dataset also provides the details about the relationship between the claimants and the insurer, mode of settlement, cover start and end date, DOB, address details like district, state, expense details under various heads etc. 
After careful observation, certain cases were dropped due to absurd entries. Few columns were added as per the requirement of analysis such as redflags. Few columns such as patient information like district, time of intimation, amount deducted, and various details of expenditure were removed, to arrive at 23 columns. Table 2 gives the details about the various column in the data set.


This part of the report presents the coding used during the execution of decision tree algorithm in Google colab and the corresponding result.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# File is read 
df=pd.read_csv("/content/Mediclaim.csv")
# returns the column details
df.head()
# returns the column types
df.info()
# returns all the columns with null values count
df.isna().sum()
# returns the size of dataset
df.shape
# returns the countplot of transaction type
sns.countplot(df['type'],label="count")
# returns the count of total redflags
df['Total Red Flag'].value_counts
#get the correlation
df.iloc[:,1:24].corr()
# visualize the correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:32].corr(),annot=True,fmt=".0%")
input = df.drop('Total Red Flag', axis='columns')
target = df['Total Red Flag']
input
target
from sklearn.preprocessing import LabelEncoder
le_disease_result = LabelEncoder()
le_capping_result = LabelEncoder()
le_duplicate_result = LabelEncoder()

input['disease_n'] = le_capping_result.fit_transform(input['disease_result'])
input['capping_n'] = le_capping_result.fit_transform(input['capping_result'])
input['duplicate_n'] = le_capping_result.fit_transform(input['duplicate_result'])
input.head()
input_n = input.drop (['type','cover start date','cover end date','relationship','gender','dob','date_intimation','date_admission','date_discharge','state',' claimed_amt ',' sum insured ',' approved_amt ','claim_status','diseases reported','Bill No.',' Capping ','disease_result','capping_result','duplicate_result'], axis='columns')

input_n
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(input_n,target)
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')

model.score(input_n, target)
sns.countplot(df['Total Red Flag'],label="count")

# gives result
model.predict(X_test)
Y_test

This project have developed the machine learning-based classification model to detect various red flags in health insurance policy claims. The Red Flags may not necessarily be dubious claim only; it may be due to typo error, data entry error or some other unintentional error. Irrespective of the cause of the claim the cost involved cannot be ignored.  Health claims and related management expenses are in the range of Rs.45,000 Crores (for FY 2019-2020, EPW Research Foundation- India Time Series) in Indian Insurance Market. Suppose even if this project is able to detect Red Flags in the range of 1% of the Health Insurance Industry in India I.e. 1% of Rs.45,000 Crores equal to Rs.450 Crores would be saved and related costs of insurance premium could be controlled making it further affordable for marginal section of the society. If we are able to scale this project and develop and integrate all the stakeholders like insurer, TPAs, Hospitals and insured through such ML algorithm using Decision Tree can improve profits and reduce insurance cost and bring in more transparency in the system.
The Association of British Insurers suggests that fraudulent claims cost the UK insurance industry over 1 billion a year and fraudsters continuously develop new types of scams (Morley et al., 2006). If we project he same amount to present times and take into consideration not only developed markets but developing insurance markets the impact of fraudulent claims on insurance industry can be huge hence our project would be able to track such claims through Red Flags for different parameters. The advantage of decision tree is that dynamic Red Flags can be added to make the model contemporary and save billions of tax payer’s money.  To make it more robust and inclusive for multi parameters and multi dimensional Random Forest can be the next stage for this project and Government should provide tax incentive to companies deploying such Red Flag projects or Applications since it will help in the development of the economy of the nations. The detected Red Flags can be further subjected to detailed investigations. The industry experts should be sensitized both by the Government, Industry bodies and insurance companies and experts be trained on this technological advancement and fraud claim detection techniques. Various incentives should be launched by the Government and industry for adoption of such technologies and applications for reduction in the number and quantum of dubious claims. Further in house teams can be trained to develop such models and applications for advanced stages Red Flags detector.
