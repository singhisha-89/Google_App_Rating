#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("googleplaystore.csv")
df.head()


# In[208]:


print(df.dtypes,df.isnull().sum())
df.shape


# In[209]:


df.dropna(inplace=True)
print(df.shape)


# **Data Cleaning**
# 
# Variables seem to have incorrect type and inconsistent formatting,also Size column has sizes in Kb as well as Mb,Multiply the value by 1,000, if size is mentioned in Mb

# In[210]:


print(df.Size.value_counts())

def change(Size):
    if 'M'in Size:
        x=Size[:-1]
        x=float(x)*1000
        return x

    elif 'k'in Size:
        x=Size[:-1]
        x=float(x)
        return x
    
    else: return None

    
df.Size=df.Size.map(change)
df.Size.value_counts()


# In[211]:


print(df.Size.isnull().sum())
df.Size.fillna(method='pad',inplace=True)
print(df.Size.isnull().sum())


#  
# * Reviews is a numeric field that is loaded as a string field.Convert it to numeric
# 
# * Installs field is currently stored as string and has values like 1,000,000+, remove ‘+’, ‘,’ from the field, convert it to integer
# 
# * Price field is a string and has  symbol. Remove‘ ’ sign, and convert it to **numeric**
# 
# 

# In[212]:


df.Reviews=df.Reviews.astype('float')
print(df.Installs.value_counts()[:5])


# In[213]:


df.Installs=df.Installs.map(lambda x:x.replace(',','').replace('+',''))
print(df.Installs.value_counts()[:5])


# In[214]:


df.Installs=df.Installs.astype('float')
print(df.Price.value_counts()[:5])


# In[215]:


df.Price=df.Price.map(lambda x:x.replace('$',''))
print(df.Price.value_counts()[:5])


# In[216]:


df.Price=df.Price.astype('float')
print(df.dtypes)


# ## Sanity Check :
# 
# * Average rating should be between 1 and 5 as only these values are allowed on the play store. Drop the rows that have a value outside this range.
# * Reviews should not be more than installs as only those who installed can review the app. If there are any such records, drop them.
# * For free apps (type = “Free”), the price should not be >0. Drop any such rows.

#  
# 
# **Outlier treatment:** 
# 
# Price: From the box plot, it seems like there are some apps with very high price. A price of $200 for an application on the Play Store is very high and suspicious!
# 
# Check out the records with very high price
# 
# Is 200 indeed a high price?
# 
# Drop these as most seem to be junk apps
# 
# Reviews: Very few apps have very high number of reviews. These are all star apps that don’t help with the analysis and, in fact, will skew it. Drop records having more than 2 million reviews.
# 
# Installs:  There seems to be some outliers in this field too. Apps having very high number of installs should be dropped from the analysis.
# 
# Find out the different percentiles – 10, 25, 50, 70, 90, 95, 99
# 
# Decide a threshold as cutoff for outlier and drop records having values more than that

# In[217]:


print(len(df[df.Rating>5]))


# In[218]:


print(len(df[df.Rating<1]))


# In[219]:


print(len(df[df.Reviews>df.Installs]))


# In[220]:


print(len(df[(df.Type=='free')&(df.Price>0)]))


# In[221]:


df=df[df.Reviews<df.Installs].copy()
print(df.shape)


# In[222]:


#A price of 200$ seems to be too much for an app thus we need to drop these as most seem to be junk apps
print(len(df[df.Price>200]))
df=df[df.Price<200].copy()
print(df.shape)


# In[223]:


# There are a very less number of Apps with high number of reviews.Removing the applications having reviews more than 
# 2 million will help with the analysis and these are all star apps
print(len(df[df.Reviews>=2000000]))
df=df[df.Reviews<=2000000].copy()
print(df.shape)


# In[224]:


print(df.Installs.quantile([.25,.50,.75,.90,.99]))


# In[225]:


sns.distplot(df["Installs"],kde=False)


# In[226]:


print(len(df[df.Installs>= 10000000]))
df=df[df.Installs<=10000000].copy()
print(df.shape)


# In[227]:


#There seems to be some outliers in installs field too. Hence setting the threshold at 10000000.


# 

# # Performing univariate analysis: 
# Boxplot for Price
# 
# Are there any outliers? Think about the price of usual apps on Play Store.
# 
# Boxplot for Reviews
# 
# Are there any apps with very high number of reviews? Do the values seem right?
# There are a very less number of Apps with high number of reviews
# **- yes there are a few Apps with higher reviews though the values seems to be outliers
# 
# Histogram for Rating
# 
# How are the ratings distributed? Is it more toward higher ratings?
# **- yes it is distributed more towards higher ratings and there are outliers as well
# 
# Histogram for Size
# **- Size and Rating does not seem to have significant outliers
# 
# Note down your observations for the plots made above. Which of these seem to have outliers?

# In[228]:


print(df.hist(['Rating','Reviews','Size','Installs','Price'],figsize=(12,8),xlabelsize=12,ylabelsize=12))


# In[229]:


df.boxplot(fontsize=15)


# In[230]:


plt.figure(figsize=(25,8))
sns.scatterplot(df.Price,df.Rating,hue=df.Rating)
plt.show()


# ### While there is not a very clean pattern, it does look that the higher priced apps have better rating. Although, there are not a lot of apps which are high priced, but the pattern is apparent

# In[231]:


plt.figure(figsize=(25,8))
sns.scatterplot(df.Size,df.Rating,hue=df.Rating)
plt.show()


# In[232]:


plt.figure(figsize=(11,8))
sns.scatterplot(df.Reviews,df.Rating,hue=df.Rating)


# Apps with low Reviews have higher Rating but there are a few apps with higher Reviews which have high Rating

# In[233]:


plt.figure(figsize=(12,8.27))
sns.boxplot(df['Content Rating'],df.Rating)


# ### While the median rating for most others is similar, the rating for “Adults Only 18+” is the highest.

# In[234]:


plt.figure(figsize=(25,8.27))
sns.boxplot(df.Category,df.Rating)
plt.xticks(fontsize=18,rotation='vertical')
plt.yticks(fontsize=18)
plt.xlabel("Category",fontsize=20)
plt.ylabel("Rating",fontsize=20)


# ### Apps around Health & Fitness, Books and Reference, Events seem to have the highest median ratings.

# # Data preprocessing

# #### It seems from the histogram the variables has some skewness and from boxplot it is evident that it has outliers too...
#   It can be corrected by applying log

# In[248]:



inp1=pd.DataFrame(df)

inp1.Reviews=inp1.Reviews.apply(func=np.log1p)
inp1.Installs=inp1.Installs.apply(func=np.log1p)
 
inp1.hist(column=['Reviews','Installs'])


#  * Deleting Unnecessary Variables

# In[249]:


inp1.drop(["App", "Last Updated", "Current Ver", "Android Ver"], axis=1, inplace=True)
print(inp1.shape)
inp2=pd.get_dummies(inp1,drop_first=True)
print(inp2.columns)


# # ** Linear Regresssion Model

# In[250]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
from statsmodels.api import OLS
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as ms


# * splitting 70% of the data to the training set while 30% of the data to test set using below code.

# In[252]:


df_train=inp2.iloc[:,1:]
df_test=inp2.iloc[:,:1]
X_train, X_test, y_train, y_test = train_test_split( df_train, df_test, test_size=0.30, random_state=1)
X_train.shape,X_test.shape


# * Building Model & Predicting the Ratings, also checking the difference between the actual value and predicted value.

# In[253]:


Model=linreg.fit(X_train, y_train)
predict=linreg.predict(X_test)

y_test=np.array(y_test)
predict=np.array(predict)

a=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':predict.flatten()});a.head(10)


# In Below figure we can observe here that the model has returned pretty good prediction results.

# In[254]:


fig=a.head(25)
fig.plot(kind='bar',figsize=(10,8))


# ### Model Summary

# In[255]:


results=OLS( y_train,X_train).fit()
results.summary()


# In[256]:


print('R2_Score=',r2_score(y_test,predict))
print('Root Mean Squared Error=',np.sqrt(ms(y_test,predict)))
print('Prediction Error Percentage is',round((0.50/np.mean(y_test))*100))


# ### Summary Interpretation

# * A large F-statistic will corresponds to a statistically significant p-value (p < 0.05). In the data, the F-statistic equals 3545. that leads to less P_Value which says that at least one of the predictor variables is significantly related to the outcome variable.
# * From the output above, the adjusted R2 is0.141322, meaning that the observed and the predicted outcome values are highly correlated, which is very good.
# * The prediction error RMSE (Root Mean Squared Error), representing the average difference between the observed known outcome values in the test data and the predicted outcome seems to be 0.50 which is very good thus represents the error rate of 12%

# In[ ]:





# In[ ]:




