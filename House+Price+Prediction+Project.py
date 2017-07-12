
# coding: utf-8

# In[41]:

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("kc_house_data.csv")


# In[42]:

df.info()


# In[43]:

df.describe()


# In[44]:

fig = plt.figure(figsize = (8,5))
ax = fig.add_subplot(111)
ax.hist(df.sqft_living)
plt.tight_layout()


# In[45]:

df.corr()


# In[46]:

df = df[['price','bedrooms','bathrooms','sqft_living','floors','waterfront','view','grade','sqft_above','sqft_basement','sqft_living15','yr_built','yr_renovated','zipcode']]
df.head()


# In[47]:

fig = plt.figure(figsize = (8,5))
ax = fig.add_subplot(111)
plt.hist(df.price)
plt.tight_layout()


# In[48]:

fig = plt.figure(figsize = (20,20))

#1

ax1 = fig.add_subplot(331)
ax1.scatter(df.price,df.bedrooms)
ax1.set_ylabel('bedrooms', fontsize = 15)
ax1.set_xlabel('price', fontsize = 15)

#2
ax2 = fig.add_subplot(332)
ax2.scatter(df.price,df.bathrooms)
ax2.set_ylabel('bathrooms', fontsize = 15)
ax2.set_xlabel('price', fontsize = 15)

#3
ax3 = fig.add_subplot(333)
ax3.scatter(df.price,df.sqft_living)
ax3.set_ylabel('sqft_living', fontsize = 15)
ax3.set_xlabel('price', fontsize = 15)

#4
ax4 = fig.add_subplot(334)
ax4.scatter(df.price,df.sqft_above)
ax4.set_ylabel('sqft_above', fontsize = 15)
ax4.set_xlabel('price', fontsize = 15)

#5
ax5 = fig.add_subplot(335)
ax5.scatter(df.price,df.grade)
ax5.set_ylabel('grade', fontsize = 15)
ax5.set_xlabel('price', fontsize = 15)

#6
ax6 = fig.add_subplot(336)
ax6.scatter(df.price,df.sqft_living15)
ax6.set_ylabel('sqft_living15', fontsize = 15)
ax6.set_xlabel('price', fontsize = 15)

#7
ax7 = fig.add_subplot(337)
ax7.scatter(df.price,df.sqft_basement)
ax7.set_ylabel('sqft_basement', fontsize = 15)
ax7.set_xlabel('price', fontsize = 15)

#8
ax8 = fig.add_subplot(338)
ax8.scatter(df.price,df.waterfront)
ax8.set_ylabel('waterfront', fontsize = 15)
ax8.set_xlabel('price', fontsize = 15)

#9
ax9 = fig.add_subplot(339)
ax9.scatter(df.price,df.view)
ax9.set_ylabel('view', fontsize = 15)
ax9.set_xlabel('price', fontsize = 15)


plt.tight_layout()
fig.savefig('House_price.png')


# In[49]:

fig = plt.figure(figsize = (20,20))

#1

ax1 = fig.add_subplot(331)
ax1.hist(df.bedrooms)
ax1.set_xlabel('bedrooms', fontsize = 15)

#2
ax2 = fig.add_subplot(332)
ax2.hist(df.bathrooms)
ax2.set_xlabel('bathrooms', fontsize = 15)

#3
ax3 = fig.add_subplot(333)
ax3.hist(df.sqft_living)
ax3.set_xlabel('sqft_living', fontsize = 15)

#4
ax4 = fig.add_subplot(334)
ax4.hist(df.sqft_above)
ax4.set_xlabel('sqft_above', fontsize = 15)

#5
ax5 = fig.add_subplot(335)
ax5.hist(df.grade)
ax5.set_xlabel('grade', fontsize = 15)

#6
ax6 = fig.add_subplot(336)
ax6.hist(df.sqft_living15)
ax6.set_xlabel('sqft_living15', fontsize = 15)

#7
ax7 = fig.add_subplot(337)
ax7.hist(df.sqft_basement)
ax7.set_xlabel('sqft_basement', fontsize = 15)

#8
ax8 = fig.add_subplot(338)
ax8.hist(df.waterfront)
ax8.set_xlabel('waterfront', fontsize = 15)

#9
ax9 = fig.add_subplot(339)
ax9.hist(df.view)
ax9.set_xlabel('view', fontsize = 15)


plt.tight_layout()
fig.savefig('House_price_hist.png')


# In[50]:

df['Age'] = 2017 - df.yr_built
df = df.drop('yr_built',1)
df['Basement_or_not'] =  df['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)

df['renovated_or_not'] =  df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

df = df.drop('sqft_basement',1)

df = df.drop('yr_renovated',1)

df = df[(df.bedrooms < 6) & (df.bedrooms > 0)]

df = df[(df.bathrooms < 5) & (df.bathrooms >= 1)]

df['bath_new'] = df['bathrooms']*2

df = df.drop('bathrooms',1)

df = df[(df.sqft_living <= 5800) & (df.sqft_living > 350)]

df = df[df.sqft_above <= 5000]

df = df[df.sqft_living15 <= 5000]

df =df[df.floors <= 3]

df =df[df.price <= 1000000]

df = df[(df.grade > 5) & (df.grade < 12)]

df = df[(df['Age'] < 80) & (df['price'] < 750000 )]


# In[51]:

fig = plt.figure(figsize = (20,20))

#1

ax1 = fig.add_subplot(421)
ax1.hist(df.bedrooms)
ax1.set_xlabel('bedrooms', fontsize = 15)


#2
ax2 = fig.add_subplot(422)
ax2.hist(df.bath_new)
ax2.set_xlabel('bath_new', fontsize = 15)


#3
ax3 = fig.add_subplot(423)
ax3.hist(df.sqft_living)
ax3.set_xlabel('sqft_living', fontsize = 15)


#4
ax4 = fig.add_subplot(424)
ax4.hist(df.sqft_above)
ax4.set_xlabel('sqft_above', fontsize = 15)


#5
ax5 = fig.add_subplot(425)
ax5.hist(df.grade)
ax5.set_xlabel('grade', fontsize = 15)


#6
ax6 = fig.add_subplot(426)
ax6.hist(df.sqft_living15)
ax6.set_xlabel('sqft_living15', fontsize = 15)

#7
ax7 = fig.add_subplot(427)
ax7.hist(df.view)
ax7.set_xlabel('view', fontsize = 15)



plt.tight_layout()


# In[52]:

fig = plt.figure(figsize = (20,20))

#1

ax1 = fig.add_subplot(421)
ax1.scatter(df.price,df.bedrooms)
ax1.set_ylabel('bedrooms', fontsize = 15)
ax1.set_xlabel('price', fontsize = 15)

#2
ax2 = fig.add_subplot(422)
ax2.scatter(df.price,df.bath_new)
ax2.set_ylabel('bathrooms', fontsize = 15)
ax2.set_xlabel('price', fontsize = 15)

#3
ax3 = fig.add_subplot(423)
ax3.scatter(df.price,df.sqft_living)
ax3.set_ylabel('sqft_living', fontsize = 15)
ax3.set_xlabel('price', fontsize = 15)

#4
ax4 = fig.add_subplot(424)
ax4.scatter(df.price,df.sqft_above)
ax4.set_ylabel('sqft_above', fontsize = 15)
ax4.set_xlabel('price', fontsize = 15)

#5
ax5 = fig.add_subplot(425)
ax5.scatter(df.price,df.grade)
ax5.set_ylabel('grade', fontsize = 15)
ax5.set_xlabel('price', fontsize = 15)

#6
ax6 = fig.add_subplot(426)
ax6.scatter(df.price,df.sqft_living15)
ax6.set_ylabel('sqft_living15', fontsize = 15)
ax6.set_xlabel('price', fontsize = 15)

#7
ax7 = fig.add_subplot(427)
ax7.scatter(df.price,df.Basement_or_not)
ax7.set_ylabel('Basement_or_not', fontsize = 15)
ax7.set_xlabel('price', fontsize = 15)

#8
ax8 = fig.add_subplot(428)
ax8.scatter(df.price,df.Age)
ax8.set_ylabel('Age', fontsize = 15)
ax8.set_xlabel('price', fontsize = 15)



plt.tight_layout()


# In[53]:

print(df.zipcode.nunique())
print(df.bedrooms.value_counts())


# In[54]:

df_zipcode = pd.get_dummies(df['zipcode'],drop_first = True)
df = pd.concat([df,df_zipcode],axis =1)
df.describe()


# In[55]:

X = df.iloc[:,1:]
X.head()


# In[56]:

y = df['price']
y.head()


# In[57]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , train_size = 0.8, random_state = 0)


# In[58]:

lassoreg = Lasso()
lassoreg.fit(X_train, y_train)
y_pred_lasso = lassoreg.predict(X_test)


# In[59]:

from sklearn.metrics import mean_squared_error
RMSE_lasso_test = mean_squared_error(y_test,y_pred_lasso)**0.5
print(RMSE_lasso_test)


# In[60]:

from sklearn.metrics import r2_score
R_squared = r2_score(y_test,y_pred_lasso)
R_squared


# In[ ]:



