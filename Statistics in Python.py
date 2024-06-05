#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries:

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Create Dummy Data

# In[2]:


np.random.random(30)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## In the 'arange' function, we provide the arguments [start], [stop], and [steps] to generate a sequence of numbers.

# In[3]:


# Create a dummy distribution::


data = np.random.normal(loc=5.5, scale=1, size=40000)


# In[4]:


data


# In[5]:


data.shape


# In[6]:


data.mean(axis=0)


# In[7]:


# Create a dataframe:

ddf = pd.DataFrame(data = data)


# In[8]:


ddf


# In[9]:


ddf.describe()


# In[10]:


sns.boxplot(data)


# In[11]:


ddf.quantile(.25)


# In[12]:


ddf.quantile(.75)


# In[13]:


ddf.median()


# In[14]:


ddf.value_counts(normalize=False)


# In[15]:


ddf.mode()


# In[16]:


plt.hist(ddf, bins=20)


# In[17]:


ddf.hist(grid=True, bins=5)


# In[18]:


ddf.boxplot()


# In[19]:


sns.boxplot(data=ddf, orient = 'h')


# In[20]:


sns.boxplot(data=ddf, orient = 'v')


# In[21]:


sns.distplot(ddf)


# In[22]:


ddf.skew()


# In[23]:


ddf.kurtosis()


# In[24]:


tips = sns.load_dataset('tips')


# In[25]:


tips


# In[26]:


corrd = tips.corr()


# In[27]:


corrd


# In[28]:


histogram_total_bill = sns.histplot(data = tips['total_bill'], bins=20)


# In[29]:


tips['total_bill'].skew()


# In[30]:


tips['total_bill'].describe()


# In[31]:


tips['total_bill'].mode()


# In[32]:


distplot = sns.distplot(tips['total_bill'])


# In[33]:


tips['total_bill']


# In[34]:


sns.histplot(tips['total_bill'], cumulative=True, kde=True, stat='density', legend=True)   #Density converts it into percentage % ||


# In[35]:


sns.histplot(tips['total_bill'], cumulative=False, kde=True, stat='density', legend=True)


# In[36]:


tips['total_bill'].describe()


# ### The Bill Amounts is 24.79; STD is 8.9; Mean is 19.7

# In[37]:


# The 1st SD:

19.7 + 8.9


# In[38]:


# 68% of people in the bill range:

19.7 - 8.9


# In[ ]:





# # How would you create CDF & PDF for total bill?

# In[39]:


np.histogram(tips['total_bill'], bins=10)


# In[40]:


count, bins = np.histogram(tips['total_bill'], bins=10)


# In[41]:


count


# In[42]:


bins


# In[43]:


pdf = count/sum(bins)


# In[44]:


pdf


# In[45]:


cdf = np.cumsum(pdf)


# In[46]:


cdf


# In[47]:


# Plotting PDF & CDF:


plt.plot(bins[1:], pdf, color='red', label='PDF')
plt.plot(bins[1:], cdf, label='CDF')
plt.legend()


# # Real Life Simulation

# In[48]:


# 27 MAY 2023


# # We will create dummy data for heights ranging from 4 feet to more than 6 feet, with a mean height that we can control. We will set the mean to 5.5 feet.

# In[49]:


# Most used functions:
    # Random Intiger
    # Random Number(0,1)
    # linspace
    # Arrange

# Use the library: Scipy (Scientific Python); RVS, Norm & Uniform:


# In[63]:


from numpy import random


# In[64]:


data1 = random.normal(loc=5.5, scale=0.5, size=40000)


# In[65]:


data1


# In[66]:


round(data1.mean(), 1)


# In[67]:


round(data1.std(), 3)


# In[68]:


plt.hist(data1, bins=25, density=True)


# In[69]:


# kernel density estimation : KDE
sns.histplot(data1, bins=100, kde=True, cumulative=False, )

# plt.figure.figsize=(80, 80):
sns.set(rc = {'figure.figsize' : (20,8)})


# In[70]:


sns.distplot(data1, kde=True)


# In[72]:


pd1 = pd.DataFrame(data1)


# In[74]:


round(pd1.median(), 1)     # Mean, Median & Mode are the same 


# In[75]:


plt.hist(data1, bins=50, density=False, cumulative=True, color='purple')


# In[76]:


Height_Data = pd.DataFrame(data1)


# In[77]:


Height_Data.rename(columns = {0:'Height'}, inplace=True)


# In[78]:


Height_Data.describe()


# #### According to the theory, 68% of the data will fall within 1 standard deviation (SD) of the mean. This means that if we add 0.5 to the average (mean height of 5.5 feet), the result will be 6 feet. In the opposite direction (to the left of the mean), it will be 5 feet.

# In[79]:


# Within 1sr SD the following number of people's height will fall:


0.68 * 40000


# In[80]:


Height_within_1SD = Height_Data.loc[(Height_Data['Height'] >= 5) & (Height_Data['Height'] <= 6)]


# In[81]:


Height_within_1SD.describe()


# In[82]:


# According to rule, we will get 68% of data within SD 1.

(27184/40000)*100


# In[83]:


# withing the 2nd SD:


Height_within_2SD = Height_Data.loc[(Height_Data['Height'] >= 4.5) & (Height_Data['Height'] <= 6.5)]


# In[84]:


Height_within_2SD.describe()


# In[85]:


# According to rule, we will get 68% of data within SD 2, it's 95% :

(38153/40000)*100


# # Standardized Score for Height Data :: Z-Score

# ### Advanced Diagnostic Tests

# In[86]:


import scipy


# In[87]:


Height_Data['Height_being_Standardized'] = (Height_Data['Height'] - np.mean(Height_Data['Height']))/np.std(Height_Data['Height'])


# In[88]:


Height_Data


# In[89]:


Height_Data.plot(kind='hist')


# In[ ]:




