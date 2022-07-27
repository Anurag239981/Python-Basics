#!/usr/bin/env python
# coding: utf-8

# #### package import

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import scipy.stats as stats


# #### import the data

# In[3]:


cust = pd.read_csv('D:/Sampledata/cust_seg.csv')


# #### Metadata nd data inspection

# In[4]:


cust.info()


# In[5]:


cust.head()


# In[6]:


cust.nunique()


# In[8]:


cust.columns


# #### Q6 - Corelations
H0 - No relationship
Ha - There is a relationship among the variables

CI - 95%
p - 0.05
# In[9]:


stats.pearsonr( cust.pre_usage, cust.Latest_mon_usage )


# #### Q5 - chi square test
H0 - No relationship
Ha - There is a relationship among the variables

CI - 95%
p - 0.05
# In[12]:


obs_freq = pd.crosstab( cust.region, cust.segment )


# In[14]:


obs_freq


# In[13]:


stats.chi2_contingency( obs_freq )


# #### Q4 - ftest or ANOVA

# In[15]:


cust.segment.nunique()


# In[16]:


cust.segment.value_counts()


# In[30]:


cust.columns


# In[34]:


usage = 'Latest_mon_usage'


# In[35]:


# data processing for the test
s1 = cust.loc[ cust.segment == 1, usage ]
s2 = cust.loc[ cust.segment == 2, usage ]
s3 = cust.loc[ cust.segment == 3, usage ]

print( 'mean s1:', s1.mean(), '| mean s2:', s2.mean(), '| mean s3:', s3.mean() )

H0 - means are from same population
Ha - means are from different population

CI - 95%
p - 0.05
# In[36]:


stats.f_oneway( s1, s2, s3 )

Business conclusion: 
Customers from different segments spend differently on the credit cards
# #### Q1: ttest, 1sample ttest

# In[37]:


cust.Post_usage_1month.mean()

H0 - u <= 50
Ha - u > 50

CI - 95%
p - 0.05
# In[39]:


stats.ttest_1samp( cust.Post_usage_1month, 50 )

Output: We reject the H0
    
Business Conclusion: Spend on the credit card has increased from last year spend of 50
# #### Q2: ttest, paired sample ttest or relational ttest

# In[40]:


print( 'mean of pre usage:', cust.pre_usage.mean() )
print( 'mean of post 1month usage:', cust.Post_usage_1month.mean() )

H0 - u1 <= u2
Ha - u1 > u2

CI - 95%
p - 0.05
# In[42]:


stats.ttest_rel( cust.pre_usage, cust.Post_usage_1month )

Output: We fail to reject the H0
    
Business Conclusion: With the given data, cant be proved that the campaign was sucessful
# #### Q3: ttest, independent sample ttest

# In[45]:


cust.sex.value_counts()


# In[46]:


cust.columns


# In[51]:


usage = 'Post_usage_1month'

male_spend = cust.loc[ cust.sex == 0, usage ]
female_spend = cust.loc[ cust.sex == 1, usage ]

print( 'mean of male spend: ', male_spend.mean(), '| mean of female spend: ', female_spend.mean() )

H0 - u1 = u2
Ha - u1 <> u2

CI - 95%
p - 0.05
# In[52]:


stats.ttest_ind( male_spend, female_spend )


# In[53]:


stats.f_oneway( male_spend, female_spend )


# In[55]:


cust['NewCol'] = np.where( cust.region == 1, 'N', 'O')


# In[56]:


cust


# In[ ]:




