#!/usr/bin/env python
# coding: utf-8

# <center>
# <h1> Linear Regression Case Study
# </center>
# 
# ### Steps for Regression Modeling:
# 1. Business problem definition - One of major automobile company would like to design new product which can improve the sales. Inorder to define the product, they want to understand/identify drivers for the sales (what are the factors driving sales) and Predicting sales of different car models given driving factors. 
# 2. Convert business problem into statistical problem  sales = F(sales attributes, product features, marketing info etc.)
# 3. Finding the right technique - Since it is predicting value (Regression Problem) problem so we can use OLS as one of the technique. We can also use other techniques like Decision Trees, Ensemble learning, KNN, SVM, ANN etc.
# 4. Data colletion(Y, X) - Identify the sources of information and collect the data
# 5. Consolidate the data - aggregate and consolidate the data at Model level/customer level/store level depends on business problem
# 6. Data preparation for modeling (create data audit report to identify the steps to perform as part of data preparation)
#     a. missing value treatment
#     b. outlier treatment
#     c. dummy variable creation
# 7. Variable creation by using transformation and derived variable creation.
# 8. Basic assumptions (Normality, linearity, no outliers, homoscadasticity, no pattern in residuals, no auto correlation etc)
# 9. Variable reduction techniques (removing multicollinerity with the help of FA/PCA, correlation matrics, VIF)
# 10. Create dev and validation data sets (50:50 if you have more data else 70:30 or 80:20)
# 11. Modeling on dev data set (identify significant variables, model interpretation, check the signs and coefficients, multi-collinierity check, measures of good neess fit, final mathematical equation etc)
# 12. validating on validation data set (check the stability of model, scoring, decile analysis, cross validation etc.)
# 13. Output interpretation and derive insights (understand the limitations of the model and define strategy to implementation)
# 14. convert statistical solution into business solutions (implementation, model monitoring etc)
# 

# ### import the packages

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

get_ipython().run_line_magic('matplotlib', 'inline')

statsmodels : can be used only the statistical models
    Linear Regression model : OLS - Ordinary Least Squares
        
    import statsmodels.formula.api as smf
    smf.ols()
    
    
sklearn - scikit learn : used to build every statistical and ML model
    
    from sklearn.Linear_model import LinearRegression
    LinearRegression()
    
    
for train and test split 
    from sklearn.model_selection import train_test_split
    
for the scoring of the models
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# In[11]:


import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# from sklearn.linear_model import LinearRegression
# sklearn - scikit learn


# ### create UDFs

# In[39]:


def continuous_var_summary( x ):
    
    # freq and missings
    n_total = x.shape[0]
    n_miss = x.isna().sum()
    perc_miss = n_miss * 100 / n_total
    
    # outliers - iqr
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lc_iqr = q1 - 1.5 * iqr
    uc_iqr = q3 + 1.5 * iqr
    
    
    return pd.Series( [ x.dtype, x.nunique(), n_total, x.count(), n_miss, perc_miss,
                       x.sum(), x.mean(), x.std(), x.var(), 
                       lc_iqr, uc_iqr, 
                       x.min(), x.quantile(0.01), x.quantile(0.05), x.quantile(0.10), 
                       x.quantile(0.25), x.quantile(0.5), x.quantile(0.75), 
                       x.quantile(0.90), x.quantile(0.95), x.quantile(0.99), x.max() ], 
                     
                    index = ['dtype', 'cardinality', 'n_tot', 'n', 'nmiss', 'perc_miss',
                             'sum', 'mean', 'std', 'var',
                        'lc_iqr', 'uc_iqr',
                        'min', 'p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max']) 


# In[13]:


# Create Data audit Report for categorical variables
def categorical_var_summary( x ):
    Mode = x.value_counts().sort_values(ascending = False)[0:1].reset_index()
    return pd.Series([x.count(), x.isnull().sum(), Mode.iloc[0, 0], Mode.iloc[0, 1], 
                          round(Mode.iloc[0, 1] * 100 / x.count(), 2)], 
                     
                  index = ['N', 'NMISS', 'MODE', 'FREQ', 'PERCENT'])


# In[14]:


# Missing value imputation for continuous variables
def missing_imputation( x, stats = 'mean' ):
    if (x.dtypes == 'float64') | (x.dtypes == 'int64'):
        x = x.fillna(x.mean()) if stats == 'mean' else x.fillna(x.median())
    return x


# In[15]:


# An utility function to create dummy variable
def create_dummies(df, colname):
    col_dummies = pd.get_dummies(df[colname], prefix = colname, drop_first = True)
    df = pd.concat([df, col_dummies], axis = 1)
    df.drop(colname, axis = 1, inplace = True )
    return df


# ### import data

# In[16]:


cars = pd.read_csv('D:/Sampledata/Car_sales.csv')


# ### data inspection

# In[ ]:


.columns
.dtypes
.shape

.info()
.describe()
.head()
.tail()


# In[17]:


cars.columns


# In[22]:


cars.shape


# In[18]:


cars.info()


# In[23]:


cars.head()


# In[24]:


# no of unique values / cardinality
cars.nunique()


# In[12]:


cars.Vehicle_type.value_counts()


# In[13]:


# type conversion in case variables are not of proper type : Not required in this data


# In[27]:


# in case we have huge data and we dont want to make the copy of the original data 
# we generally go with the option of getting column/variable names for obejat and numeric type

numeric_columns = cars.select_dtypes(include = ['float64', 'int64']).columns
object_columns = cars.select_dtypes(include = ['object']).columns


# In[35]:


cars[object_columns].head(2)


# In[29]:


numeric_columns


# In[36]:


cars_conti_vars.head(2)


# In[37]:


# seperate object and numeric variables
cars_conti_vars = cars.loc[:, (cars.dtypes == 'float64') | (cars.dtypes == 'int64')]
cars_cat_vars = cars.loc[:, (cars.dtypes == 'object')]

# Simper way of doing:
# cars_conti_vars = cars.select_dtypes(include = ['float64', 'int64'])
# car_sales_cat = cars.select_dtypes(include = ['object'])


# In[40]:


# alternate of .describe() for continuous variables
cars_conti_vars.apply( continuous_var_summary ).round(1)

# cars_conti_vars.apply( lambda x: continuous_var_summary(x)).round(1)


# In[41]:


# alternate of .describe() for categorical variables
cars_cat_vars.apply(categorical_var_summary).T


# ### outlier treatment

# In[43]:


cars_conti_vars.Price_in_thousands.clip( lower = cars_conti_vars.Price_in_thousands.quantile(0.01), 
                                       upper = cars_conti_vars.Price_in_thousands.quantile(0.99) )


# In[44]:


cars_conti_vars = cars_conti_vars.apply( lambda x: x.clip(lower = x.quantile(0.01),
                                                         upper = x.quantile(0.99)))


# In[45]:


cars_conti_vars.apply(continuous_var_summary).round(1)


# ### missing value treatment

# In[20]:


cars_conti_vars.isna().sum() * 100 / cars_conti_vars.isna().count()


# In[46]:


cars_conti_vars = cars_conti_vars.apply(missing_imputation)


# In[49]:


cars_conti_vars.apply(continuous_var_summary).round(1)


# ## Handling categorical features
# 
# scikit-learn expects all features to be numeric. So how do we include a categorical feature in our model?
# 
# - **Ordered categories:** transform them to sensible numeric values (example: small=1, medium=2, large=3)
# - **Unordered categories:** use dummy encoding (0/1)
# 
# What are the categorical features in our dataset?
# 
# - **Ordered categories:** weather (already encoded with sensible numeric values)
# - **Unordered categories:** season (needs dummy encoding), holiday (already dummy encoded), workingday (already dummy encoded)
# 
# For season, we can't simply leave the encoding as 1 = spring, 2 = summer, 3 = fall, and 4 = winter, because that would imply an **ordered relationship**. Instead, we create **multiple dummy variables:**
# Steps to be followed to create dummy variables:
-----------------------------------------------------------------------------------------------
1. use pd.get_dummies() method to create dummy varaibles for each value inside the variable
2. drop any one variable from the dummy variables to avoid multicolinearity
3. concat the dummy data with the original dataset
4. drop the variable from which dummy variable has been created
# In[51]:


cars_cat_vars.columns


# In[55]:


cars_cat_vars.Vehicle_type.nunique()


# In[62]:


pd.get_dummies( cars_cat_vars.Manufacturer, prefix = 'Manufacturer', drop_first = True ).head(3)


# In[23]:


# get the count of all the categories of the variable
cars_cat_vars.Manufacturer.value_counts()


# In[24]:


cars_cat_vars.Vehicle_type.value_counts()


# In[25]:


pd.get_dummies( cars_cat_vars.Vehicle_type, drop_first = True, prefix = 'Vehicle_type' )


# In[69]:


pd.Series([1, 2, 3, 4, 3, 2, 1, 2,3 , 4, 5, 6, 7, 2, 1, 3]).dtype


# In[72]:


pd.Series([1, 2, 3, 4, 3, 2, 1, 2,3 , 4, 5, 6, 7, 2, 1, 3]).astype('category').dtype


# In[ ]:


# An utility function to create dummy variable
def create_dummies(df, colname):
    
    col_dummies = pd.get_dummies(df[colname], prefix = colname, drop_first = True)
    df = pd.concat([df, col_dummies], axis = 1)
    df.drop(colname, axis = 1, inplace = True )
    return df


# **in case we have categorical featue as numeric type, we have to first convert the variable into 
# category type**

# In[63]:


# get the useful categorical variables
cars_cat_vars = cars.loc[:, ['Manufacturer', 'Vehicle_type']]

# for c_feature in categorical_features
for c_feature in cars_cat_vars.columns:
    
    cars_cat_vars.loc[:, c_feature] = cars_cat_vars[c_feature].astype('category') # you can this step for this example
    cars_cat_vars = create_dummies(cars_cat_vars, c_feature)
    
# see the data in the output
#cars_cat_vars


# In[64]:


cars_cat_vars


# In[28]:


cars_cat_vars.columns


# In[29]:


cars_cat_vars.rename( columns = {'Manufacturer_Mercedes-B' : 'Manufacturer_Mercedes_B'}, inplace = True)


# In[30]:


cars_cat_vars.columns


# ### final data for analysis

# In[31]:


cars_new = pd.concat([cars_conti_vars, cars_cat_vars], axis = 1)


# In[32]:


cars_new.head()


# In[33]:


cars_new.shape


# ### assumptions check

# In[34]:


# Very first assumtion is that all the variables should be normally distributed, however that can't be possible
# However we have to be atleast strict about the dependant Y variable

# Distribution of variables
sns.distplot(cars_new.Sales_in_thousands)
plt.show()
# this distribution is highly skewed

# Notes:
#-----------------------------------------------------
# 1. if we get skewed data, then we have to transform the data and there are multiple methods to go about it
# 2. most commonly used and which works on most of the data is log transformation
# 3. Ideally we can do this for each of the dependant variable as well, 
#    however it will depend on amount of data and the amount of analytical rigour
# 4. In no case we can proceed if dependant variable is not normal/near to normal distributed


# In[35]:


cars_new.Sales_in_thousands.skew()


# In[36]:


# Note: good practice is to take the log of the data plus 1, bcoz we don't have log of zero defined
# In thios data its not required as sales are always greater than zero

# apply log transformation: log is rescalling the data and making the distribution normal
cars_new.loc[:, 'ln_sales_in_thousands'] = np.log(cars_new.loc[:, 'Sales_in_thousands'])

# Distribution of variables
sns.distplot(cars_new.ln_sales_in_thousands)
plt.show()


# In[37]:


cars_new.ln_sales_in_thousands.skew()


# In[38]:


# Linearity: correlation matrix (ranges from 1 to -1)
corrm = cars_new.corr()
corrm.to_excel('corrm.xlsx')
corrm


# In[39]:


# visualize correlation matrix in Seaborn using a heatmap
plt.figure(figsize = (10, 8))
sns.heatmap(cars_new.corr())

# fuel efficiency vs fuel capacity
# Curb weight vs Engine Size

# in case we can't make any concrete decision looking at the variables;
# we can also check on the VAR of the variables into consideration e.g Curb weight vs Wheel base


# In[40]:


# no of variables and obs in the final data to be used for modelling
cars_new.shape


# ### feature selection based on importance using F - Regression

# In[74]:


# Feature Selection based on importance
from sklearn.feature_selection import f_regression


# In[75]:


# splitting the data: separate out the feature/input/independant columns and dependant variable
feature_columns = cars_new.columns.difference(['ln_sales_in_thousands', 'Sales_in_thousands'])
feature_columns


# In[76]:


len(feature_columns)


# In[ ]:


a = 10
b = 20


# In[ ]:


a, b = 10, 20


# In[79]:


# seperate the X and y columns
features = cars_new[feature_columns]
target = cars_new.ln_sales_in_thousands

# do the f_regression
F_values, p_values  = f_regression( features, target )


# In[86]:


pd.concat( [ pd.Series(feature_columns), 
                    pd.Series(F_values).round(2), 
                        pd.Series(p_values).round(4) ], axis = 1 )


# In[85]:


pd.DataFrame([feature_columns, F_values.round(2), p_values.round(4)]).T


# In[87]:


# combine the output in dataframe
F_regression_op = pd.DataFrame([feature_columns, F_values.round(2), p_values.round(4)]).T

# add the column names
F_regression_op.columns = ['Features', 'F_values', 'p_values' ]


# In[88]:


F_regression_op


# In[89]:


# output of the f_regression
feature_columns = list( F_regression_op.loc[ F_regression_op.p_values <= 0.1, 'Features' ] )


# In[90]:


feature_columns


# ####  VIF (Variance Inflation Factor): Check the multicollinieirity for all the variables in the model

# In[94]:


# High VIF of the variable means information in that variable has already been explained by 
# other X variables present in the model

# import the packages for vif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices


# In[92]:


feature_columns


# In[93]:


model_param = 'ln_sales_in_thousands ~ ' + ' + '.join(feature_columns)
model_param


# In[104]:


model_param = '''ln_sales_in_thousands ~ Fuel_efficiency + Length + 
        Manufacturer_Audi + Manufacturer_Ford + Manufacturer_Honda + Manufacturer_Mercedes_B + 
        Manufacturer_Plymouth + Manufacturer_Porsche + Manufacturer_Toyota + Manufacturer_Volvo + 
        Price_in_thousands + Vehicle_type_Passenger'''


# In[105]:


# separate the Y and X variables
y, X = dmatrices( model_param, cars_new, return_type = 'dataframe' )

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()

vif['Features'] = X.columns
vif['VIF Factor'] = [ variance_inflation_factor(X.values, i) for i in range(X.shape[1]) ]

# display the output
vif.round(1)


# In[106]:


# output of the VIF
feature_columns = list( set(vif.loc[:, 'Features']).difference(['Intercept']) )


# In[107]:


feature_columns


# ### split the data for model building

# In[44]:


# feature_columns = cars_new.columns.difference(['Sales_in_thousands', 'ln_sales_in_thousands'])


# In[108]:


# method 1: divide the data into training and testing and separate out Y and X variables
# this will be used in sklearn related functions
train_X, test_X, train_y, test_y = train_test_split(cars_new[feature_columns], 
                        cars_new['ln_sales_in_thousands'], test_size = 0.3, random_state = 12345)


# In[109]:


# method 2: divide the data into training and testing
train, test = train_test_split(cars_new, test_size = 0.3, random_state = 12345)


# In[110]:


# verify the no of obs in training and testing after split
print('No of obs in training: ', len(train), ' | ', 'No of obs in testing: ', len(test))


# ## Form of linear regression
# 
# $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$
# 
# - $y$ is the response
# - $\beta_0$ is the intercept
# - $\beta_1$ is the coefficient for $x_1$ (the first feature)
# - $\beta_n$ is the coefficient for $x_n$ (the nth feature)
# 
# The $\beta$ values are called the **model coefficients**:
# 
# - These values are estimated (or "learned") during the model fitting process using the **least squares criterion**.
# - Specifically, we are find the line (mathematically) which minimizes the **sum of squared residuals** (or "sum of squared errors").
# - And once we've learned these coefficients, we can use the model to predict the response.

# ### building a linear regression model
Steps for model building:
------------------------------------------------------------------
Step 1: lm = smf.ols('y~x1+x2+x3...xn', data = train) # defining Y and X with classs

Step 2: lm.fit()     # building model (estimating the betas)

Step 3: lm.summary() # get the output summary of the model

Step 4: lm.predict(train) # predict the sales on the training data

Step 5: lm.predict(test) # predict the sales on the testing/validation data

Step 6: test the accuracy of the model
    a.  MAPE: Mean Absolute Percentage Error
    b.  RMSE: Root Mean Square Error
    c.  Corelation between actual and predicted
    d.  Decile analysis: for validation of models - Business validation
    
    
R square:
1 - SSE(best fit) / SSE(no slope)

Adjusted R square:
1 - [(1 - R2)(n - 1) / (n - k - 1)]
# ## model 0

# In[48]:


m0_equation = '''ln_sales_in_thousands ~ Vehicle_type_Passenger + 
                Manufacturer_Ford + 
                Fuel_efficiency + 
                Length + Price_in_thousands'''


# In[111]:


feature_columns


# In[112]:


m1_equation = 'ln_sales_in_thousands ~ ' + ' + '.join(feature_columns)


# In[124]:


m1_equation = '''ln_sales_in_thousands ~ Fuel_efficiency + Length + 
            Price_in_thousands + Vehicle_type_Passenger + 
            Manufacturer_Ford'''

m1_equation


# In[125]:


lm0 = smf.ols( m1_equation, train ).fit()


# In[126]:


print(lm0.summary())


# #### Step 4: predict the sales on the training and testing data

# In[54]:


# training
train.loc[:, 'pred_sales'] = np.exp(lm0.predict(train))


# In[55]:


# testing/validation
test.loc[:, 'pred_sales'] = np.exp(lm0.predict(test))


# In[56]:


train[['Sales_in_thousands', 'pred_sales']].mean()


# In[57]:


train[['Sales_in_thousands', 'pred_sales']].head()


# #### MSE - mean squared error

# In[58]:


MSE_train = round(mean_squared_error( train.Sales_in_thousands, train.pred_sales ), 2)
MSE_test = round(mean_squared_error( test.Sales_in_thousands, test.pred_sales ), 2)

RMSE_train = round(np.sqrt( MSE_train ), 2)
RMSE_test = round(np.sqrt( MSE_test ), 2)

print('MSE of training data: ', MSE_train,  ' | ', 'MSE of testing data: ', MSE_test)
print('RMSE of training data: ', RMSE_train,  ' | ', 'RMSE of testing data: ', RMSE_test)


# #### MAE - mean absolute error

# In[59]:


MAE_train = round(mean_absolute_error( train.Sales_in_thousands, train.pred_sales ), 2)
MAE_test = round(mean_absolute_error( test.Sales_in_thousands, test.pred_sales ), 2)

print('MAE of training data: ', MAE_train,  ' | ', 'MAE of testing data: ', MAE_test)


# #### MAPE: Mean Absolute Percentage Error

# In[60]:


# accuracy metrics (a. MAPE: Mean Absolute Percentage Error)
MAPE_train = np.mean(np.abs(train.Sales_in_thousands - train.pred_sales)/train.Sales_in_thousands)
MAPE_test = np.mean(np.abs(test.Sales_in_thousands - test.pred_sales)/test.Sales_in_thousands)

# print the values of MAPE for train and test
print('MAPE of training data: ', MAPE_train,  ' | ', 'MAPE of testing data: ', MAPE_test)


# #### corelations

# In[61]:


sns.heatmap(train[['Sales_in_thousands', 'pred_sales']].corr(), annot = True)
plt.show()
sns.heatmap(test[['Sales_in_thousands', 'pred_sales']].corr(), annot = True)
plt.show()


# In[62]:


sns.scatterplot( train.Sales_in_thousands, train.pred_sales )
plt.show()
sns.scatterplot( test.Sales_in_thousands, test.pred_sales )
plt.show()


# #### Decile Analysis: for validation of models - Business validation

# In[67]:


# create the deciles on train and test data
train.loc[:, 'decile'] = pd.qcut( train.pred_sales, 10, labels = False )
test.loc[:, 'decile'] = pd.qcut( test.pred_sales, 10, labels = False )


# In[ ]:


train.groupby( 'decile' )[['Sales_in_thousands', 
                            'pred_sales']].mean()


# In[65]:


# create the summaries - decile analysis
decile_analysis_train = train.groupby( 'decile' )[['Sales_in_thousands', 
                            'pred_sales']].mean().sort_index( ascending = False).reset_index()
decile_analysis_test = test.groupby( 'decile' )[['Sales_in_thousands', 
                            'pred_sales']].mean().sort_index( ascending = False).reset_index()


# In[66]:


decile_analysis_train


# In[ ]:


decile_analysis_test


# In[ ]:


# write the data into the file
decile_analysis_train.to_csv('Decile_analysis_train.csv')
decile_analysis_test.to_csv('Decile_analysis_test.csv')


# #### validate the poor model performance due of LM assumptions

# In[68]:


lm0.resid


# In[70]:


# assumption: Normality of the residuals/error (using distplot)
sns.distplot(lm0.resid)
plt.show()


# In[71]:


# assumption: mean of residuals/errors is zero
print(lm0.resid.mean())


# In[72]:


# assumption: residuals/errors of the model should not be correlated with dependant (Y) variable
print(stats.stats.pearsonr(lm0.resid, train.ln_sales_in_thousands))


# In[73]:


# assumption: homoscedasticity of residuals/errors
sns.scatterplot(lm0.resid, train.ln_sales_in_thousands)
plt.show()


# #### What can be the possible reasons for poor model performance?
1. Small sample 
2. Assumptions of linear/regression modelling are not true for the model in consideration
2. Influential Observations (check this from QQ plot)
# ### Tips/guidlines for imporvement of model accuracy
Possible reasons for model is not validating (over fitting)
---------------------------------------------------------------------------------------------
1. Data preparation problem (outliers, missings, variable conversions etc. not correct)
2. not included right variables
3. If the data have multicollinerity
4. Including more number of variables 
5. Data size is very low  (ideally we should have 1varaible = 100 obs)
6. The assumptions are not 100% valid
7. The variables are not explaining completely

How to over come this problem?
--------------------------------------------------------------------------------------------
1. Increase the data/sample size
2. Change the variables - Reiterate the model with different combinations of variables
3. Apply right transformations on X variables such the the linear relationship between Y & X will imrpvove
4. Add dervied variables which can explain Y better
5. Re look into data preparation steps
6. Look at the importance of variables include them in the model
7. Change the modelling technique

******************
There are few techniques can help you to identify important variables (Variable selection - Feature selection)
* F-Regression
* RFE (Recursive feature elimination) - Stepwise regression
******************
# ### Other Reading information

# ### Feature selection
# 
# How do we choose which features to include in the model? We're going to use **train/test split** (and eventually **cross-validation**).
# 
# Why not use of **p-values** or **R-squared** for feature selection?
# 
# - Linear models rely upon **a lot of assumptions** (such as the features being independent), and if those assumptions are violated, p-values and R-squared are less reliable. Train/test split relies on fewer assumptions.
# - Features that are unrelated to the response can still have **significant p-values**.
# - Adding features to your model that are unrelated to the response will always **increase the R-squared value**, and adjusted R-squared does not sufficiently account for this.
# - p-values and R-squared are **proxies** for our goal of generalization, whereas train/test split and cross-validation attempt to **directly estimate** how well the model will generalize to out-of-sample data.
# 
# More generally:
# 
# - There are different methodologies that can be used for solving any given data science problem, and this course follows a **machine learning methodology**.
# - This course focuses on **general purpose approaches** that can be applied to any model, rather than model-specific approaches.

# ### Evaluating Model Accuracy
# > R-squared is a statistical measure of how close the data are to the fitted regression line. <br>
# > R-square signifies percentage of variations in the reponse variable that can be explained by the model. <br>
# > - R-squared = Explained variation / Total variation <br>
# > - Total variation is variation of response variable around it's mean. <br>
# 
# > R-squared value varies between 0 and 100%. 0% signifies that the model explains none of the variability, <br>
# while 100% signifies that the model explains all the variability of the response. <br>
# The closer the r-square to 100%, the better is the model. <br>

# ## Other Evaluation metrics for regression problems
# 
# Evaluation metrics for classification problems, such as **accuracy**, are not useful for regression problems. We need evaluation metrics designed for comparing **continuous values**.
# 
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$

# ## Comparing linear regression with other models
# 
# Advantages of linear regression:
# 
# - Simple to explain
# - Highly interpretable
# - Model training and prediction are fast
# - No tuning is required (excluding regularization)
# - Features don't need scaling
# - Can perform well with a small number of observations
# - Well-understood
# 
# Disadvantages of linear regression:
# 
# - Presumes a linear relationship between the features and the response
# - Performance is (generally) not competitive with the best supervised learning methods due to high bias
# - Can't automatically learn feature interactions
