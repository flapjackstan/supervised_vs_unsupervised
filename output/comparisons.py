#!/usr/bin/env python
# coding: utf-8

# # Project 2
# 
# **this is not a group project**
# 
# This project will focus on comparing and contrasting the supervised and unsupervised algorithms we have learned so far. **Clearly label each section and number in the notebook**
# 

# 
# ## Part I
# 
# Use the dataset *burgersOrPizza.csv* to build 3 models that predict whether a food is a burger or pizza (you can use any of the predictive models we've learned).
# 
# For each model:
# 
# 0. Explore data (with ggplot)
# 1. Explain which variables you're using to predict the outcome. 
# 2. Explain which model validation technique you're using and why. 
# 3. Explain why you did or did not choose to standardize your continuous variables.
# 4. Evaluate how the model performed. Explain.
# 
# At the end:
# 
# 5. Compare the performance of the 3 models using the accuracy, and the confusion matrix (consider things like how many it got correct, which errors it w

# In[49]:


#https://docs.bamboolib.8080labs.com/documentation/how-tos/installation-and-setup/install-bamboolib
#https://docs.bamboolib.8080labs.com/documentation/how-tos/installation-and-setup/install-bamboolib/test-bamboolib
#https://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column

import yaml
#import bamboolib as bam
import pandas as pd
import numpy as np
import warnings
from plotnine import *
#import matplotlib as plt
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler #Z-score variables
from sklearn.preprocessing import LabelEncoder 

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

from sklearn.model_selection import train_test_split # simple TT split cv
from sklearn.linear_model import LogisticRegression # Logistic Regression Model
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve,roc_auc_score #model eval

get_ipython().run_line_magic('matplotlib', 'inline')
#plt.rcParams['figure.figsize'] = (8,25)
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 500)


# In[2]:


foods = pd.read_csv('data/burgersOrPizza.csv')
foods.head()


# In[3]:


foods_string = foods.select_dtypes('object')


# In[4]:


conditions = [(foods_string['Food_Category'] == 'Burgers'),
              (foods_string['Food_Category'] == 'Pizza'),
             ((foods_string['Food_Category'] != 'Pizza').all() or (foods_string['Food_Category'] !='Burgers').all())]
choices = [0,1,2]

foods_string['binary'] = np.select(conditions, choices, default=2)


# In[5]:


foods_string['Item_Description'].replace(',','', regex=True, inplace=True)
foods_string['Item_Description'] = foods_string['Item_Description'].str.lower()


# In[6]:


burger = foods_string[foods_string['binary'] == 0]
words = burger.Item_Description.sample(n = 20).head(20).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).sort_values(ascending=[False])
words = pd.DataFrame(words, columns=['sum'])
words.reset_index(inplace=True)


# In[7]:


(ggplot(words.head(10),aes(x = 'index', y= 'sum'))+geom_bar(stat='identity'))


# In[8]:


pizza = foods_string[foods_string['binary'] == 1]
words = pizza.Item_Description.sample(n = 20).head(20).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).sort_values(ascending=[False])
words = pd.DataFrame(words, columns=['sum'])
words.reset_index(inplace=True)


# In[9]:


(ggplot(words.head(10),aes(x = 'index', y= 'sum'))+geom_bar(stat='identity'))


# In[10]:


with open("data/words.yaml") as file:
        words = yaml.load(file, Loader=yaml.FullLoader)

pizza = words['pizza']
burger = words['burger']


# In[11]:


foods_string['pizza_words'] = np.nan
foods_string['burger_words'] = np.nan


# In[12]:


for i in range(0,len(foods_string.Item_Description)):
    burger_freq = []
    pizza_freq = []
    for w in foods_string.Item_Description[i].split(" "):
        burger_freq.append(burger.count(w))
        pizza_freq.append(pizza.count(w))
    
    
    foods_string.burger_words.loc[i] = sum(burger_freq)
    foods_string.pizza_words.loc[i] = sum(pizza_freq)


# In[13]:


foods_string.sample(n = 15).head(15)


# ### Logistic Regression
# 
# I am using the created common words count variables as the predictor variables because the decription of the food is much more telling of what a food is than any nutitional information. Also if the data is there, might as well use it. I used a traditional train test split for this model because it is the easiest to implement and think other methods are overkill in this situation. I did not standardize any variables because both variables are on the same scale. The model did very well given the accuracy score of .94 meaning that the model correctly classified 94% of the data. False positives rate was also very high meaning that the outcome predicted is trustable and precise such that that 99% of the times the model predicts a positive (pizza) outcome it is correct. Sensitivity score is fairly high (93%) suggesting that the model will falsly classify 7% of pizzas as burgers.

# In[14]:


predictors = {'pizza_words','burger_words'}
X_train, X_test, y_train, y_test = train_test_split(foods_string[predictors], foods_string["binary"], test_size=0.4)


# In[15]:


model = LogisticRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)


# In[16]:


foods_string.reset_index(inplace=True)
test_copy = X_test.copy()
test_copy.reset_index(inplace=True)

test_copy = pd.merge(test_copy, foods_string[['index','Item_Description','binary']], on='index', how='left')
test_copy['log_preds'] = test_pred


# In[17]:


print("Accuracy - TP+TN/TP+FP+FN+TN:",accuracy_score(y_test, test_pred))
print("Precision - TP/TP+FP:",precision_score(y_test, test_pred)) #relates to low false positivity
print("Recall/Sensitivity/TPR - TP/TP+FN:",recall_score(y_test, test_pred))


# ### Decision Tree
# 
# I am using the created common words count variables as the predictor variables because the decription of the food is much more telling of what a food is than any nutitional information. Also if the data is there, might as well use it. I used a traditional train test split for this model because it is the easiest to implement and think other methods are overkill in this situation. I did not standardize any variables because both variables are on the same scale. The model did very well given the accuracy score of .92 meaning that the model correctly classified 92% of the data. False positives rate was also high meaning that the outcome predicted is trustable and precise such that that 89% of the times the model predicts a positive (pizza) outcome it is correct. Sensitivity score is suspiciously high (100%) suggesting that the model will default to pizza.

# In[18]:


tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)


# In[19]:


test_copy['dt_preds'] = y_pred


# In[20]:


print("Accuracy - TP+TN/TP+FP+FN+TN:",accuracy_score(y_test, y_pred))
print("Precision - TP/TP+FP:",precision_score(y_test, y_pred)) #relates to low false positivity
print("Recall/Sensitivity/TPR - TP/TP+FN:",recall_score(y_test, y_pred))


# ### Naive-Bayes
# 
# I am using the created common words count variables as the predictor variables because the decription of the food is much more telling of what a food is than any nutitional information. Also if the data is there, might as well use it. I used a traditional train test split for this model because it is the easiest to implement and think other methods are overkill in this situation. I did not standardize any variables because both variables are on the same scale. The model did very well given the accuracy score of .91 meaning that the model correctly classified 91% of the data. False positives rate was also high meaning that the outcome predicted is trustable and precise such that that 90% of the times the model predicts a positive (pizza) outcome it is correct. Sensitivity score is suspiciously high (99%) suggesting that the model will most likely default to pizza.

# In[21]:


nb = GaussianNB()

nb.fit(X_train,y_train)

y_pred = nb.predict(X_test)


# In[22]:


test_copy['nb_preds'] = y_pred


# In[23]:


print("Accuracy - TP+TN/TP+FP+FN+TN:",accuracy_score(y_test, y_pred))
print("Precision - TP/TP+FP:",precision_score(y_test, y_pred)) #relates to low false positivity
print("Recall/Sensitivity/TPR - TP/TP+FN:",recall_score(y_test, y_pred))


# ### Comparison
# 
# 
# Looking at all the models performances in accuracy, you can clearly see the small differences in each model. Both the logistic and decision tree models have high true positives as indicated in the accuracy score as well as 0 false positives. However it is interesting to note that the naive bayes model has a significantly lower false positives meaning that this model is bias and is likely to predict pizza more than burger. In a case where outcome is more important to predict a positive, it might be better to use this model, but since were predicting pizza and burgers id say this is the worse models. Differences in the models can also be seen actually inspecting the output and it is clear that the models have a tough time when word counts are equal. This can be easily solved by adding more descriptive words in the ingredient list such as mayo and triple since these words are more used in describing burgers than pizza.

# In[24]:


cnf_matrix = confusion_matrix(test_copy['binary'], test_copy['log_preds'])
cnf_matrix


# In[25]:


cnf_matrix = confusion_matrix(test_copy['binary'], test_copy['dt_preds'])
cnf_matrix


# In[26]:


cnf_matrix = confusion_matrix(test_copy['binary'], test_copy['nb_preds'])
cnf_matrix


# In[27]:


test_copy[(test_copy['binary'] != test_copy['log_preds']).all() or 
          (test_copy['binary'] != test_copy['dt_preds']).all() or
          (test_copy['binary'] != test_copy['nb_preds'])]


# ## Part II
# 
# Use the dataset *KrispyKreme.csv* to build 2 clustering models (you can use any of the clustering models we've learned). 
# 
# For each model:
# 
# 0. Explore data (with ggplot)
# 1. Explain which variables you're using to predict the outcome. 
# 2. Evaluate how the model performed using sihouette scores. Look at different numbers of cluseters (like k = 3,5..etc). Which number of clusters is the best fit?
# 3. Describe the clusters (what are they like? how are they different)
# 
# At the end: 
# 
# 4. Compare the clusters obtained by the two models. Overall are they similar? or really different (i.e. do they contain mostly the same members?)
# 
# 
# **Please get rid of extra analyses/superfluous code before turning it in**. Turn in A PDF on Blackboard.

# In[28]:


donuts = pd.read_csv('data/KrispyKreme.csv')
donuts.head(10)


# In[29]:


(ggplot(donuts, aes(x = 'Carbohydrates', y= 'Sugar'))+geom_point())


# In[30]:


(ggplot(donuts,aes(x = 'Food_Category', y= 'Calories'))+geom_bar(stat='identity'))


# In[31]:


donuts[donuts['Food_Category'] == 'Desserts'].head(10)


# In[32]:


donuts.drop(donuts.loc[donuts['Food_Category']=='Desserts'].index, inplace=True)


# In[33]:


(ggplot(donuts,aes(x = 'Food_Category', y= 'Calories'))+geom_bar(stat='identity'))


# In[34]:


le = LabelEncoder() 
  
donuts['Food_Label']= le.fit_transform(donuts['Food_Category']) 


# In[35]:


# for an approximate result, multiply the mass value by 28.35
mask = donuts['Serving_Size_Unit'] == 'oz'
donuts.loc[mask, 'Serving_Size'] = donuts.loc[mask, 'Serving_Size'].apply(lambda oz: oz * 28.38)


# In[36]:


vars = ['Item_Name', 'Item_Description', 'Food_Label', 'Serving_Size', 'Total_Fat', 'Carbohydrates', 'Protein']
features = ['Food_Label', 'Serving_Size', 'Total_Fat', 'Carbohydrates', 'Protein']


# In[37]:


donuts = donuts[vars]


# In[38]:


X = donuts[features]


# In[39]:


z = StandardScaler()
X[features] = z.fit_transform(X)


# ### Model
# 
# Im using the scaled serving size, label category, and macro nutrients for each items as my variables because every other variable can be derived from these main variables for example sugar at the end of the day is still accounted for in carbs, and same goes for saturated fat being a type of fat. Overall, models had better sihouette scores around the 7th cluster with the highest being in the hierarchy model. I decided to compare clusters using only 3 cluster groups because its a bit easier to make sense of the data as such.

# In[40]:


n_components = [2,3,4,5,6,7]
Xdf = X


# ### KM

# In[41]:


sils = []

for n in n_components:
    km = KMeans(n_clusters = n)
    km.fit(X)
    colName = str(n) + "KM_assign"
    clusters = km.predict(X)
    
    Xdf[colName] = clusters
    
    sils.append(silhouette_score(X, clusters))
    
print(sils)


# ### GM

# In[42]:


sils = []

for n in n_components:
    gmm = GaussianMixture(n_components = n)
    gmm.fit(X)
    colName = str(n) + "GM_assign"
    clusters = gmm.predict(X)
    
    Xdf[colName] = clusters
    
    sils.append(silhouette_score(X, clusters))
    
print(sils)


# ### HAC

# In[43]:


sils = []

for n in n_components:
    hac = AgglomerativeClustering(n_clusters = n,
                              affinity = "euclidean",
                             linkage = "ward")
    hac.fit(X)
    
    colName = str(n) + "HAC_assign"
    clusters = hac.labels_
    
    Xdf[colName] = clusters
    
    sils.append(silhouette_score(X, clusters))
    
print(sils)


# In[44]:


groups = ['3GM_assign', '3KM_assign', '3HAC_assign', 'index']


# In[45]:


donuts.reset_index(inplace = True)
Xdf.reset_index(inplace = True)


# In[46]:


donuts = pd.merge(donuts,Xdf[groups], on='index')


# In[47]:


dendro = sch.dendrogram(sch.linkage(donuts[features], metric = "cosine", method='average'))


# ### Analysis
# 
# Overall all models did well distinguishing what was a drink and what was a baked good item most likely from the drastic jump in serving size. Think about this now I think it would have been more interesting to see if it would make the distinction from the macronutrients. I think it would because of the jump in protein between both foods. Although I wanted this model to distinguish between healthy vs non healthy items, i got more of a seperation between drinks with skim milk and "skinny" options vs drinks with added confections.

# In[60]:


donuts = donuts.sort_values(by=['Carbohydrates'], ascending=[False])
print(donuts[donuts['Food_Label'] == 1].describe())
donuts[donuts['Food_Label'] == 1].head(60)


# In[56]:


donuts = donuts.sort_values(by=['Total_Fat'], ascending=[False])
print(donuts[donuts['Food_Label'] == 0].describe())
donuts[donuts['Food_Label'] == 0]


# In[ ]:




get_ipython().system("jupyter nbconvert --output-dir='output/' --to pdf comparisons.ipynb")
get_ipython().system("jupyter nbconvert --output-dir='output/' --to markdown comparisons.ipynb")
get_ipython().system("jupyter nbconvert --output-dir='output/' --to html comparisons.ipynb")
get_ipython().system("jupyter nbconvert --output-dir='output/' --to python comparisons.ipynb")

