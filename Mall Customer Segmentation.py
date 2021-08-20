#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing essential libraries
import numpy as np
import pandas as pd


# In[2]:


# Loading the dataset
df = pd.read_csv("Mall_Customers.csv")


# # **Exploring the dataset**

# In[3]:


# Returns number of rows and columns of the dataset
df.shape


# In[4]:


# Returns an object with all of the column headers 
df.columns


# In[5]:


# Returns different datatypes for each columns (float, int, string, bool, etc.)
df.dtypes


# In[6]:


# Returns the first x number of rows when head(x). Without a number it returns 5
df.head()


# In[7]:


# Returns the last x number of rows when tail(x). Without a number it returns 5
df.tail()


# In[8]:


# Returns basic information on all columns
df.info()


# In[9]:


# Returns basic statistics on numeric columns
df.describe().T


# In[10]:


# Returns true for a column having null values, else false
df.isnull().any()


# # **Data Cleaning**

# In[11]:


# Creating the copy of dataset
df_copy = df.copy(deep=True)


# In[12]:


df_copy.head(3)


# In[13]:


# Dropping the column of 'CustomerID' as it does not provide any value
df_copy.drop('CustomerID', axis=1, inplace=True)
df_copy.columns


# # **Data Visualization**

# In[14]:


# Loading essential libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


df_copy.columns


# ## Gender Plot

# In[16]:


# Visualising the columns 'Gender' using Countplot
sns.countplot(x='Gender', data=df_copy)
plt.xlabel('Gender')
plt.ylabel('Count')


# **Gender plot - Observation**
# 
# *From the Count plot it is observed that the number of Female customers are more that the total number of Male customers.*

# ## Age Plot

# In[17]:


# Visualising the columns 'Age' using Histogram
plt.hist(x=df_copy['Age'], bins=10, orientation='vertical', color='red')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# **Age plot - Observation**
# 
# *From the Histogram it is evident that there are 3 age groups that are more frequently shop at the mall, they are: 15-22 years, 30-40 years and 45-50 years.*

# ## Age Vs Spending Score

# In[18]:


# Visualising the columns 'Age', 'Spending Score (1-100)' using Scatterplot and Jointplot
sns.scatterplot(data=df_copy, x='Age', y='Spending Score (1-100)', hue='Gender')
sns.jointplot(data=df_copy, x='Age', y='Spending Score (1-100)')


# **Age Vs Spending Score - Observation**
# 
# *1. From the Age Vs Spending Score plot we observe that customers whose spending score is more than 65 have their Age in the range of 15-42 years. Also from the Scatter plot it is observed that customers whose spending score is more than 65 consists of more Females than Males.*
# 
# *2. Also, the customers having average spending score ie: in the range of 40-60 consists of age group of the range 15-75 years and the count of Male and Female in this age group is also approximatly the same.*
# 

# ## Annual Income Vs Spending Score

# In[19]:


# Visualising the columns 'Annual Income (k$)', 'Spending Score (1-100)' using Scatterplot and Jointplot
sns.scatterplot(data=df_copy, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
sns.jointplot(data=df_copy, x='Annual Income (k$)', y='Spending Score (1-100)')


# **Annual Income Vs Spending Score - Observation**
# 
# *From the Annual Income Vs Spending Score plot we observe that there are 5 clusters and can be categorised as:*
# 
# ---
# 
# *a. High Income, High Spending Score (Top Right Cluster)*
# 
# *b. High Income, Low Spending Score (Bottom Right Cluster)*
# 
# *c. Average Income, Average Spending Score (Center Cluster)*
# 
# *d. Low Income, High Spending Score (Top Left Cluster)*
# 
# *e. Low Income, Low Spending Score (Bottom Left Cluster)*

# # **Data Preprocessing**

# In[20]:


df_copy.head()


# In[21]:


df_copy.min()


# In[22]:


df_copy["Annual Income (k$)"]=df_copy["Annual Income (k$)"]/(137-15)


# In[23]:


df_copy["Spending Score (1-100)"]=df_copy["Spending Score (1-100)"]/(99-1)


# In[24]:


df_copy.head()


# In[25]:


X = df_copy.iloc[:, [2,3]]


# In[26]:


X


# ## Finding optimal number of clusters using Elbow Method

# In[27]:


# Calculating WCSS values for 1 to 10 clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
  kmeans_model = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans_model.fit(X)
  wcss.append(kmeans_model.inertia_)


# In[28]:


# Plotting the WCSS values
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# *From the above plot it is observed that **5 clusters** are optimal for the given dataset.*

# ## Feature Scaling

# In[29]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# *Feature Scaling is performed because KMeans uses Distance (Euclidean, Manhattan, etc.) and the model perfoms faster on scaling the values*

# # **Model Building**

# In[30]:


# Training the KMeans model with n_clusters=5
kmeans_model = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans_model.fit_predict(X)


# In[31]:


# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 30, c = 'yellow', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 30, c = 'cyan', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 30, c = 'lightgreen', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 30, c = 'orange', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 30, c = 'red', label = 'Cluster 5')
plt.scatter(x=kmeans_model.cluster_centers_[:, 0], y=kmeans_model.cluster_centers_[:, 1], s=100, c='black', marker='+', label='Cluster Centers')
plt.legend()
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# **Clustering - Observation**
# 
# a. High Income, High Spending Score (Cluster 5) - *Target these customers by sending new product alerts which would lead to increase in the revenue collected by the mall as they are loyal customers.*
# 
# *b. High Income, Low Spending Score (Cluster 3) - Target these customers by asking the feedback and advertising the product in a better way to convert them into Cluster 5 customers.*
# 
# c. Average Income, Average Spending Score (Cluster 2) - *Can target these set of customers by providing them with Low cost EMI's etc.*
# 
# d. Low Income, High Spending Score (Cluster 1) - *May or may not target these group of customers based on the policy of the mall.*
# 
# e. Low Income, Low Spending Score (Cluster 4) - *Don't target these customers since they have less income and need to save money.*
