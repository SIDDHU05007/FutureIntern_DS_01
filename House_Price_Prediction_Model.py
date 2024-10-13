#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


# Load the dataset (Boston Housing dataset for demonstration)

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)


# In[14]:


# Import libraries for EDA

import seaborn as sns
import matplotlib.pyplot as plt


# In[16]:


# 1. Basic Dataset Overview
print("Basic Information about Dataset:")
print(data.info())


# In[17]:


print("\nFirst few rows of the dataset:")
print(data.head())


# In[18]:


# 2. Summary Statistics

print("\nSummary Statistics:")
print(data.describe())


# In[19]:


# 3. Check for Missing Values

print("\nChecking Missing Values:")
print(data.isnull().sum())


# In[22]:


# Visualizing missing values (if any)

plt.figure(figsize=(6, 4))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


# In[25]:


# 4. Correlation Matrix and Heatmap

plt.figure(figsize=(8, 4))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[27]:


# 5. Distribution of Target Variable ('medv')

plt.figure(figsize=(8, 4))
sns.histplot(data['medv'], kde=True, color='blue', bins=30)
plt.title('Distribution of Target Variable (House Price)')
plt.xlabel('House Price (in $1000s)')
plt.ylabel('Frequency')
plt.show()


# In[28]:


# 6. Pairplot for Relationships between Features and Target
# Selecting a few important features for pairplots

important_features = ['crim', 'rm', 'lstat', 'medv']
sns.pairplot(data[important_features])
plt.show()


# In[32]:


# 7. Outlier Detection using Boxplots
plt.figure(figsize=(8, 6))
for i, column in enumerate(important_features[:-1], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data[column], color='orange')
    plt.title(f'Boxplot of {column}')

plt.tight_layout()
plt.show()


# In[4]:


# Features (X) and target variable (y)

X = data.drop(columns=['medv'])  # 'medv' is the median value of the house (target)
y = data['medv']


# In[5]:


# Handle missing values (if any)

X.fillna(X.mean(), inplace=True)


# In[6]:


# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Scale numerical features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[8]:


# Initialize the models

lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_reg = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)


# In[9]:


# Train the models

lin_reg.fit(X_train_scaled, y_train)
tree_reg.fit(X_train_scaled, y_train)
rf_reg.fit(X_train_scaled, y_train)
xgb_reg.fit(X_train_scaled, y_train)


# In[10]:


# Make predictions

lin_pred = lin_reg.predict(X_test_scaled)
tree_pred = tree_reg.predict(X_test_scaled)
rf_pred = rf_reg.predict(X_test_scaled)
xgb_pred = xgb_reg.predict(X_test_scaled)


# In[11]:


# Evaluate the models

lin_mse = mean_squared_error(y_test, lin_pred)
tree_mse = mean_squared_error(y_test, tree_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)


# In[12]:


# Display the Mean Squared Errors
print(f'Linear Regression MSE: {lin_mse}')
print(f'Decision Tree Regression MSE: {tree_mse}')
print(f'Random Forest Regression MSE: {rf_mse}')
print(f'XGBoost Regression MSE: {xgb_mse}')


# In[ ]:




