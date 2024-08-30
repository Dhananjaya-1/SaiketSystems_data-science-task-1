#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
data = pd.read_csv('data1')
data.head()  # First few rows
data.info()  # Data types and missing values
data.describe()  # Summary statistics


# In[5]:


data.dropna()  # Remove missing values
data.fillna(5)  # Fill missing values with a specified value


# In[6]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Assuming 'data' is your DataFrame

# Step 1: Separate numeric and non-numeric columns
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
non_numeric_cols = data.select_dtypes(exclude=['float64', 'int64']).columns

# Step 2: Scale the numeric columns
scaler = StandardScaler()
data_scaled_numeric = scaler.fit_transform(data[numeric_cols])

# Convert the scaled data back to a DataFrame
data_scaled_numeric = pd.DataFrame(data_scaled_numeric, columns=numeric_cols)

# Step 3: Combine scaled numeric data with the non-numeric data
data_final = pd.concat([data_scaled_numeric, data[non_numeric_cols]], axis=1)

# 'data_final' now contains your scaled numeric data and original non-numeric data


# In[7]:


# One-Hot Encoding
data_encoded = pd.get_dummies(data, columns=['SeniorCitizen'])

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['PaymentMethod'] = le.fit_transform(data['SeniorCitizen'])


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming 'data' is your DataFrame and 'target_column' is the column you want to predict

# Step 1: Define X (features) and y (target)
X = data.drop(columns='Partner')  # Replace 'target_column' with the actual column name
y = data['SeniorCitizen']  # The column you want to predict

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you can proceed with training your model using X_train, y_train


# In[11]:


print(data)

