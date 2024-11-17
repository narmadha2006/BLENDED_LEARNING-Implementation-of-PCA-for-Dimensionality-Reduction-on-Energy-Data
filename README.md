# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1. Data Loading
- Load the dataset into a DataFrame from a CSV file.
#### 2. Data Exploration
- View the first few rows and check column names to ensure correct data loading.
#### 3. Data Preprocessing (Feature Scaling)
- Standardize the features (height and weight) to normalize them for PCA.
#### 4. Applying PCA for Dimensionality Reduction
- Reduce the dataset to 2 principal components to capture most of the variance.
#### 5. Explained Variance Analysis
- Calculate the explained variance ratio to understand how much variance is captured by each principal component.
#### 6. Visualization of Principal Components
- Plot the reduced dataset (principal components) to visualize the distribution and structure.



## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: Narmadha S
RegisterNumber:  212223220065
*/


# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset from a local file
# Ensure the correct file path is provided where the dataset is saved
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/HeightsWeights.csv"
data = pd.read_csv(url)

# Step 2: Explore the data
# Display the first few rows and column names for initial inspection
print(data.head())
print(data.columns)

# Step 3: Preprocess the data (Feature Scaling)
# Select the relevant columns for analysis
X = data[['Height(Inches)', 'Weight(Pounds)']]  # Use the appropriate column names

# Standardize the features to bring them to the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA for dimensionality reduction
# Initialize PCA to reduce the features to 2 components (for simplicity)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Analyze the explained variance
# Print the explained variance ratio for each principal component
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio for each Principal Component:", explained_variance)
print("Total Explained Variance:", sum(explained_variance))

# Step 6: Visualize the principal components
# Create a DataFrame to store the principal components
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Plot the first two principal components
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Heights and Weights Dataset")
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/ac237989-c15b-4a47-b3ff-8aca5b176348)
![image](https://github.com/user-attachments/assets/e1f53942-fa1f-4bd2-9191-379b93045b26)




## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
