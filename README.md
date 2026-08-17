# Credit Risk Modelling / Prediction using Machine Learning technique in Python (Work in Progress)

This project builds a multi-class credit risk prediction model using Python.  
The goal is to classify customers into different risk categories (P1, P2, P3, P4) based on some external factors.

---

## Main Workflow

### 1. Data Loading
- Import dataset using Pandas
- Cleaning Dataset 
- Identify categorical and numerical columns

### 2. Statistical Feature Selection
- **Chi-Square Test:** for categorical variable dependency
- **Variance Inflation Factor (VIF):** to remove multicollinearity
- **ANOVA Test:** to select numerical features with significant variance across classes

### 3. Data Preprocessing
- Mapping education levels
- One-hot encoding categorical variables
- Final feature selection

### 4. Machine Learning Models
Models implemented:
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Decision Tree Classifier**

Metrics evaluated:
- Accuracy
- Precision
- Recall
- F1 Score (per class)

### 5. Hyperparameter Tuning (XGBoost)
A custom grid search loop is used to evaluate combinations of:
- colsample_bytree  
- learning_rate  
- max_depth  
- alpha  
- n_estimators  

Results (train/test accuracy) are exported in a separate file.

