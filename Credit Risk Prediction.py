import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os

df = pd.read_csv("Data.csv")

# Checking for how many categorical and numerical columns are there in the merged dataset
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

print("Categorical columns:", categorical_columns)
print("Numerical columns:", numerical_columns)

df['MARITALSTATUS'].value_counts()


# Chi-square test of independence
for i in range(len(categorical_columns)):
    for j in range(i+1, len(categorical_columns)):
        contingency_table = pd.crosstab(df[categorical_columns[i]], df[categorical_columns[j]])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-square test between {categorical_columns[i]} and {categorical_columns[j]}: p-value = {p}, dof = {dof}")


# VIF sequentially Check
VIF_data = df[numerical_columns]
total_columns = VIF_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range(0, total_columns):
    VIF_value = variance_inflation_factor(VIF_data.values, column_index)
    print(column_index, VIF_value)
    if VIF_value <= 6:
        columns_to_be_kept.append(numerical_columns[i])
        column_index += 1
    else:
        VIF_data = VIF_data.drop([numerical_columns[i]], axis=1)
        
print("Columns to be kept after VIF check:", len(columns_to_be_kept))


# check Anova for columns_to_be_kept 
from scipy.stats import f_oneway

columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])  
    b = list(df['Approved_Flag'])  
    
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']


    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)

print("Columns to be kept after Anova check:", len(columns_to_be_kept_numerical))

# listing all the final features
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]


print(df['MARITALSTATUS'].unique())
print(df['EDUCATION'].unique())
print(df['GENDER'].unique())
print(df['last_prod_enq2'].unique())
print(df['first_prod_enq2'].unique())

mapping = {
    'SSC': 1,
    '12TH': 2,
    'GRADUATE': 3,
    'UNDER GRADUATE': 3,
    'POST-GRADUATE': 4,
    'OTHERS': 1, 
    'PROFESSIONAL': 3
}

df['EDUCATION'] = df['EDUCATION'].map(mapping)


df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER', 'last_prod_enq2' ,'first_prod_enq2'])
df_encoded.info()

# Data Processing
# 1. Random Forest Classifier

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Random Forest Classifier: {accuracy}")
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    
    
# 2. xgboost Classifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis=1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

xgb_classifier.fit(x_train, y_train)
y_pred_xgb = xgb_classifier.predict(x_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Accuracy of the XGBoost Classifier: {accuracy_xgb}")
precision_xgb, recall_xgb, f1_score_xgb, _ = precision_recall_fscore_support(y_test, y_pred_xgb)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision_xgb[i]}")
    print(f"Recall: {recall_xgb[i]}")
    print(f"F1 Score: {f1_score_xgb[i]}")
    
# 3. Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy of the Decision Tree Classifier: {accuracy_dt}")

precision_dt, recall_dt, f1_score_dt, _ = precision_recall_fscore_support(y_test, y_pred_dt)
for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision_dt[i]}")
    print(f"Recall: {recall_dt[i]}")
    print(f"F1 Score: {f1_score_dt[i]}")
    
# Hypertuning for XGBoost Classifier
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'colsample_bytree': [0.5, 0.9],
    'learning_rate': [0.1, 1],
    'max_depth': [3, 8],
    'alpha': [1, 10],
    'n_estimators': [50, 100],
}


index = 0

answers = {
    'combination': [],
    'train_Accuracy': [],
    'test_Accuracy': [],
    'colsample_bytree': [],
    'learning_rate': [],
    'max_depth': [],
    'alpha': [],
    'n_estimators': []
}



for colsample_bytree in param_grid['colsample_bytree']:
    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for learning_rate in param_grid['learning_rate']:
                for alpha in param_grid['alpha']:
                    index += 1
                    xgb_classifier = xgb.XGBClassifier(
                        objective='multi:softmax',
                        num_class=4,
                        colsample_bytree=colsample_bytree,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        alpha=alpha,
                        random_state=42
                    )
                    xgb_classifier.fit(x_train, y_train)
                    train_accuracy = accuracy_score(y_train, xgb_classifier.predict(x_train))
                    test_accuracy = accuracy_score(y_test, xgb_classifier.predict(x_test))

                    answers['combination'].append(index)
                    answers['train_Accuracy'].append(train_accuracy)
                    answers['test_Accuracy'].append(test_accuracy)
                    answers['colsample_bytree'].append(colsample_bytree)
                    answers['learning_rate'].append(learning_rate)
                    answers['max_depth'].append(max_depth)
                    answers['alpha'].append(alpha)
                    answers['n_estimators'].append(n_estimators)

                    '''
                    # Print results for this combination
                    print(f"Combination {index}")
                    print(f"colsample_bytree: {colsample_bytree}, n_estimators: {n_estimators}, max_depth: {max_depth}, learning_rate: {learning_rate}, alpha: {alpha}")
                    print(f"Train Accuracy: {train_accuracy : .2f}")
                    print(f"Test Accuracy: {test_accuracy : .2f}")
                    print("--------------------------------------------------")
                    '''

RESULTS = pd.DataFrame(answers)
RESULTS.to_excel("xgb_hyperparameter_tuning_results.xlsx", index=False)
model = xgb.XGBClassifier(objective='multi:softmax',
                          colsample_bytree=0.9, n_estimators=100, 
                          max_depth=3, learning_rate=1, alpha=10)



