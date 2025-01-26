import pandas as pd  # Importing pandas for data manipulation
import numpy as np  # Importing numpy for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
import seaborn as sns  # Importing seaborn for advanced data visualization
from sklearn.model_selection import train_test_split  # Importing function to split data
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Importing preprocessing tools
from sklearn.ensemble import RandomForestClassifier  # Importing Random Forest model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve  # Importing evaluation metrics
from sklearn.model_selection import GridSearchCV  # Importing grid search for hyperparameter tuning

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")  # Reading dataset from CSV file

# Data preprocessing
df.drop(['customerID'], axis=1, inplace=True)  # Dropping 'customerID' column as it's not useful

# Convert 'TotalCharges' to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Convert to numeric, handle errors
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())  # Fill missing values with median

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns  # Selecting categorical columns
le = LabelEncoder()  # Initializing label encoder
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])  # Encoding categorical features

# Split features and target variable
X = df.drop(columns=['Churn'])  # Defining feature set
y = df['Churn']  # Defining target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Splitting data

# Scale numeric features
scaler = StandardScaler()  # Initializing scaler
X_train = scaler.fit_transform(X_train)  # Scaling training data
X_test = scaler.transform(X_test)  # Scaling test data

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initializing Random Forest
rf_model.fit(X_train, y_train)  # Training model

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, None],      # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split
    'min_samples_leaf': [1, 2, 4]     # Minimum number of samples per leaf
}

# Initialize Random Forest model for grid search
rf = RandomForestClassifier(random_state=42)

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1)  # Hyperparameter tuning
grid_search.fit(X_train, y_train)  # Fitting model with best parameters

# Print the best parameters
print("Best Hyperparameters for Random Forest:", grid_search.best_params_)

# Train Random Forest with the best parameters
best_rf_model = grid_search.best_estimator_  # Getting the best model
best_rf_model.fit(X_train, y_train)  # Training the best model

# Evaluate the tuned model
y_pred_tuned = best_rf_model.predict(X_test)  # Making predictions
print("Tuned Random Forest Performance:")
print(classification_report(y_test, y_pred_tuned))  # Displaying performance metrics

# Updated Accuracy
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)  # Calculating accuracy
print(f"Tuned Random Forest Accuracy: {accuracy_tuned:.2f}")

# Evaluate the initial Random Forest model
y_pred = rf_model.predict(X_test)  # Making predictions
print("Random Forest Performance:")
print(classification_report(y_test, y_pred))  # Displaying performance metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)  # Creating confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')  # Visualizing confusion matrix
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
y_prob = rf_model.predict_proba(X_test)[:, 1]  # Getting probabilities for ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)  # Calculating ROC values
roc_auc = roc_auc_score(y_test, y_prob)  # Calculating AUC score

plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')  # Plotting ROC curve
plt.plot([0, 1], [0, 1], 'k--')  # Plotting diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Feature importance analysis
feature_importance = best_rf_model.feature_importances_  # Extracting feature importance
feature_names = X.columns  # Getting feature names

plt.figure(figsize=(10, 6))  # Setting figure size
sns.barplot(x=feature_importance, y=feature_names)  # Plotting feature importance
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
