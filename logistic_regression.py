# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization of data and model performance
import seaborn as sns  # For enhanced visualization, especially for confusion matrices
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding categorical variables and scaling features
from sklearn.linear_model import LogisticRegression  # For implementing logistic regression model
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve)  # For model evaluation metrics

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")  # Read the dataset into a Pandas DataFrame

# Data preprocessing
df.drop(['customerID'], axis=1, inplace=True)  # Remove the 'customerID' column as it is not useful for prediction

# Convert 'TotalCharges' to numeric, coercing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values in 'TotalCharges' with the median value
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns  # Identify categorical columns
le = LabelEncoder()  # Initialize the label encoder

# Loop through each categorical column and apply label encoding
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])  # Convert categorical values to numeric codes

# Split features and target variable
X = df.drop(columns=['Churn'])  # Feature matrix excluding the target column 'Churn'
y = df['Churn']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Split data into 80% training, 20% testing
)

# Scale numeric features
scaler = StandardScaler()  # Initialize standard scaler for feature scaling
X_train = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test = scaler.transform(X_test)  # Transform the test data using the same scaling

# Train Logistic Regression model
lr_model = LogisticRegression()  # Initialize logistic regression model
lr_model.fit(X_train, y_train)  # Train the model using training data

# Make predictions
y_pred = lr_model.predict(X_test)  # Predict the target variable for test data

# Evaluate the model
print("Logistic Regression Performance:")  # Print evaluation header
print(classification_report(y_test, y_pred))  # Print precision, recall, f1-score for each class
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")  # Print the overall model accuracy

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)  # Generate confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')  # Visualize confusion matrix with annotations
plt.title("Logistic Regression Confusion Matrix")  # Add title to the plot
plt.xlabel("Predicted")  # Label the x-axis
plt.ylabel("Actual")  # Label the y-axis
plt.show()  # Display the confusion matrix plot

# ROC Curve
y_prob = lr_model.predict_proba(X_test)[:, 1]  # Get probability scores for the positive class
fpr, tpr, _ = roc_curve(y_test, y_prob)  # Compute false positive rate and true positive rate for ROC curve
roc_auc = roc_auc_score(y_test, y_prob)  # Calculate AUC (Area Under the Curve)

# Plot ROC curve
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')  # Plot ROC curve with AUC score
plt.plot([0, 1], [0, 1], 'k--')  # Plot the diagonal line for random guessing
plt.xlabel('False Positive Rate')  # Label x-axis
plt.ylabel('True Positive Rate')  # Label y-axis
plt.title('ROC Curve')  # Add title to the ROC curve plot
plt.legend()  # Add legend to identify the curve
plt.show()  # Display the ROC curve

# Get feature importance (absolute values of coefficients)
feature_importance_lr = abs(lr_model.coef_[0])
feature_names_lr = X.columns

# Visualize feature importance for Logistic Regression
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_lr, y=feature_names_lr)
plt.title("Feature Importance in Logistic Regression Model")
plt.xlabel("Importance Score (Coefficient Magnitude)")
plt.ylabel("Feature")
plt.show()
