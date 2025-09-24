# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load the dataset
# Replace `file_path` with your file path in GitHub Codespaces
file_path = "Customer_Churn_Dataset_Final.csv"  # Save the dataset as a CSV file
data = pd.read_csv(file_path)

# Step 2: Data Cleaning
# Remove rows with entirely missing Customer ID
data = data.dropna(subset=['Customer ID'])

# Replace missing numerical values with the mean
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Monthly Charges'] = data['Monthly Charges'].fillna(data['Monthly Charges'].mean())
data['Tenure (Months)'] = data['Tenure (Months)'].fillna(data['Tenure (Months)'].mean())

# Replace missing categorical values with the mode
data['Contract Type'] = data['Contract Type'].fillna(data['Contract Type'].mode()[0])
data['Has Internet Service'] = data['Has Internet Service'].fillna(data['Has Internet Service'].mode()[0])

# Step 3: Feature Engineering
# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
data['Contract Type'] = label_encoder.fit_transform(data['Contract Type'])
data['Has Internet Service'] = label_encoder.fit_transform(data['Has Internet Service'])
data['Churn'] = label_encoder.fit_transform(data['Churn'])  # Target variable

# Standardize numerical features for Naïve Bayes
scaler = StandardScaler()
data[['Age', 'Monthly Charges', 'Tenure (Months)']] = scaler.fit_transform(
    data[['Age', 'Monthly Charges', 'Tenure (Months)']]
)

# Step 4: Data Splitting
# Define features (X) and target (y)
X = data.drop(columns=['Customer ID', 'Churn'])  # Drop irrelevant columns
y = data['Churn']

# Split dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 5: Model Training
# Initialize and train the Naïve Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_model.predict(X_test)
y_proba = nb_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

# Step 6: Model Evaluation
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy, Precision, and Recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# ROC Curve and AUC Score
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()
