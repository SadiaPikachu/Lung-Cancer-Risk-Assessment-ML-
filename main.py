import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("C:\\Users\\ACER\\Downloads\\Lung Cancer Dataset.csv")

#The dataset doesn't have any missing values, so I didn't perform any additional steps related to missing data.
# target to 0/1
df['PULMONARY_DISEASE'] = df['PULMONARY_DISEASE'].map({'YES': 1, 'NO': 0})

#features X and target y
X = df.drop('PULMONARY_DISEASE', axis=1)
y = df['PULMONARY_DISEASE']

#Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Random Forest classifier, as shown in related work, has better accuracy than other models.
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

#Evaluate model performance
y_pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save the model
joblib.dump(rf, 'lung_cancer_risk_model.joblib')
print('Model saved to lung_cancer_risk_model.joblib')
