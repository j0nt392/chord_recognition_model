from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

host = ""
database = ""
user = ""
password = ""

conn = psycopg2.connect(
    host=host,
    database=database,
    user=user,
    password=password
)

#features is pitch_sum, labels are chord_names.
features = []
labels = []

cursor = conn.cursor()
query = "SELECT * FROM pitch_sums"
cursor.execute(query)
rows = cursor.fetchall()

for row in rows:
    features.append(row[1])
    labels.append(row[2])

# Encode chord labels to numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)

# Train a machine learning model (Random Forest in this example)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Decode the predicted labels
predicted_labels = label_encoder.inverse_transform(y_pred)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

class_names = label_encoder.classes_

# Confusion matrix for evaluation
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# #dumps the model into a file.
joblib.dump(model, 'chord_identifier.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# conn.commit()
cursor.close()
conn.close()

