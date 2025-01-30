import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset dari file CSV lokal
file_path = "diabetes_prediction_dataset.csv" 
df = pd.read_csv(file_path)

# Pastikan kolom "Outcome" ada dalam dataset
if "Outcome" not in df.columns:
    raise ValueError("Dataset harus memiliki kolom 'Outcome' sebagai label diabetes.")

# 2. Memisahkan fitur dan label
X = df.drop(columns=["Outcome"])  
y = df["Outcome"]  

# 3. Membagi dataset menjadi data latih dan data uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standarisasi fitur agar SVM bekerja lebih baik
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Membuat dan melatih model SVM
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")  
svm_model.fit(X_train, y_train)

# 6. Prediksi pada data uji
y_pred = svm_model.predict(X_test)

# 7. Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Menampilkan confusion matrix dan classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
