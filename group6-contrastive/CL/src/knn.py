import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import normalize   # 👈 ADD THIS

# Load embeddings
X_train = np.load("../outputs/X_train_embeddings.npy")
X_val   = np.load("../outputs/X_val_embeddings.npy")
X_test  = np.load("../outputs/X_test_embeddings.npy")

y_train = np.load("../outputs/y_train.npy")
y_val   = np.load("../outputs/y_val.npy")
y_test  = np.load("../outputs/y_test.npy")

print("Data loaded.")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

X_train = normalize(X_train)
X_test  = normalize(X_test)

# Train k-NN
knn = KNeighborsClassifier(
    n_neighbors=10,
    metric='cosine'
)

knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)

print("\n=== k-NN Results ===")
print("Accuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))