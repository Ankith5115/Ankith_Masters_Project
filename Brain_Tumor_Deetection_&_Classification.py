import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



#----------------------------------------------------------------------------------------------
#File paths

base_path = "/home/tukl/Desktop/free/"
tumor_types = ["glioma", "meningioma", "pituitary"]

# Function to load and preprocess images
def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        labels.append(label)
    return np.array(images), np.array(labels)

X_tumor = []
y_tumor = []
X_type = []
y_type = []

X_no_tumor, y_no_tumor = load_images(os.path.join(base_path, "Training", "notumor"), 0)

for tumor_type in tumor_types:
    X, y = load_images(os.path.join(base_path, "Training", tumor_type), 1)
    X_tumor.extend(X)
    y_tumor.extend(y)
    X_type.extend(X)
    y_type.extend([tumor_types.index(tumor_type)] * len(y))

X_tumor = np.array(X_tumor)
y_tumor = np.array(y_tumor)
X_type = np.array(X_type)
y_type = np.array(y_type)

X_cnn = np.concatenate([X_tumor, X_no_tumor])
y_cnn = np.concatenate([y_tumor, y_no_tumor])

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42)

X_train_cnn = X_train_cnn.astype('float32') / 255
X_test_cnn = X_test_cnn.astype('float32') / 255
X_type = X_type.astype('float32') / 255


#-------------------------------------------------------------------------------------------------------

# Variables for KNN
X_train_knn = X_train_cnn
X_test_knn = X_test_cnn
y_train_knn = y_train_cnn
y_test_knn = y_test_cnn

# Variables for Naive Bayes
X_train_nb = X_train_cnn
X_test_nb = X_test_cnn
y_train_nb = y_train_cnn
y_test_nb = y_test_cnn

# Variables for Random Forest
X_train_rf = X_train_cnn
X_test_rf = X_test_cnn
y_train_rf = y_train_cnn
y_test_rf = y_test_cnn

# Variables for SVM
X_train_svm = X_train_cnn
X_test_svm = X_test_cnn
y_train_svm = y_train_cnn
y_test_svm = y_test_cnn



#---------------------------------------------------------------------------------------------------
#CNN MODEL
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Create and compile CNN model
cnn_model = create_cnn_model()
cnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train CNN model
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = cnn_model.fit(X_train_cnn, y_train_cnn, epochs=50,
                        validation_data=(X_test_cnn, y_test_cnn),
                        batch_size=32,
                        callbacks=[early_stopping])

cnn_loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print(f"CNN Accuracy: {cnn_acc:.4f}")




# Extract accuracy and loss from history
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot Accuracy
plt.figure(figsize=(12, 6))

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()





#--------------------------------------------------------------------------------------------
#SVM MODEL

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Flatten data for SVM
X_train_svm_flat = X_train_svm.reshape(X_train_svm.shape[0], -1)
X_test_svm_flat = X_test_svm.reshape(X_test_svm.shape[0], -1)

# Initialize and train SVM model
svm_model = SVC(C=1000, degree=2, gamma=0.001, kernel='rbf', probability=True)
svm_model.fit(X_train_svm_flat, y_train_svm)

# Evaluate SVM model
svm_acc = svm_model.score(X_test_svm_flat, y_test_svm)
print(f"SVM Accuracy: {svm_acc:.4f}")


#ACCURACY PLOT FOR SVM

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Predict using the SVM model
y_pred_svm = svm_model.predict(X_test_svm_flat)

# Calculate accuracy
svm_acc = accuracy_score(y_test_svm, y_pred_svm)

# Scatter plot for Actual vs Predicted (SVM)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_svm, y_pred_svm, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([min(y_test_svm), max(y_test_svm)], [min(y_test_svm), max(y_test_svm)], color='red', linestyle='--', label="Perfect Prediction")

# Add accuracy on top
plt.text(0.5, max(y_pred_svm) - 0.05, f'Accuracy: {svm_acc*100:.2f}%', ha='center', fontsize=12, color='blue')

# Labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVM: Actual vs Predicted')
plt.legend()

# Show plot
plt.show()






#-----------------------------------------------------------------------------------------------------
#NAIVE BYES MODEL

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Flatten data for Naive Bayes
X_train_nb_flat = X_train_nb.reshape(X_train_nb.shape[0], -1)
X_test_nb_flat = X_test_nb.reshape(X_test_nb.shape[0], -1)

# Initialize and train Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train_nb_flat, y_train_nb)

# Evaluate Naive Bayes model
nb_acc = nb_model.score(X_test_nb_flat, y_test_nb)
print(f"Naive Bayes Accuracy: {nb_acc:.4f}")


#ACCURACY PLOT

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Predict using the Naive Bayes model
y_pred_nb = nb_model.predict(X_test_nb_flat)

# Calculate accuracy
accuracy = accuracy_score(y_test_nb, y_pred_nb)

# Scatter plot for Actual vs Predicted (Naive Bayes)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_nb, y_pred_nb, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([min(y_test_nb), max(y_test_nb)], [min(y_test_nb), max(y_test_nb)], color='red', linestyle='--', label="Perfect Prediction")

# Add accuracy on top
plt.text(0.5, max(y_pred_nb) - 0.05, f'Accuracy: {accuracy*100:.2f}%', ha='center', fontsize=12, color='green')

# Labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Naive Bayes)')
plt.legend()

# Show plot
plt.show()





#-----------------------------------------------------------------------------------------------------
#RANDOM FOREST MODEL

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf.reshape(X_train_rf.shape[0], -1), y_train_rf)

# Evaluate Random Forest model
y_pred_rf = rf_model.predict(X_test_rf.reshape(X_test_rf.shape[0], -1))
rf_acc = accuracy_score(y_test_rf, y_pred_rf)
print(f"Random Forest Accuracy: {rf_acc:.4f}")


#ACCURACY PLOT

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Predict using the Random Forest model
y_pred_rf = rf_model.predict(X_test_rf.reshape(X_test_rf.shape[0], -1))

# Calculate accuracy
accuracy = accuracy_score(y_test_rf, y_pred_rf)

# Scatter plot for Actual vs Predicted (Random Forest)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_rf, y_pred_rf, color='green', alpha=0.5, label='Predicted vs Actual')
plt.plot([min(y_test_rf), max(y_test_rf)], [min(y_test_rf), max(y_test_rf)], color='red', linestyle='--', label="Perfect Prediction")

# Add accuracy on top
plt.text(0.5, max(y_pred_rf) - 0.05, f'Accuracy: {accuracy*100:.2f}%', ha='center', fontsize=12, color='green')

# Labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Random Forest)')
plt.legend()

# Show plot
plt.show()





#---------------------------------------------------------------------------------------------------------------------
#KNN MODEL

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Initialize and train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_knn.reshape(X_train_knn.shape[0], -1), y_train_knn)

# Evaluate KNN model
y_pred_knn = knn_model.predict(X_test_knn.reshape(X_test_knn.shape[0], -1))
knn_acc = accuracy_score(y_test_knn, y_pred_knn)
print(f"KNN Accuracy: {knn_acc:.4f}")


#ACCURACY MODEL

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)

# Scatter plot for Actual vs Predicted (KNN)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_knn, y_pred_knn, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([min(y_test_knn), max(y_test_knn)], [min(y_test_knn), max(y_test_knn)], color='red', linestyle='--', label="Perfect Prediction")

# Add accuracy on top
plt.text(0.5, max(y_pred_knn) - 0.05, f'Accuracy: {accuracy_knn*100:.2f}%', ha='center', fontsize=12, color='blue')

# Labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('KNN: Actual vs Predicted')
plt.legend()

# Show plot
plt.show()





#--------------------------------------------------------------------------------------------------------------------------------
#OVERALL COMPARISION OF FOUR MODELS

import matplotlib.pyplot as plt

# Collect all model names and accuracies
model_names = ['SVM', 'Naive Bayes', 'KNN', 'Random Forest']
accuracies = [svm_acc, nb_acc, knn_acc, rf_acc]

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim(0, 1)  # Accuracy range
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.4f}", ha='center', fontsize=12)
plt.show()


#LINE PLOT

import matplotlib.pyplot as plt

# Compare performances
model_names = ['SVM', 'Naive Bayes', 'KNN', 'Random Forest']
accuracies = [svm_acc, nb_acc, knn_acc, rf_acc]

# Plot the accuracies as a line plot
plt.figure(figsize=(10, 6))
plt.plot(model_names, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim(0, 1)  # Accuracy range
plt.grid(True)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.4f}", ha='center', fontsize=12)
plt.show()


#SCATTER PLOT

import matplotlib.pyplot as plt

# Compare performances
model_names = ['SVM', 'Naive Bayes', 'KNN', 'Random Forest']
accuracies = [svm_acc, nb_acc, knn_acc, rf_acc]

# Plot the accuracies as a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(model_names, accuracies, color='purple', s=100, marker='o')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)  # Accuracy range
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.4f}", ha='center', fontsize=12)
plt.show()



<img width="466" height="301" alt="Screenshot 2026-06-22 193809" src="https://github.com/user-attachments/assets/48f73093-725b-4fc6-a199-32a033987bc1" />


#====================================================================== END OF CODE =============================================================================

