''''''''''''''''''GENAI PIONEER LABS'''''''''''''''''''
#GENAI PIONEER LABS
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split




 '''''''''''''''Simulate dataset'''''''''''''''''''''''
# Simulate dataset
def generate_data(n=200, img_size=64):
    X, y = [], []
    for i in range(n):
        img = np.random.rand(img_size, img_size)
        label = 0  # non-defective
        if i % 2 == 0:
            cv2.line(img, (5, 5), (50, 50), 1, 1)  # add defect
            label = 1
        X.append(img)
        y.append(label)
    return np.array(X).reshape(-1, img_size, img_size, 1), np.array(y)





    import os
import cv2
import numpy as np

def load_images_from_colab_folder(folder="/content/random", img_size=64):
    X = []
    y = []
    # Traverse through subfolders
    for root, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(root, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    X.append(img)
                    # Determine label based on subfolder name or filename
                    # Modify this logic based on your dataset structure
                    label = 0 # Default to normal

                    # Example 1: Labeling based on subfolder names (e.g., 'normal', 'defective')
                    if 'defective' in root.lower():
                         label = 1
                    # Example 2: Labeling based on filename (if "defect" is in the name)
                    # elif "defect" in file.lower():
                    #      label = 1

                    y.append(label)
                else:
                    print(f"Warning: Could not load image at {filepath}")

    X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
    y = np.array(y)
    return X, y

# # load (commented out the previous call)
# X, y = load_images_from_colab_folder("/content/random")





import matplotlib.pyplot as plt

# show first 9 images
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X[i].squeeze(), cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()





from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_img = Input(shape=(64,64,1))

# Encoder
x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

# Decoder
x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train on *normal* images only
normal_idx = np.where(y_train == 0)[0]
X_normal_train = X_train[normal_idx]

autoencoder.fit(X_normal_train, X_normal_train, epochs=10, batch_size=16, validation_split=0.2)

# calculate reconstruction error on ALL images in the test set
X_test_pred = autoencoder.predict(X_test)
reconstruction_error = np.mean((X_test - X_test_pred)**2, axis=(1,2,3))

# threshold can be picked by percentile or validation:
# We'll calculate the threshold based on the reconstruction errors of the normal images in the training set
reconstruction_error_normal_train = np.mean((X_normal_train - autoencoder.predict(X_normal_train))**2, axis=(1,2,3))
threshold = np.percentile(reconstruction_error_normal_train, 95)


anomalies_indices = np.where(reconstruction_error > threshold)[0]
print(f"Detected {len(anomalies_indices)} potential anomalies in the test set based on a threshold of {threshold:.4f}.")





import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.hist(reconstruction_error, bins=30, color='orange')
plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold:.4f}")
plt.title("Reconstruction Error Histogram")
plt.xlabel("MSE Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()





import matplotlib.pyplot as plt

# Sort anomalies by reconstruction error to show the most anomalous first
sorted_anomaly_indices = anomalies_indices[np.argsort(reconstruction_error[anomalies_indices])[::-1]]

n_show = min(9, len(sorted_anomaly_indices))
plt.figure(figsize=(10,10))
for idx, anomaly_idx in enumerate(sorted_anomaly_indices[:n_show]):
    plt.subplot(3,3,idx+1)
    # Need to index X_test here, as reconstruction_error is calculated on X_test
    plt.imshow(X_test[anomaly_idx].squeeze(), cmap='gray')
    plt.title(f"Anomaly {anomaly_idx}\nError={reconstruction_error[anomaly_idx]:.4f}")
    plt.axis('off')
plt.tight_layout()
plt.show()





import zipfile
import os

# Define the path to the uploaded zip file
zip_path = '/content/archive.zip'

# Define the directory to extract the contents to
extract_dir = '/content/dataset'

# Check if the zip file exists
if not os.path.exists(zip_path):
    print(f"Error: The file '{zip_path}' was not found.")
    print("Please upload the 'archive.zip' file to your Colab environment.")
else:
    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)

    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Dataset extracted to: {extract_dir}")

    # Now modify the load_images_from_colab_folder function to use the extracted directory
    # The function is already defined in a previous cell, so we will just call it with the new path
    # Make sure to adjust the label logic in the load_images_from_colab_folder function
    # if your dataset's file naming convention is different from the current "defect" rule.

    # For example, if you have subfolders named 'normal' and 'defective', you would need to modify
    # the load_images_from_colab_folder function to iterate through those subfolders and assign labels accordingly.

    # Assuming the current file naming convention with "defect" is applicable or you will adjust
    # the load_images_from_colab_folder function in the previous cell.
    # Call the existing function with the new path
    X, y = load_images_from_colab_folder(folder=extract_dir)

    print(f"Loaded {len(X)} images with {len(y)} labels.")
