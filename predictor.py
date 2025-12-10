import os
import numpy as np
import cv2
from tensorflow import keras
import joblib
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Get root project path dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_final.keras")
MOBILENET_MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv2_severity_final.keras")


RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_severity.pkl")

print("Loading models from:", MODEL_DIR)
print("CNN:", CNN_MODEL_PATH)
print("MobileNet:", MOBILENET_MODEL_PATH)
print("Random Forest:", RF_MODEL_PATH)

cnn_model = keras.models.load_model(CNN_MODEL_PATH)
mobilenet_model = keras.models.load_model(MOBILENET_MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)

# Class mapping
CLASS_NAMES = ["Minor", "Moderate", "None", "Severe"]

# IMAGE PREPROCESSING
def preprocess_cnn(image_path, size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")
    return np.expand_dims(img, axis=0)

def preprocess_mobilenet(image_path, size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype("float32"), axis=0)

def preprocess_for_rf(image_path, size=(128, 128)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.flatten().reshape(1, -1)

def predict_severity(image_path, model_name):
    model_name = model_name.lower()

    if model_name == "cnn":
        img = preprocess_cnn(image_path)
        probs = cnn_model.predict(img)
        print("CNN prediction:", probs)
        return CLASS_NAMES[np.argmax(probs)]

    elif "mobilenetv2" in model_name:
        img = preprocess_mobilenet(image_path)
        probs = mobilenet_model.predict(img)
        print("MobileNetV2 prediction:", probs)
        return CLASS_NAMES[np.argmax(probs)]

    elif "random forest" in model_name:
        img = preprocess_for_rf(image_path)
        pred = rf_model.predict(img)
        print("RF prediction:", pred)
        return CLASS_NAMES[pred[0]]
    

    
    return "Unknown"

