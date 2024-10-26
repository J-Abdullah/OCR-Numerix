import joblib
import torch
import cv2
import json
import os
import numpy as np
import pandas as pd
from PIL import Image
from math import atan2, degrees
import pytesseract
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the saved label encoders, scaler, and model
label_encoder_doc_type = joblib.load('label_encoder_doc_type.pkl')
label_encoder_category = joblib.load('label_encoder_category.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('classification_model.pkl')

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) 

def image_croper(image_path, model):
    # Load and detect objects in image
    image = cv2.imread(image_path)
    results = model(image)

    if results.xyxy[0].shape[0] == 0:
        return image
    else:
        # Extract bounding box coordinates for the document 
        for result in results.xyxy[0]:  # Iterate through results
            x1, y1, x2, y2, confidence, class_id = result  # Bounding box coordinates and class info
            # Crop document from the image
            document_crop = image[int(y1):int(y2), int(x1):int(x2)]

        return document_crop
    
def perform_ocr(image):
    height, width, _ = image.shape
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR with bounding box data
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    
    # Prepare output dictionary
    ocr_results = {"ocr": {}}
    
    # Process each detected word
    for i in range(len(data["text"])):
        word = data["text"][i]
        if word.strip() == "":  # Skip empty results
            continue

        # Get bounding box coordinates
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        
        # Normalize coordinates to the image dimensions
        normalized_bbox = [
            {"x": x / width, "y": y / height},
            {"x": (x + w) / width, "y": y / height},
            {"x": (x + w) / width, "y": (y + h) / height},
            {"x": x / width, "y": (y + h) / height},
        ]
        
        # Store word and bounding box in the output dictionary
        if word in ocr_results["ocr"]:
            ocr_results["ocr"][word].append(normalized_bbox)
        else:
            ocr_results["ocr"][word] = [normalized_bbox]
    
    return ocr_results

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size
    
def calculate_slope(x_coords, y_coords):
    if len(x_coords) >= 2 and len(y_coords) >= 2:
        return degrees(atan2(y_coords[1] - y_coords[0], x_coords[1] - x_coords[0]))
    return 0

def extract_features(ocr_results, image_path, doc_type):
    feature_list = []
    # Get image size
    image_width, image_height = get_image_size(image_path)    
    for text, coords in ocr_results['ocr'].items():
        for coord_set in coords:
            x_coords = np.array([coord['x']  for coord in coord_set])
            y_coords = np.array([coord['y']  for coord in coord_set])
        
        # Calculate width, height, and slope
        width = np.ptp(x_coords)  # np.ptp gives the range (max - min)
        height = np.ptp(y_coords)
        slope = calculate_slope(x_coords, y_coords)
        
        # Append features, including X_Y coordinates
        feature_list.append({
            'document_type': doc_type,
            'min_x': np.min(x_coords),
            'max_x': np.max(x_coords),
            'min_y': np.min(y_coords),
            'max_y': np.max(y_coords),
            'width': width,
            'height': height,
            'slope': slope,
            'image_width': image_width,
            'image_height': image_height,
            'text': text,
        })

    return pd.DataFrame(feature_list) 

def predict_category(df, model, category_encoder, doc_encoder):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    predictions = []

    for index, row in df_copy.iterrows():
        # Extract the document_type from the row and encode it
        document_type = row[0]
        document_type_encoded = doc_encoder.transform([document_type])[0]
        
        # Replace the document_type in the row with its encoded value
        row[0] = document_type_encoded
        
        # Exclude the last column from the features array
        features_array = np.array(row[:-1]).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make a prediction
        prediction = model.predict(features_scaled)
        
        # Decode the predicted label
        predicted_category = category_encoder.inverse_transform(prediction)[0]
        
        # Append the predicted category to the predictions list
        predictions.append(predicted_category)
    
    # Add the predictions as a new column in the DataFrame copy
    df_copy['Predicted Category'] = predictions
    
    return df_copy

def main(image_path, doc_type):
    # Step 1: Crop the image
    img = image_croper(image_path, yolo_model)

    # Step 2: Perform OCR on the cropped image
    ocr = perform_ocr(img)

    # Step 3: Extract features from the OCR results
    df = extract_features(ocr, image_path, doc_type)

    # Step 4: Predict the category of each text segment
    df_with_predictions = predict_category(df, model, label_encoder_category, label_encoder_doc_type)

    df_with_predictions.to_csv('ocr_system_results.csv', index=False)
    # Step 5: Filter out unknown predictions
    df_filtered = df_with_predictions[df_with_predictions['Predicted Category'] != 'unknown']

    # Print the final DataFrame
    print(df_filtered)

if __name__ == "__main__":
    image_path = 'sample.jpeg'
    # TODO: make system to detect doc_type automaticly
    doc_type = 'botswana_none_idcard_design1'
    main(image_path, doc_type)