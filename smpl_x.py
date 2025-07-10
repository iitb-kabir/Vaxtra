import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoids duplicate library errors
import cv2
import numpy as np
from smplx import SMPLX  # Requires SMPL-X installation
import openpifpaf
import pandas as pd
from yolov5 import YOLOv5  # Install via: pip install yolov5

def detect_person(image):
    """
    Detect a person in the image and return their bounding box.
    
    Args:
        image: NumPy array (H, W, C) representing the image.
    
    Returns:
        list: Bounding box [x1, y1, x2, y2] or None if no person is detected.
    """
    try:
        # Load the pre-trained YOLOv5 model (small version for speed)
        model = YOLOv5('yolov5s.pt', device='cpu')  # Explicitly set to CPU
        # Run detection on the image
        results = model.predict(image)
        # Extract detections (format: [x1, y1, x2, y2, confidence, class])
        detections = results.xyxy[0].cpu().numpy()  # Convert tensor to NumPy array
        # Look for a person (class 0 in COCO dataset)
        for det in detections:
            if int(det[5]) == 0:  # Class 0 is 'person'
                x1, y1, x2, y2 = map(int, det[:4])  # Extract bounding box coordinates
                return [x1, y1, x2, y2]
        print("No person detected in the image.")
        return None
    except Exception as e:
        print(f"Error in detect_person: {e}")
        return None

def estimate_pose(image):
    """
    Estimate 2D keypoints from the cropped image.
    
    Args:
        image: Cropped image of the person (NumPy array).
    
    Returns:
        array: Keypoints or None if not detected.
    """
    try:
        # Load the predictor with a lightweight model
        predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
        # Run pose estimation on the image
        predictions, _, _ = predictor.numpy_image(image)
        if predictions:
            return predictions[0].data  # Returns keypoints for the first person
        print("No keypoints detected.")
        return None
    except Exception as e:
        print(f"Error in estimate_pose: {e}")
        return None

def get_clothes():
    """
    Retrieve clothes from a CSV file.
    
    Returns:
        list: List of clothing items with size ranges.
    """
    try:
        # Read the CSV file
        df = pd.read_csv('clothes.csv')
        # Expected columns: name, size, chest_min, chest_max, waist_min, waist_max
        return df.to_dict('records')
    except Exception as e:
        print(f"Error in get_clothes: {e}")
        return []

def process_image(image_path):
    """
    Process an image to recommend fitting clothes based on body measurements.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        list: Recommended clothing items.
    """
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Detect person
        person_box = detect_person(image)
        if person_box is None:
            print("Cannot proceed: No person detected.")
            return []
        
        # Crop and preprocess image
        x1, y1, x2, y2 = person_box
        cropped_image = image[y1:y2, x1:x2]
        if cropped_image.size == 0:
            print("Cannot proceed: Cropped image is empty.")
            return []
        cropped_image = cv2.resize(cropped_image, (224, 224))  # Normalize size
        
        # Estimate 2D keypoints
        keypoints = estimate_pose(cropped_image)
        if keypoints is None:
            print("Cannot proceed: No keypoints detected.")
            return []
        
        # Fit SMPL-X model
        smpl_x_path = r"C:\Users\nasir\OneDrive\Desktop\Machine Learning\Extra_project\smplx_models\SMPLX_NEUTRAL.npz"
        if not os.path.exists(smpl_x_path):
            raise FileNotFoundError(f"SMPL-X model file not found at: {smpl_x_path}")
        smpl_x = SMPLX(model_path=smpl_x_path, gender='neutral')  # Specify gender
        # Placeholder for SMPL-X fitting; requires optimization
        # Example: smpl_params = optimize_smplx(keypoints, smpl_x)
        # mesh = smpl_x(**smpl_params)
        mesh = smpl_x()  # Dummy call; implement proper fitting here
        
        # Extract body measurements
        measurements = extract_measurements(mesh)
        
        # Get clothes from database
        clothes = get_clothes()
        if not clothes:
            print("Cannot proceed: No clothes data available.")
            return []
        
        # Recommend clothes
        recommendations = recommend_clothes(measurements, clothes)
        return recommendations
    except Exception as e:
        print(f"Error in process_image: {e}")
        return []

def extract_measurements(mesh):
    """
    Extract body measurements from the SMPL-X 3D mesh.
    
    Args:
        mesh: SMPL-X mesh output.
    
    Returns:
        dict: Body measurements (height, chest, waist).
    """
    try:
        vertices = mesh.vertices.detach().cpu().numpy().squeeze()
        height = np.max(vertices[:, 1]) - np.min(vertices[:, 1])  # Y-axis height
        chest = np.max(vertices[:, 0]) - np.min(vertices[:, 0])  # X-axis approximation
        waist = chest * 0.8  # Simplified ratio; replace with accurate computation
        return {'height': height, 'chest': chest, 'waist': waist}
    except Exception as e:
        print(f"Error in extract_measurements: {e}")
        return {'height': 0, 'chest': 0, 'waist': 0}

def recommend_clothes(measurements, clothes):
    """
    Recommend clothes based on body measurements.
    
    Args:
        measurements (dict): User's body measurements.
        clothes (list): List of clothing items with size ranges.
    
    Returns:
        list: Recommended clothing items.
    """
    recommendations = []
    for item in clothes:
        try:
            if (measurements['chest'] <= item['chest_max'] and
                measurements['chest'] >= item['chest_min'] and
                measurements['waist'] <= item['waist_max'] and
                measurements['waist'] >= item['waist_min']):
                recommendations.append(item)
        except KeyError as e:
            print(f"Error in clothing item {item.get('name', 'unknown')}: Missing key {e}")
    return recommendations

# Example usage
if __name__ == "__main__":
    image_path = r"C:\Users\nasir\OneDrive\Desktop\Machine Learning\Extra_project\Krishna.jpg"
    recommended_clothes = process_image(image_path)
    if recommended_clothes:
        for item in recommended_clothes:
            print(f"Recommended: {item['name']} (Size: {item['size']})")
    else:
        print("No clothing recommendations available.")
