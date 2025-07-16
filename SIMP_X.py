import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Avoids duplicate library errors
import cv2
from smplx import SMPLX ## for 3D human body modeling  || Output=3D mesh
import openpifpaf       ## for 2D human pose estimation || Output=2D coordinates
import pandas as pd 
from yolov5 import YOLOv5
import numpy as np


def detect_preson(image):
    """
    Detect  a person in the images and return their bounding box.
    
    Args:
      image :Numpy array(H,W,C) representing the images ====> (Heigth,Width,Channel)
    
    Return: list od bounding box [x1,y1,x2,y2] or none if no preson detected 
    """
    try:
        model=YOLOv5('yolov5s.pt',device='cpu')  # Load the pre-trained YOLOv5 model (small version for speed)
        results=model.predict(image) # Run detection on the image  and return tensor of  [x1, y1, x2, y2, confidence, class]
        detection= results.xyxy[0].cpu().numpy()   # Convert tensor to NumPy array  
        ## IN PREDICTION is class=0 meaning person because YOLO trianed on COCO dataset in that dataset class=0 , meaning person
        
        for det in detection:
            if int(det[5])==0:
                x1,y1,x2,y2=map(int,det[:4]) # Extract bounding box coordinates
                return [x1,y1,x2,y2]
        print("No person detected in the image")
        return None
    except Exception as e:
        print(f"Error in detect_person: {e}")
        return None

def estimated_pose(image):
    """
    Estimate 2D keypoints (like nose, eyes, shoulders, elbows, knees, etc.) from the cropped image
    
    Args:
     image:Cropped imageof the person(Numpy array)
     
     Return :
       array:keyPoint or None if Not detected
    """
    try:
        predictor=openpifpaf.Predictor(checkpoint='shufflenetv2k16') #OpenPifPaf pose estimation model, using a small and fast model: shufflenetv2k16
        predictions,_,_=predictor.numpy_image(image)  # takes NumPy image and returns only predictions=list of detected people and _,_ unused value 
        if predictions:
            return predictions[0].data # Returns keypoints for the first person
        print("No keypoints detected.")
        return None
    except Exception as e:
        print(f"Error in estimation_pose :{e}")

def get_clothes():
    """
       load a CSV file and then the size into a list
       Returns:
        list: List of clothing items with size ranges.
    """
    try:
        df=pd.read_csv(r"C:\Users\nasir\OneDrive\Desktop\Machine Learning\Extra_project\clothes.csv")
        return df.to_dict('records')  ## make a dictionary and retunr it according to size M:{12,24,45,78}
    except Exception as e:
        print(f"Error in get_clothes :{e} ")
        
        
def extract_measurements(mesh):
    """Extract Body measurement
    Args:
        mesh: SMPL-X mesh output.
    Retunrs:
    dict:Body MeasureMent(Height,chest,wasit)
    """
    try:
        vertices = mesh.vertices.detach().cpu().numpy().squeeze()
        """
        mesh.vertices ===> contains 3D coordinates of mesh vertices 
        .detach() ===> Detaches the tensor from the computation graph, meaning it won't track gradients anymore
        .cpu()    ===> Moves the tensor from GPU (if it was on one) to CPU
        .numpy()  ===> Converts the PyTorch tensor into a NumPy array
        .squeeze() ===> Removes dimensions of size 1
        """
        vertices = mesh.vertices.detach().cpu().numpy().squeeze()
        height = np.max(vertices[:, 1]) - np.min(vertices[:, 1])  # Y-axis 
        chest = np.max(vertices[:, 0]) - np.min(vertices[:, 0])  # X-axis 
        chest=chest*55
        waist = chest * 0.8  # Simplified ratio; replace with accurate computation
        print(f"Body Measumenent{[height,chest,waist]}")
        
        return {'height':height,'chest':chest,'waist':waist}
    except Exception as e:
        print(f"Error in extract_measurements: {e}")
        return {'height': 0, 'chest': 0, 'waist': 0}

def recommend_clothes(measurements, clothes):
    """Recommend clothes based on body measurement
    Arge:
        measurements (dict): User's body measurements.
        clothes (list): List of clothing items with size ranges.
    Returns:
        list: Recommended clothing items.
    """
    recommendations=[]
    for item in clothes:
        try:
            if(measurements['chest']<=item['chest_max'] and
               measurements['chest'] >= item['chest_min'] and
               measurements['waist'] <= item['waist_max'] and
               measurements['waist'] >= item['waist_min']):
               recommendations.append(item)
        except KeyError as e:
            print(f"Error in clothing item {item.get('name', 'unknown')}: Missing key {e}")
    print(recommendations)
    return recommendations   

def process_image(image_path):
    """Process of recomanding fitting clothes

    Args:
        image_path(str) :path to the imput image.
    
    Retunr:
        list:Recommended clothing items.
    """
    try:
        #Load and preprocess image
        image=cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image:{image_path}")
        
        ## if person is present 
        person_box=detect_preson(image)
        
        if person_box is None:
            print("Cannot Proceed :No person detected")
            return []
        
        #crop and preprocess image
        x1,y1,x2,y2=person_box
        cropped_image=image[y1:y2,x1:x2]
        if cropped_image.size == 0:
            print("Cannot Proceed :Cropped image in empty ")
        cropped_image=cv2.resize(cropped_image,(224,224))
        
        ## Estimate 2D keypoints 
        keypoints=estimated_pose(cropped_image)
        if keypoints is None:
            print("Cannot Proceed :No keyPoint detected")
            return []
        
        ## Fit SMPL-X model 
        smpl_x_path=r"C:\Users\nasir\OneDrive\Desktop\Machine Learning\Extra_project\models_smplx_v1_1\models\smplx\SMPLX_NEUTRAL.npz"
        if not os.path.exists(smpl_x_path):
            raise FileNotFoundError(f"SMPL-x model file not foind at :{smpl_x_path}")
        
        # Placeholder for SMPL-X fitting; requires optimization
        # Example: smpl_params = optimize_smplx(keypoints, smpl_x)
        # mesh = smpl_x(**smpl_params)
        smpl_x=SMPLX(model_path=smpl_x_path,gender='neutural')
        mesh=smpl_x()
        
        # Extract body Measurement 
        measurement=extract_measurements(mesh)
        
        clothes=get_clothes()
        if not clothes:
            print("Cannot Proceed :No Clothes data Available")
            return []
         ## Recommended clothes
        recommendations=recommend_clothes(measurement,clothes)
        return recommendations
    except Exception as e:
        print(f"Error is process_image :{e}")
        return []

if __name__ == "__main__":   ## it give a name to this from importing in another py file 
    image_path = r"C:\Users\nasir\OneDrive\Desktop\Machine Learning\Extra_project\Krishna.jpg"
    recommended_clothes = process_image(image_path)
    if recommended_clothes:
        for item in recommended_clothes:
            print(f"Recommended: {item['name']} (Size: {item['size']})")
    else:
        print("No clothing recommendations available.")


    
        
        
        
        
        

        
        

            
        
        
        