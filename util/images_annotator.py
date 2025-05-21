"""
Script pour l'annotation d'images africaines en utilisant Fashion MNIST, iMaterialist, 
Azure OpenAI API et YOLOv5.
"""

import os
import base64
import io
import openai
from openai import AzureOpenAI
from collections import Counter
import os
import glob
import torch
from PIL import Image
import openai
import matplotlib.pyplot as plt
from PIL import Image
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from ultralytics import YOLO  # Import the YOLO class from the ultralytics package

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)


def image_base64(image_path):
        """Prépare l'image pour l'envoi à Azure OpenAI"""
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()

# Étape 1 : Configurer YOLO
def load_best_custom_model(model_path='runs/detect/train17/weights/best.pt'):
    """Charge le modèle YOLO pré-entraîné."""
    model = YOLO(model_path) 
    print("YOLO chargé.")
    model.fuse()  # Fuse model layers for faster inference
    return model    


# Étape 2 : Détecter les objets avec YOLO
def detect_objects_with_yolo_using_image(model, image: Image):
    """Détecte les objets dans une image en utilisant YOLOv."""
    # Convert PIL image to a format YOLO can process
    results = model(image, conf=0.25)  # Perform detection

    # Extract detection boxes and class names
    detections = results[0].boxes  # Extract detection boxes
    class_ids = detections.cls.cpu().numpy().astype(int)  # Extract class IDs
    class_names = [model.names[class_id] for class_id in class_ids]  # Map class IDs to names
    print(f"Class Names: {class_names}")
    return class_names


def detect_objects_with_yolo_using_path(model, image_path):
    """Détecte les objets dans une image en utilisant YOLO."""
    results = model(image_path,conf=0.25)  # Perform detection
    
    for result in results:
        boxes = result.boxes.cpu().numpy()  # Get the boxes
        print(f"Boxes: {boxes}")
        
    # detections = results[0].boxes.data.cpu().numpy()  # Extract detection results
    detections = results[0].boxes  # Extract detection boxes
    class_ids = detections.cls.cpu().numpy().astype(int)  # Extract class IDs
    class_names = [model.names[class_id] for class_id in class_ids]  # Map class IDs to names
    
    print(f"Class Names: {class_names}")
    return class_names
    

# Étape 3 : Annoter les objets détectés avec Azure OpenAI
def annotate_with_openai_using_path(detections, image_path, model_name):
    """Génère des annotations basées sur les objets détectés en utilisant Azure OpenAI."""
    # Construire un prompt basé sur les objets détectés

    # Appeler Azure OpenAI
    try:
    
        response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un expert en analyse d'images. Fournis moi une description détaillée de l'image en tenant compte des objets détectés."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Sachant que les objects dectectes sont: {detections}. Décrivez cette image en tenant compte des objets détectés. La description doit être concise, pertinente et ne doit pas dépasser 8 mots."
                            }
                        ]
                    }
                ],
                max_tokens=10,
                temperature=0.2,
            )
    
        return response.choices[0].message.content

    except Exception as e:
        print(f"Erreur lors de l'annotation de l'image {image_path} : {e}")
        return None
    

# Étape 4 : Pipeline complet
def process_images_with_yolo_and_openai(image_directory, yolo_model, openai_model_name):
    """Pipeline complet : détection avec YOLOv5 et annotation avec Azure OpenAI."""
    
    image_list = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_list.extend(glob.glob(os.path.join(image_directory, ext)))
    # image_list = glob.glob(os.path.join(image_directory, "*.jpg"))
    if not image_list:
        print(f"Aucune image trouvée dans le répertoire : {image_directory}")
        return

    for image_path in image_list:
        print(f"Traitement de l'image : {image_path}")

        # Détection des objets avec YOLO
        detections = detect_objects_with_yolo_using_path(yolo_model, image_path)
        # image_description = analyze_image(image_path)  # Utiliser Azure Computer Vision pour obtenir une description de l'image
        print(f"Objets détectés : {detections}")

        # Annotation avec Azure OpenAI
        annotation = annotate_with_openai_using_path(detections, image_path, openai_model_name)

        # annotate_with_openai = generate_response_with_openai(image_description=analyze_image(image_path))
        
        
        print(f"Annotation générée : {annotation}")
        print("-" * 50)
        
# Étape 5 : Visualisation des résultats
def predict_with_yolo(model, source, conf=0.25,save=True):
    """Effectue une prédiction avec YOLO."""
    results = model.predict(source=source, conf=conf, save=save)  # Effectuer la prédiction
    print("Prédiction terminée.")
    return results

# Train model
# def train_yolo_model(model, data_path, epochs=50, imgsz=640,task='detect', plots=True):
def train_yolo_model(model, data_path, epochs=50, imgsz=640,task='detect'):
    """Entraîne un modèle YOLO avec les paramètres spécifiés."""
    # model = YOLO(model_path)  # Charger le modèle YOLO
    results = model.train(data=data_path, epochs=epochs, imgsz=imgsz,task=task)  # Entraîner le modèle
    print("Entraînement terminé.")
    return results

# Validate model
def validate_yolo_model(model, data_path, task='detect'):
    """
    Validate a YOLO model on a dataset.

    Args:
        model_path (str): Path to the trained YOLO model (e.g., 'runs/detect/train/weights/best.pt').
        data_path (str): Path to the dataset configuration file (e.g., 'data.yaml').
        task (str): Task type ('detect', 'segment', or 'classify').

    Returns:
        results: Validation results.
    """
    
    # Perform validation with the specified task
    results = model.val(data=data_path, task=task)
    
    print(f"Validation completed for task: {task}.")
    return results


    
    