import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException

from fastapi import FastAPI, File, status,UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from typing import List
from PIL import Image
from loguru import logger
import sys
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware

from util.images_annotator import (
    load_best_custom_model,
    detect_objects_with_yolo_using_image,
    annotate_with_openai_using_path,
)


####################################### logger #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI Setup #############################

# title
# Initialize FastAPI app
app = FastAPI(
    title="Image annotation using FastAPI",
    description=""" Detect image using YOLOv8 and annotate them using Azure OpenAI""",
    version="0.1.0",
)

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


best_custom_yolo_model = load_best_custom_model(model_path="models/best.pt")

openai_model_name = "gpt-4o-pionners29"
model="gpt-4-vision-preview",


# Allowed image extensions
ALLOWED_EXTENSIONS = {".png", ".jpeg", ".jpg", ".webp", ".bmp", ".gif"}

# Ensure the output directory exists
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/process-images/")
async def process_images(files: List[UploadFile] = File(...)):
    """
    Endpoint to process a list of image files and save annotations to a JSON file.
    """
    results = []

    for file in files:
        # Check file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue  # Skip non-image files

        try:
            # Read the image file
            image = Image.open(file.file).convert("RGB")

            # Save the image temporarily for YOLO processing
            temp_image_path = f"temp_{file.filename}"
            image.save(temp_image_path)

            # Detect objects with YOLO
            detections = detect_objects_with_yolo_using_image(best_custom_yolo_model, temp_image_path)

            # Generate annotation with Azure OpenAI
            annotation = annotate_with_openai_using_path(detections, temp_image_path, openai_model_name)

            # Append results
            results.append({
                "image": file.filename,
                "objects_detected": detections,
                "annotation": annotation,
            })

            # Clean up temporary file
            os.remove(temp_image_path)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")

    if not results:
        raise HTTPException(status_code=400, detail="No valid image files provided.")

    # Save results to a JSON file in the /output directory
    output_file_path = os.path.join(OUTPUT_DIR, "annotations.json")
    with open(output_file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    
    print(f"Annotations saved to {output_file_path}")
    # return {"results": results}

    return {"results": results}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


