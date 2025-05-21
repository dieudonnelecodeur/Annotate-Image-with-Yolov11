[![Coverage Status](https://img.shields.io/badge/coverage-97%25-brightgreen.svg)](https://github.com/Alex-Lekov/yolov8-fastapi)

# YOLOv8-FastAPI:
This repository serves as a template for object detection using YOLOv8 and FastAPI. With YOLOv8, you get a popular real-time object detection model and with FastAPI, you get a modern, fast (high-performance) web framework for building APIs. The project also includes Docker, a platform for easily building, shipping, and running distributed applications.

### Sample
Here's a sample of what you can expect to see with this project:

# What's inside:

- YOLO: A popular real-time object detection model
- FastAPI: A modern, fast (high-performance) web framework for building APIs
- Streamlit: An intuitive web interface for uploading images and visualizing results
- Docker: A platform for easily building, shipping, and running distributed applications

<img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png"></a>

---
# Getting Started

You have two options to start the application: using Docker or locally on your machine.

## Using Docker
Start the application with the following command:
```
docker-compose up
```

## Locally
To start the application locally, follow these steps:

1. Install the required packages:

```
pip install -r requirements.txt
```
2. Set up the Python Environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

2. Launch App:
```
.\launch_app.bat
```  
*Note: You can change the address and port in the file **docker-compose.yaml***

## FAST API Docs url:
http://0.0.0.0:8001/docs#/

Ready to start your object detection journey with YOLOv8-FastAPI? 🚀

---
# 🚀 Code Examples
### Example 1: Object Detection to JSON   
The following code demonstrates how to perform object detection and receive the results in JSON format:
```python
import requests

input_image_names = ['test/1.webp', 'test/2.webp']  # List of image file names
api_host = 'http://127.0.0.1:8000'
type_rq = '/process-images/'

# Prepare the files dictionary for multiple files
files = [('files', open(image_name, 'rb')) for image_name in input_image_names]

response = requests.post(api_host + type_rq, files=files)

data = response.json()
print(data)
```
Output:
```
{
  "results": [
    {
      "image": "39fe7920-71c1-11ef-a237-49738a978907.jpg.webp",
      "objects_detected": [
        "flood",
        "flood",
        "person",
        "person"
      ],
      "annotation": "Deux personnes traversant une zone inondée."
    }
  ]
}
```
