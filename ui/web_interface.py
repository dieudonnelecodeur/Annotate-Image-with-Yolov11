import streamlit as st
import requests
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="AfroVision Annotator",
    layout="centered",
    page_icon="🌍"
)

# Title and Description with African theme
st.markdown("""
    <h1 style='text-align: center; color: #1e3d59;'>🌍 AfroVision Annotator</h1>
    <p style='text-align: center; color: #ff914d;'>Describe African industry and climate anomaly images with the power of AI.</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar for upload only
st.sidebar.header("Upload Images")

image_paths = []

uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}")


if st.button("Get Annotation"):
    process_images_endpoint = "http://localhost:8000/process-images/"
    try:
        if uploaded_files:
            files = [("files", (uploaded_file.name, uploaded_file, uploaded_file.type)) for uploaded_file in uploaded_files]
            response = requests.post(process_images_endpoint, files=files)
            if response.status_code == 200:
                st.success("Detection/Annotation Results:")
                data = response.json()
                if "results" in data:
                    results = data["results"]
                    for item in results:
                        st.markdown(f"**Image name:** {item.get('image', 'N/A')}")
                        st.markdown(f"**Description:** {item.get('annotation', 'N/A')}")
                        st.markdown("---")
                else:
                    st.warning("No results found in the response.")
            else:
                st.error(f"Error: {response.json().get('detail', response.text)}")
        else:
            st.warning("Please provide at least one image.")
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")

# Footer with African flair
st.markdown("""
    <hr/>
    <p style='text-align: center; color: gray;'>Made for Africa | By Dieudonné N'DEDJELE</p>
""", unsafe_allow_html=True)