@echo off
set API_PORT=8000
set STREAMLIT_PORT=4600
set API_FILE=main.py
set STREAMLIT_FILE=ui\web_interface.py

echo 🚀 Lancement de FastAPI sur http://localhost:%API_PORT%
start cmd /k fastapi dev %API_FILE% --host 127.0.0.1 --port %API_PORT%

echo 🖼️  Lancement de l'interface Streamlit sur http://localhost:%STREAMLIT_PORT%
streamlit run %STREAMLIT_FILE% --server.address=127.0.0.1 --server.port=%STREAMLIT_PORT%