#!/bin/bash

# Nom du script : launch_app.sh
# Usage : ./launch_app.sh
# Description : Lance l'API FastAPI et l'interface utilisateur Streamlit

# Ports
API_PORT=8000
STREAMLIT_PORT=4600

# Fichiers
API_FILE="main.py"
STREAMLIT_FILE="ui/web_interface.py"

# Lancer le serveur FastAPI en arrière-plan
echo "🚀 Lancement de FastAPI sur http://localhost:$API_PORT"
fastapi dev $API_FILE --host 127.0.0.1 --port $API_PORT &

# Lancer l'interface Streamlit
echo "🖼️  Lancement de l'interface Streamlit sur http://localhost:$STREAMLIT_PORT"
streamlit run $STREAMLIT_FILE --server.address=127.0.0.1 --server.port=$STREAMLIT_PORT
