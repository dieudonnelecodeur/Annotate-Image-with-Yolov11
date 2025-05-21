FROM python:3.12-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

# Expose both FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 4600

# Default command (can be overridden in docker-compose)
CMD ["fastapi", "run", "app/main.py", "--port", "8000"]