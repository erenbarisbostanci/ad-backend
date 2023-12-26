FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN pip install -r requirements.txt

EXPOSE 5000

# Set environment variables (optional, for production)
# ENV FLASK_ENV=production
# ENV FLASK_APP=app.py

# Use Gunicorn for production-grade server (preferred for production)
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

# Alternatively, for development with Flask's built-in server:
CMD ["python", "app.py"]