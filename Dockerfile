# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

ENV PYTHONPATH=/app

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy specific application files and directories
COPY alembic.ini ./
COPY migrations/ ./migrations/
RUN rm -rf /app/app
COPY app/. ./app/
COPY build_features.py ./
COPY data_processor.py ./
COPY sport_formulas.py ./
COPY utils.py ./
COPY nutrition_calculator.py ./
COPY train_model.py ./
COPY data/ ./data/


# Make port 8000 available to the world outside this container
EXPOSE 8000
# Run app.main:app when the container launches
# Assuming the FastAPI app is in app/main.py and named 'app'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]