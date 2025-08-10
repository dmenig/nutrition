# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy tests
COPY tests /app/tests

# Copy the application code
COPY app /app/app
COPY train_model.py /app/
COPY build_features.py /app/
COPY sport_formulas.py /app/
COPY data_processor.py /app/
COPY utils.py /app/
COPY nutrition_calculator.py /app/
COPY process_nutrition_journal.py /app/
COPY plot_results.py /app/
COPY analyze_sensitivity.py /app/
COPY data /app/data
COPY app/db/populate_db.py /app/app/db/

# List the contents of the directory to debug file path issues

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.main:app when the container launches
CMD ["sh", "-c", "python app/db/populate_db.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]