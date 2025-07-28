# Nutrition Tracker

This is a nutrition tracking application.

## Local Development with Docker Compose

To set up a local development environment that mirrors the Render production setup, follow these steps:

### Prerequisites
*   Docker installed on your machine.

### Building and Running the Application

1.  **Build the Docker images:**
    Open your terminal in the project's root directory and run:
    ```bash
    docker-compose build
    ```
    This command will build the Docker image for the FastAPI backend based on the `Dockerfile` in the current directory.

2.  **Run the application:**
    Start the backend and database services by running:
    ```bash
    docker-compose up
    ```
    This command will start the PostgreSQL database and the FastAPI backend. The backend will automatically connect to the database using the credentials and URL defined in the `.env` file.

### Stopping the Application

To stop the running Docker containers, press `Ctrl+C` in the terminal where `docker-compose up` is running, or run the following command in a new terminal:
```bash
docker-compose down