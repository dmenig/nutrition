# Milestone 6: Migration from Heroku

This document outlines the plan to migrate the application from Heroku to a new architecture based on Render, GitHub Container Registry (GHCR), and GitHub Actions.

## 1. Update CI/CD to use GHCR

The existing Continuous Deployment workflow in [`.github/workflows/cd.yml`](./.github/workflows/cd.yml:1) needs to be updated to build the Docker image and push it to GHCR instead of the Heroku Container Registry.

**New `cd.yml` content:**

```yaml
name: CD

on:
  push:
    branches:
      - main

jobs:
  build_and_push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
    - uses: actions/checkout@v3

    - name: Log in to the GitHub Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v3
      with:
        context: .
        push: true
        tags: ghcr.io/${{ github.repository }}:${{ github.sha }}

    - name: Trigger Render Deployment
      run: curl -X POST ${{ secrets.RENDER_DEPLOY_HOOK_URL }}
```

**Required Secrets:**

*   `RENDER_DEPLOY_HOOK_URL`: This secret will be obtained from Render in the next step. It allows GitHub Actions to trigger a new deployment on Render.

## 2. Set up Application on Render

We will use Render to host our FastAPI application.

**Instructions:**

1.  Go to the Render Dashboard and create a new **Web Service**.
2.  Connect the GitHub repository for this project.
3.  Configure the service with the following settings:
    *   **Name:** `nutrition-api` (or a name of your choice)
    *   **Runtime:** `Docker`
    *   **Docker Repository:** The path to the image in GHCR (e.g., `ghcr.io/your-username/your-repo`).
    *   **Branch:** `main`
    *   **Auto-Deploy:** Yes
4.  Add the following environment variables:
    *   `DATABASE_URL`: The connection string for the Neon database.
    *   `SECRET_KEY`: The secret key for JWT.
    *   `ALGORITHM`: The algorithm for JWT.
5.  Create a **Deploy Hook URL** under the "Settings" tab. Copy this URL and add it as a secret named `RENDER_DEPLOY_HOOK_URL` in the GitHub repository settings.

## 3. Set up Scheduled Model Retraining

We will use a scheduled GitHub Actions workflow to run the daily model retraining job.

**Create a new file: `.github/workflows/retrain.yml`**

```yaml
name: Daily Model Retraining

on:
  schedule:
    - cron: '0 0 * * *' # Runs every day at midnight UTC

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run model retraining
      env:
        DATABASE_URL: ${{ secrets.DATABASE_URL }}
      run: python -m app.jobs.retrain_model
```

**Required Secrets:**

*   `DATABASE_URL`: The same Neon database connection string used by the Render application.

## 4. Update All Documentation

All references to Heroku must be removed from the project documentation.

**Files to Update:**

*   [`plan/01_backend_setup.md`](./plan/01_backend_setup.md:1)
*   [`plan/05_deployment.md`](./plan/05_deployment.md:1) (This file should be updated to reflect the new Render/GHCR architecture)
*   Any other markdown files containing the word "Heroku".