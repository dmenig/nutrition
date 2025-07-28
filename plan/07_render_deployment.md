# Deploying the FastAPI Backend to Render

This document provides a step-by-step guide for deploying the FastAPI backend to Render.

## 1. Prerequisites

*   **Render Account:** You need a Render account. If you don't have one, sign up at [render.com](https://render.com).
*   **GitHub Repository:** Your code must be in a GitHub repository.
*   **Neon Account:** You need a Neon account for the PostgreSQL database. Sign up at [neon.tech](https://neon.tech).

## 2. Deployment Steps

### Step 2.1: Create a New Web Service on Render

1.  Go to the Render Dashboard and click **New +** > **Web Service**.
2.  Connect your GitHub account and select the repository for this project.

### Step 2.2: Configure the Web Service

*   **Name:** Give your service a unique name (e.g., `nutrition-api`).
*   **Region:** Choose a region close to you or your users.
*   **Branch:** Select the branch you want to deploy (e.g., `main` or `master`).
*   **Root Directory:** Leave this as the default unless your backend code is in a subdirectory.
*   **Runtime:** Select **Python 3**.
*   **Build Command:** `pip install -r requirements.txt`
*   **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
*   **Instance Type:** Choose an appropriate instance type (the free tier is a good starting point).

### Step 2.3: Set Up Environment Variables

You will need to add the following environment variable:

*   **Key:** `DATABASE_URL`
*   **Value:** Get this from your Neon database. In your Neon project, go to the **Connection Details** section and copy the **Postgres connection URL**.

You may also need to add other environment variables depending on your application's configuration.

### Step 2.4: Deploy

Click **Create Web Service**. Render will now build and deploy your application.

## 3. Post-Deployment Verification

1.  **Check the Logs:** Go to the **Logs** tab for your service in the Render dashboard to monitor the deployment process. Look for any errors.
2.  **Access the API:** Once the deployment is complete, you can access your API at the URL provided by Render (e.g., `https://your-service-name.onrender.com`).
3.  **Test an Endpoint:** Open a browser or use a tool like `curl` or Postman to test one of your API endpoints (e.g., `https://your-service-name.onrender.com/api/v1/plots/weight`).

## 4. Obtaining the Backend URL

The public URL for your backend is displayed at the top of your service's page in the Render dashboard. It will look something like this:

`https://your-service-name.onrender.com`

This is the URL you will use to update your Android application's configuration.