# Milestone 5: Deployment & Automation (Revised)

This document covers the final steps for deploying the backend, automating the model training, and building the final Android app.

### **Objective:**
To have a fully automated, self-improving production system and a polished Android app installed on the user's device.

---

### **Tasks:**

#### 5.1: Configure and Deploy to Render
-   **Action:** Finalize the production deployment.
-   **Details:**
    -   Follow the instructions in the [Migration from Heroku Guide](./06_migration_from_heroku.md) to set up the application on Render.
    -   Ensure all necessary environment variables are set on Render (e.g., `DATABASE_URL` for Neon, `SECRET_KEY`, `ADMIN_API_KEY`).
    -   Push the `main` branch to trigger the CD pipeline, which will build and push the image to GHCR and trigger a deployment on Render.
    -   Verify the deployment by checking the Render logs and hitting the API endpoints.

#### 5.2: Schedule Daily Model Retraining
-   **Action:** Use GitHub Actions to automate the daily training job.
-   **Configuration:**
    -   A new workflow file, [`.github/workflows/retrain.yml`](./.github/workflows/retrain.yml:1), will be created to handle the scheduled job.
    -   This workflow will run daily at midnight UTC.
-   **Monitoring:** Check the GitHub Actions logs to ensure the scheduled job ran successfully.

#### 5.3: Build and Install the Final Android App
-   **Action:** Generate a signed, release-ready APK.
-   **Steps:**
    1.  Ensure the `build.gradle` file has the correct version code and version name.
    2.  Go to **Build > Generate Signed Bundle / APK**.
    3.  Use the previously created keystore to sign the APK.
    4.  Select the "release" build variant.
    5.  Transfer the generated `app-release.apk` to your device and install it.

#### 5.4: Final Verification
-   **Action:** Perform an end-to-end test of the complete system.
-   **Checklist:**
    -   [ ] User can register and log in on the Android app.
    -   [ ] User can log food and sport activities, and the data appears correctly.
    -   [ ] User can edit and delete past entries.
    -   [ ] The daily summary is accurate.
    -   [ ] The interactive plots load and display data correctly.
    -   [ ] The backend prediction endpoint returns a valid result.
    -   [ ] (After 24 hours) Check logs to confirm the retraining job ran as scheduled.