# Milestone 5: Deployment & Automation (Revised)

This document covers the final steps for deploying the backend, automating the model training, and building the final Android app.

### **Objective:**
To have a fully automated, self-improving production system and a polished Android app installed on the user's device.

---

### **Tasks:**

#### 5.1: Configure and Deploy to Heroku
-   **Action:** Finalize the production deployment.
-   **Details:**
    -   Ensure all necessary environment variables are set in Heroku (e.g., `DATABASE_URL`, `SECRET_KEY`, `ADMIN_API_KEY`).
    -   Push the `main` branch to trigger the CD pipeline, which will build, test, migrate the database, and deploy the backend.
    -   Verify the deployment by checking the Heroku logs and hitting the API endpoints.

#### 5.2: Schedule Daily Model Retraining
-   **Action:** Use the Heroku Scheduler to automate the daily training job.
-   **Configuration:**
    -   Add the Heroku Scheduler add-on to the application.
    -   Create a new job in the scheduler dashboard.
    -   **Command:** `python -m app.jobs.retrain_model`
    -   **Frequency:** Set to run `Daily` at a specific time (e.g., 3:00 AM UTC).
-   **Monitoring:** Check the Heroku logs the next day to ensure the scheduled job ran successfully.

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