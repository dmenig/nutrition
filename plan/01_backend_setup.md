# Milestone 1: Backend Foundation (Revised)

This document details the tasks required to build the foundational backend infrastructure, incorporating the expanded data model and authentication requirements.

### **Objective:**
To establish a secure, well-structured, and auto-deploying backend service with a database schema that supports detailed food, sport, and custom food logging.

---

### **Tasks:**

#### 1.1: Initialize Git Repository & Set Up Project
-   **Action:** Create a new repository on GitHub and set up the FastAPI project structure as previously defined.
-   **Branching Strategy:** Use GitFlow (`main`, `develop`, feature branches).

#### 1.2: Define Expanded SQLAlchemy Models
-   **Action:** Create the database models in `app/db/models.py` based on the revised specification.
-   **Models to Create:**
    -   `User`: No changes.
    -   `FoodLog`: Add `quantity` (Float), `unit` (String), and change `logged_at` to `TIMESTAMP(timezone=True)`.
    -   `SportActivity`: New model with `id`, `user_id`, `activity_name`, `duration_minutes`, `calories_expended`, and `logged_at`.
    -   `CustomFood`: New model with `id`, `user_id`, `food_name`, and nutritional info per 100g.
-   **Database Migrations:** Set up `Alembic` to manage database schema migrations. Create an initial migration.

#### 1.3: Implement User Authentication
-   **Action:** Implement JWT-based authentication.
-   **Details:**
    -   Implement password hashing with `passlib` in `app/core/security.py`.
    -   Create `POST /api/v1/auth/register` and `POST /api/v1/auth/token` endpoints.
    -   Create a reusable authentication dependency to protect routes.
    -   Implement `GET /api/v1/users/me` to test authentication.

#### 1.4: Implement Custom Food Management
-   **Action:** Create CRUD endpoints for user-defined foods.
-   **Endpoints:**
    -   `POST /api/v1/custom-foods`: Allows an authenticated user to add a new food to their personal library.
    -   `GET /api/v1/custom-foods`: Retrieves all custom foods for the authenticated user.
    -   `DELETE /api/v1/custom-foods/{food_id}`: Deletes a custom food entry.

#### 1.5: Implement Sport Activity Logging
-   **Action:** Create CRUD endpoints for logging sport activities.
-   **Endpoints:**
    -   `POST /api/v1/sports`: Allows a user to log a new activity.
    -   `GET /api/v1/sports?date=YYYY-MM-DD`: Retrieves activities for a specific day.
    -   `DELETE /api/v1/sports/{activity_id}`: Deletes a logged activity.

#### 1.6: Set Up CI/CD Pipeline
-   **Action:** Create GitHub Actions workflows for automated testing and deployment.
-   **CI Workflow (`ci.yml`):**
    -   Triggers on push to any branch except `main`.
    -   Installs dependencies, runs `pytest` against all unit tests for the new endpoints.
-   **CD Workflow (`cd.yml`):**
    -   Triggers on push to `main`.
    -   Builds and pushes the backend Docker image to Heroku's container registry.
    -   Runs `alembic upgrade head` to apply any new database migrations.
    -   Releases the new version.