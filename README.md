# US Accident Severity Prediction System

[![Deployment Status](https://img.shields.io/badge/Deployment-Live%20%26%20Secured-success?style=for-the-badge&logo=render)](https://accident-frontend.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)

**[ Live Demo: https://accident-frontend.onrender.com/](https://accident-frontend.onrender.com/)**

A robust, academically grounded Machine Learning pipeline and secured microservices architecture designed to predict traffic accident severity in real-time. This project implements a **Hierarchical Imputation Framework** and **Cost-Sensitive Gradient Boosting** to address the systematic sensor outages and extreme class imbalance inherent in US traffic data.

---

## Table of Contents
1.  [Methodology (Machine Learning)](#-methodology-machine-learning)
    *   [Data Forensics & Preprocessing](#1-data-forensics--preprocessing)
    *   [Hierarchical Imputation Strategy](#2-hierarchical-imputation-strategy)
    *   [Advanced Feature Engineering](#3-advanced-feature-engineering)
    *   [Model Selection & Performance](#4-model-selection--performance)
2.  [System Architecture](#-system-architecture)
3.  [Security & Hardening](#-security--hardening)
4.  [Deployment Strategy](#-deployment-strategy)
5.  [How to Run Locally](#-how-to-run-locally)

---

## Methodology (Machine Learning)


The core intelligence is built upon the **[US Accidents Dataset (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)**. Our approach moves beyond standard accuracy metrics to prioritize **Recall (Safety)**, ensuring that life-threatening incidents are not misclassified as minor events.

### 1. Data Forensics & Preprocessing
*   **The Seasonality Trap**: Initial temporal analysis revealed significant volume biases. To ensure relevance to modern traffic patterns, we implemented a **Strict Rolling Year Extraction** (March 2022 - March 2023), processing over 1.2 million records.
*   **Leakage Prevention (The "Whitelist")**: To eliminate "Look-Ahead Bias", features that are only known *after* an accident is cleared (e.g., `End_Lat`, `End_Lng`, `Duration`) were rigorously excluded. Instead, we simulated a "Zero-Hour" dispatch environment using only initial sensor readings and text feed keywords.

### 2. Hierarchical Imputation Strategy
We addressed the challenge of "Station-Wide Sensor Failure" (Missing at Random - MAR) using a novel **Three-Tier Imputation Framework** rather than generic mean filling:

1.  **Tier 1: Spatial Recovery (BallTree)**: Leveraged `BallTree` algorithms to map locations with missing weather data to their nearest active neighbor within a 50-mile radius.
2.  **Tier 2: Temporal Interpolation**: Used biochemical-style linear interpolation for short-term (<4 hour) sensor dropouts, respecting the thermal inertia of weather systems.
3.  **Tier 3: Seasonal/Diurnal Backfill**: For long-term gaps, data was reconstructed using the specific Month-Hour average for that city, preserving critical day/night cycles.

*Furthermore, missing `City` labels were recovered using a **K-Nearest Neighbors (KNN)** classifier based on precise geospatial coordinates.*

### 3. Advanced Feature Engineering
*   **Cyclical Time Encoding**: Transformed `Hour` and `Month` into Sine/Cosine vectors to preserve temporal continuity (e.g., 23:00 is mathematically adjacent to 00:00).
*   **Log-Normalization**: Applied `Log1p` transformation to skewed variance features like `Precipitation(in)` to stabilize gradient descent.
*   **NLP Whitelisting**: Mapped high-cardinality text descriptions to physics-based binary flags (e.g., `Desc_Blocked`, `Desc_Queue`) to capture kinetic energy states (Safety-in-Congestion theory).

### 4. Model Selection & Performance
We redefined the problem from a 4-class ordinal regression to a **Binary Classification** (Standard vs. Emergency Response) to resolve decision boundary fragmentation.

*   **Champion Model**: **LightGBM (Light Gradient Boosting Machine)**.
*   **Why LightGBM?**: It outperformed XGBoost and Random Forest in our "Audit of Errors", achieving a **96.9% Recall** on severe cases.
*   **Optimization**: Calibrated with a `class_weight=16` penalty to strictly prioritize False Negative reduction (missing a fatality) over False Positive production.

---

## System Architecture

The application is decoupled into two autonomous microservices ensuring scalability and isolation.

### Frontend (The Face)
*   **Tech Stack**: Streamlit (Python 3.9).
*   **Role**: Renders a reactive User Interface, validates inputs against the specific schema required by the model, and visualizes prediction probabilities.
*   **Interaction**: Communicates with the Backend via REST API, handling connection timeouts gracefully.

### Backend (The Brain)
*   **Tech Stack**: FastAPI + Uvicorn.
*   **Role**: Hosts the trained LightGBM model (`lgbm_tuned_model.pkl`) and RobustScaler artifacts.
*   **Efficiency**: Features `numpy`-accelerated vectorization for sub-50ms inference times.
*   **Endpoints**:
    *   `POST /predict`: The core inference engine.
    *   `GET /health`: For uptime monitoring.

---

## Security & Hardening

This project implements a **"Defense in Depth"** strategy suitable for public cloud deployment.

### 1. Shared Secret Authentication
To protect the publicly exposed Backend API from unauthorized access:
*   **Mechanism**: A cryptographic `X-Service-Token` is shared between Frontend and Backend as a guarded Environment Variable.
*   **Enforcement**: Middleware intercepts every request. If the token is missing or invalid, the connection is rejected with `403 Forbidden` before touching the model.

### 2. Container Security
*   **Non-Root Execution**: `Dockerfile` explicitly creates and switches to a non-privileged `appuser` (UID 1000). This mitigates privilege escalation attacks.
*   **Context Isolation**: strict `.dockerignore` patterns ensure that sensitive files (`.env`, `*.pem`, `*.git`) are never copied into the build image.

### 3. Error Sanitization
*   **Leakage Prevention**: The generic Exception Handler catches internal failures and suppresses stack traces, returning only Safe Error Messages to the client to prevent Information Disclosure.

---

## Deployment Strategy
**Platform**: [Render.com](https://render.com) (Cloud Container PaaS)

*   **Infrastructure as Code (IaC)**: The entire deployment topology is defined declaratively in `render.yaml`.
*   **Service Orchestration**:
    *   **Backend**: Deployed as a private-networking-capable Web Service.
    *   **Frontend**: Publicly exposed Web Service.
*   **Resilience**: Custom logic handles "Cold Start" latency, ensuring the user is informed if the free-tier instance is waking up.

---

## How to Run Locally

Replicate the production environment entirely on your machine using Docker.

**Prerequisites**: Docker Desktop installed.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Nduka99/accident_severity_prediction_repo.git
    cd accident_severity_prediction_repo
    ```

2.  **Launch via Docker Compose**
    ```bash
    docker-compose up --build
    ```

3.  **Access the App**
    *   Frontend: `http://localhost:8501`
    *   Backend Docs: `http://localhost:8000/docs` (Note: Docs are hidden in Production).

---
*Based on the MSc AI module assessment: "Engineering Road Safety: A Machine Learning Approach to Predicting Traffic Incident Lethality"*
