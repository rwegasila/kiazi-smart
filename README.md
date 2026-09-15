# 🥔 AI-Powered Smart Potato Farming and Marketplace Platform

![Kiazi Smart](https://img.shields.io/badge/Kiazi%20Smart-AI%20Agriculture-green)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌱 Overview

**Kiazi Smart** is an AI-powered smart agriculture platform designed to support potato farmers with **early disease detection, farm management, weather information, and agricultural market access**.

The platform combines **Artificial Intelligence, Deep Learning, Web Technologies, REST APIs, Docker, and Cloud Deployment** to provide farmers with accessible digital tools for improving potato production and decision-making.

The project is designed with the agricultural environment of **Tanzania**, particularly potato-growing regions such as **Mbeya**, in mind.

---

## 🎯 Project Objectives

Kiazi Smart aims to:

* 🥔 Detect potato leaf diseases using Artificial Intelligence.
* 🔬 Provide early disease identification from uploaded leaf images.
* 🌦️ Provide useful weather and climate information for farming decisions.
* 📊 Help farmers maintain digital farm records.
* 🛒 Connect farmers with agricultural markets.
* 📱 Provide an accessible mobile-friendly farming platform.
* 🚀 Deploy AI services through a scalable cloud-based backend.
* 🇹🇿 Support digital transformation in Tanzania's agricultural sector.

---

## 🤖 AI Disease Detection

One of the core features of Kiazi Smart is **AI-powered potato disease detection**.

The system uses a deep learning model trained to classify potato leaf images into four categories:

| Class             | Description                          |
| ----------------- | ------------------------------------ |
| 🟢 Healthy        | Healthy potato leaf                  |
| 🟡 Early Blight   | Potato leaf affected by Early Blight |
| 🔴 Late Blight    | Potato leaf affected by Late Blight  |
| ⚪ Not Potato Leaf | Image is not a potato leaf           |

### AI Pipeline

```text
Farmer
   │
   ▼
Upload / Capture Leaf Image
   │
   ▼
Image Preprocessing
   │
   ▼
Deep Learning Model
   │
   ▼
Disease Classification
   │
   ▼
Prediction Result
   │
   ▼
Farming Recommendation
```

---

## 🧠 Machine Learning Model

The project uses a **MobileNetV2-based deep learning model** for image classification.

### Model Configuration

```text
Architecture: MobileNetV2
Input Size: 224 × 224
Framework: TensorFlow / Keras
Task: Image Classification
Deployment: FastAPI + Docker
Inference: CPU
```

Model file:

```text
potato_disease_mobilenetv2_best.keras
```

---

## 🏗️ System Architecture

```text
                    ┌───────────────────────┐
                    │       Farmer          │
                    │   Mobile / Browser    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    Kiazi Smart UI     │
                    │      HTML / CSS / JS   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     FastAPI Backend   │
                    │        REST API       │
                    └───────────┬───────────┘
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
                 ▼              ▼              ▼
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │ AI Disease  │ │ Farm Data   │ │   Weather   │
        │ Detection   │ │ / Database  │ │ Information │
        └─────────────┘ └─────────────┘ └─────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ MobileNetV2 Model   │
        │ TensorFlow / Keras  │
        └─────────────────────┘
```

---

## 🚀 Main Features

### 👨‍🌾 Farmer Dashboard

Farmers can access important farming information from a centralized dashboard.

Features include:

* Farmer profile
* Farm information
* Farming records
* AI disease detection
* Weather information
* Market information

### 🔬 AI Disease Detection

Farmers can upload a potato leaf image and receive an AI prediction.

The system:

1. Receives the image.
2. Validates the uploaded file.
3. Resizes the image to `224 × 224`.
4. Processes the image.
5. Sends it to the trained MobileNetV2 model.
6. Generates a disease prediction.
7. Returns the result to the farmer.

### 🌦️ Weather Information

The platform is designed to provide climate and weather information useful for agricultural decision-making.

### 📋 Farm Records

Farmers can maintain digital information about their farms and agricultural activities.

### 🛒 Agricultural Marketplace

The platform provides a foundation for connecting farmers with markets and agricultural opportunities.

---

# 🛠️ Technology Stack

## Backend

* Python
* FastAPI
* Uvicorn
* SQLite
* REST API

## Artificial Intelligence

* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Pillow

## Frontend

* HTML
* CSS
* JavaScript

## Deployment

* Docker
* Docker Desktop
* Git
* GitHub
* Render

---

# 📁 Project Structure

```text
Kiazi Smart/
│
├── backend.py
├── kiazi-smart.html
├── kiazi.db
│
├── potato_disease_mobilenetv2_best.keras
│
├── requirements_backend.txt
├── Dockerfile
├── .dockerignore
│
└── README.md
```

---

# ⚙️ Installation and Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/rwegasila/kiazi-smart.git
```

Move into the project directory:

```bash
cd kiazi-smart
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements_backend.txt
```

---

## 4. Run the Backend

```bash
uvicorn backend:app --reload --port 8000
```

The application will be available at:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "kiazi-smart-ai-backend"
}
```

---

# 🐳 Running with Docker

Kiazi Smart is containerized using Docker.

## Build the Docker Image

```bash
docker build -t kiazi-smart .
```

## Run the Container

```bash
docker run -d --name kiazi-smart -p 8000:8000 kiazi-smart
```

Check running containers:

```bash
docker ps
```

Open:

```text
http://localhost:8000
```

---

# ☁️ Cloud Deployment

The application is deployed using **Render** with Docker.

### Production URL

🌐 **https://kiazi-smart.onrender.com**

### API Health Check

```text
https://kiazi-smart.onrender.com/health
```

The deployment architecture is:

```text
GitHub
   │
   ▼
Render
   │
   ▼
Docker Build
   │
   ▼
FastAPI Container
   │
   ├── Web Application
   ├── REST API
   └── AI Model
```

---

# 🔌 API Endpoints

## Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "service": "kiazi-smart-ai-backend"
}
```

## Disease Prediction

```http
POST /api/predict
```

The endpoint accepts a potato leaf image and returns the AI classification result.

---

# 🔐 Application Security

The backend is designed to separate the user interface from the AI inference service through a REST API architecture.

Future production improvements include:

* User authentication
* JWT-based authorization
* Role-based access control
* API rate limiting
* Secure environment variables
* Database migration to PostgreSQL
* HTTPS
* Input validation
* Logging and monitoring

---

# 📈 Future Improvements

Kiazi Smart is designed to grow into a complete digital agricultural ecosystem.

Planned improvements include:

### 🤖 Advanced AI

* More potato disease classes
* Higher-quality agricultural datasets
* Disease severity estimation
* Treatment recommendation system
* Pest detection
* Crop yield prediction
* AI farming assistant

### 🌦️ Smart Agriculture

* Real-time weather APIs
* Rainfall prediction
* Soil moisture monitoring
* IoT sensor integration
* Smart irrigation recommendations

### 🛒 Marketplace

* Farmer-to-buyer marketplace
* Potato price monitoring
* Market price prediction
* Buyer and seller accounts
* Agricultural input marketplace
* Digital payment integration

### 📱 Mobile Application

* Progressive Web App (PWA)
* Android application
* Offline functionality
* Push notifications
* GPS-based farm services

### 🗄️ Production Infrastructure

* PostgreSQL database
* Cloud object storage
* Model versioning
* Automated CI/CD
* Monitoring and logging
* Scalable AI inference

---

# 🇹🇿 Impact in Tanzania

Potato farming provides an important source of income and food security for many communities in Tanzania.

Kiazi Smart aims to use modern technology to help farmers:

* Detect diseases earlier.
* Reduce crop losses.
* Make better farming decisions.
* Access useful agricultural information.
* Connect with potential markets.
* Digitize farm management.
* Improve productivity and profitability.

The long-term vision is to build a **scalable Tanzanian agricultural technology platform** that can support farmers beyond potato production.

---

# 🎓 Academic & Technical Value

This project demonstrates practical application of:

* Data Science
* Machine Learning
* Deep Learning
* Computer Vision
* Artificial Intelligence
* REST API Development
* FastAPI
* TensorFlow
* Docker
* Cloud Deployment
* Database Management
* Web Application Development

It also demonstrates the complete machine-learning deployment lifecycle:

```text
Problem Definition
       ↓
Data Collection
       ↓
Data Preparation
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Export
       ↓
API Development
       ↓
Docker Containerization
       ↓
Cloud Deployment
       ↓
Real-World Application
```

---

# 👨‍💻 Developer

**Devis Rwegasila**

Data Science Engineering | Artificial Intelligence | Machine Learning | Data Analytics | Cloud & Backend Development

### Areas of Interest

* Data Science
* Artificial Intelligence
* Machine Learning
* Deep Learning
* Data Engineering
* Power Bi
* Computer Vision
* Cloud Computing
* Software Development
* Smart Agriculture

---

# 📄 License

This project is licensed under the **MIT License**.

---

# ⭐ Support the Project

If you find **Kiazi Smart** useful or interesting:

⭐ Star this repository
🍴 Fork the project
🐛 Report issues
💡 Suggest improvements
🤝 Contribute to the project

---

## 🌱 Kiazi Smart

> **Using Artificial Intelligence and Digital Technology to build a smarter future for potato farmers in Tanzania.**
