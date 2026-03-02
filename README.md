M3 Model API – Marian & Mona

Mini Project – Integration and Distribution

## Purpose of the project

This project is based on a previous deep learning project where we built a modular and versioned deep learning pipeline handling data loading, training, and evaluation.

The purpose of this project was to integrate the trained deep learning model into an application and distribute it using containerization.

We used Marian’s K2 model trained on the CIFAR-10 dataset.
The model is exported using TorchScript, served via a FastAPI application, dependency-managed using uv, and containerized with Docker.

## Project Architecture

Model exported using TorchScript

API built with FastAPI

Dependency management using uv

Containerization using Docker

Model inference executed at application startup

## Marian – ML Pipeline & Model Distribution
Responsible for the machine learning pipeline and model distribution:

Designed and trained the TinyCNN model
Implemented TorchScript export workflow
Ensured reproducible model loading via artifact management
Integrated exported model with FastAPI lifecycle (startup loading)
Verified containerized inference with Docker volume mounting
Debugged integration issues between model architecture and API input handling

## Mona – Application Foundation & API Integration
Responsible for application foundation and API integration:

Designed and implemented the application structure using FastAPI  
Developed the /health and /predict endpoints  
Managed JSON-to-tensor transformation for CIFAR-10 input data  
Integrated the TorchScript model into the inference pipeline  
Implemented mock-mode fallback when no model artifact is mounted  
Configured dependency management using uv  
Designed and built the Dockerfile for containerized deployment  
Tested API endpoints and verified end-to-end inference functionality  

## API TESTING

The /health endpoint was tested to verify that the application starts correctly.

The /predict endpoint was tested using a properly formatted CIFAR-10 tensor (32x32 spatial dimensions JSON input). The API successfully returned a predicted class, probability distribution, and model version, confirming correct TorchScript model loading and inference integration.

## CODE REVIEW
PR #1 – Model Export

https://github.com/monaabdi9085-dev/M3-model-api-Marian-Mona/pull/1

PR #2 – API + Docker

https://github.com/monaabdi9085-dev/M3-model-api-Marian-Mona/pull/2

Run locally

uv run python -m src.train
uv run python -m scripts.export_torchscript
uv run uvicorn app.main:app --reload

Run with Docker (real model)

docker build -t m3-api .
docker run -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" m3-api

Open: http://127.0.0.1:8000/docs

If no TorchScript artifact is mounted, the API runs in mock mode.

## Reflection

The highlights of this project were the opportunity to mimic real-life development workflows by using pull requests and providing structured feedback to each other.

One of the main challenges was coordinating our individual parts to avoid conflicts, which reflects common challenges in real-world collaborative software development.