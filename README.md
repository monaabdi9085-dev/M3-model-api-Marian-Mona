M3 Model API – Marian & Mona

Mini Project – Integration and Distribution

## Purpose of the project 
This project is based on a previous deep learning project where we built a modular and versioned deep learning pipeline handling data loading, training, and evaluation.

The purpose of this project was to integrate the trained deep learning model into an application and distribute it using containerization.

We used Marian’s K2 model trained on the CIFAR-10 dataset.
The model is exported using TorchScript, served via a FastAPI application, dependency-managed using uv, and containerized with Docker.

## Project Architechture 

Model exported using TorchScript

API built with FastAPI

Dependency management using uv 

Contanerization using Docker 

Model inference executed at application startup


# CODE REVIEW 
PR #1 – Model Export

[Link to PR #1]

PR #2 – API + Docker
[Link to PR #2]


## Reflection 

The highlights of this project were the opportunity to mimic real-life development workflows by using pull requests and providing structured feedback to each other.

One of the main challenges was coordinating our individual parts to avoid conflicts, which reflects common challenges in real-world collaborative software development.