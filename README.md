# Intelligent Monitoring, Analytics & Dashboarding System

## Overview
This project is an intelligent monitoring system that analyzes sensor-based data, visualizes trends, detects risk levels, and predicts system performance using machine learning.

## Features
- Reads sensor data from CSV
- Analyzes temperature, humidity, AQI, machine load, and energy usage
- Detects risk levels based on air quality
- Predicts system performance using Linear Regression
- Generates visualizations using matplotlib
- Saves graphs inside the visualizations folder

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Machine Learning
- Data Analytics

## Machine Learning Model
The system uses Linear Regression to predict performance based on:
- Temperature
- Humidity
- Air Quality Index
- Machine Load
- Energy Usage

## How to Run
```bash
pip install -r requirements.txt
python monitoring_system.py