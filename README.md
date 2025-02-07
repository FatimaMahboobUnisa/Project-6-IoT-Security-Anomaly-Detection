# IoT Security Anomaly Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.15.1-green)](https://onnx.ai/)
[![Docker](https://img.shields.io/badge/Docker-ARM32v7-blue)](https://www.docker.com/)

A lightweight CNN model to detect malicious IoT device behavior using network traffic analysis. Deployed on Raspberry Pi via Docker.

## 📋 Project Overview
- **Objective**: Identify suspicious IoT activity with 92% accuracy.
- **Tools**: PyTorch, ONNX, Scapy, Docker (ARM32v7).
- **Key Results**: 15% reduction in false positives, edge-optimized inference.

## 🚀 Features
- **Packet Analysis**: Feature extraction from PCAP files using Scapy.
- **Edge AI**: ONNX model optimized for Raspberry Pi.
- **Real-Time Alerts**: MQTT integration for anomaly notifications.

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/iot-anomaly-detection.git
   cd iot-anomaly-detection

2.install dependencies:
   ```bash
pip install -r requirements.txt

3.Train the model (see train_model.ipynb).

4.Build Docker image for Raspberry Pi:
   ```bash
   docker build -f Dockerfile.edge -t iot-detector .
