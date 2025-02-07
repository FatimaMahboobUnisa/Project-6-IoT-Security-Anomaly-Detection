import torch
import torch.nn as nn
import numpy as np
from scapy.all import rdpcap

class AnomalyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3),  # Input: (batch, 1, sequence_length)
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 48, 2)  # Adjust based on input size
        )

    def forward(self, x):
        return self.cnn(x)

def preprocess_pcap(pcap_file):
    """Convert PCAP to feature vectors."""
    packets = rdpcap(pcap_file)
    features = []
    for pkt in packets[:1000]:  # Sample 1000 packets
        if pkt.haslayer("IP"):
            features.append([pkt["IP"].len, len(pkt.payload)])
    return torch.tensor(features).unsqueeze(1).float()  # Shape: (seq_len, 1, features)

# Example: Load ONNX model for edge deployment (see notebook for training)
def infer_anomaly(model_path="anomaly_detector.onnx"):
    sample_data = preprocess_pcap("sample.pcap")
    ort_session = ort.InferenceSession(model_path)
    outputs = ort_session.run(None, {"input": sample_data.numpy()})
    return outputs[0]
