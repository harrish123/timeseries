import torch
import numpy as np
import torch.nn as nn

def normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min + 1e-7), data_min, data_max

def denormalize(data, data_min, data_max):
    return data * (data_max - data_min + 1e-7) + data_min

class LSTMModel(nn.Module):
    def __init__(self, input_size = 1, hidden_layer_size=64, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
    
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1,-1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def predict_sequence(model, receipt_counts, num_days):
    device = "cuda"
    future_predictions = []

    normalized_data, data_min, data_max = normalize(np.array(receipt_counts, dtype=np.float32))

    current_seq = torch.tensor(normalized_data[:12], dtype=torch.float32).to(device=device)

    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

    for _ in range(num_days):
        with torch.inference_mode():
            model.eval()
            pred = model(current_seq)
            future_predictions.append(pred.item())

            current_seq = torch.cat((current_seq[1:], pred.view(1)), dim=0)
    
    predictions = denormalize(np.array(future_predictions), data_min, data_max)
    return predictions
