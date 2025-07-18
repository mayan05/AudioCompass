import torch
import torch.nn as nn

class CNN_LSTM_Model_PyTorch(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        # After 3 MaxPool2d(2), height & width are divided by 8
        self.lstm_input_size = 128 * ((150//8))  # height collapsed into features
        self.lstm = nn.LSTM(self.lstm_input_size, 128, batch_first=True, bidirectional=True)

        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128*2, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)  # (B, C, H, W)

        # Reshape to (B, time_steps, features)
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.contiguous().view(batch_size, x.size(1), -1)  # (B, W, C*H)

        lstm_out, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, 256)

        h = self.dropout1(h)
        h = torch.relu(self.fc1(h))
        h = self.dropout2(h)
        out = self.out(h)

        return out
    
    def predict(self,features):
        self.eval()

        device = next(self.parameters()).device
        features = features.to(device)

        with torch.no_grad():
            logits = self(features)
            logits_mean = logits.mean(dim=0)
            pred = torch.argmax(logits_mean).item()
            
        return pred




