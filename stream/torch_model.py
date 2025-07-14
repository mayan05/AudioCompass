# %%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import sys

# %%
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %%
features = np.load("features.npy")
labels = np.load("labels.npy")

X_train, X_temp, y_train, y_temp = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

BATCH_SIZE = 32

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)
test_ds  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)


# %%
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


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM_Model_PyTorch().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.96)

# %%
writer = SummaryWriter(log_dir='./logs')

best_val_acc = 0.0

for epoch in range(15):
    model.train()
    train_loss, train_correct = 0.0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(output, dim=1)
        train_loss += loss.item() * X_batch.size(0)
        train_correct += (preds == y_batch).sum().item()
    
    scheduler.step()
    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)

            preds = torch.argmax(output, dim=1)
            val_loss += loss.item() * X_batch.size(0)
            val_correct += (preds == y_batch).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/15] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Val", val_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_acc, epoch)
    writer.add_scalar("Accuracy/Val", val_acc, epoch)

    # checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")



# %%
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
test_correct = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        preds = torch.argmax(output, dim=1)
        test_correct += (preds == y_batch).sum().item()

test_acc = test_correct / len(test_loader.dataset)
print(f"✅ Test Accuracy: {test_acc:.4f}")



