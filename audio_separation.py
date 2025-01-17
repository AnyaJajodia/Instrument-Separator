import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from librosa.core import stft, istft
from mir_eval.separation import bss_eval_sources

# --- Configurations ---
AUDIO_PATH = 'training_audio'  # Path to the folder containing training audio files
INSTRUMENT = 'vocals'  # Instrument to isolate (e.g., vocals, bass, drums, other)
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 0.001
N_FFT = 1024
HOP_LENGTH = 512

# --- Dataset Setup ---
class MUSDBDataset(Dataset):
    def __init__(self, audio_path, instrument, n_fft=1024, hop_length=512):
        self.audio_path = audio_path
        self.instrument = instrument
        self.tracks = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith('.wav')]
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track_path = self.tracks[idx]
        audio, _ = librosa.load(track_path, sr=None, mono=True)
        input_spec = self.audio_to_spectrogram(audio)

        # Placeholder for the target (modify this to load real targets)
        target_spec = input_spec  # Replace with actual separated instrument target

        return torch.tensor(input_spec, dtype=torch.float32), torch.tensor(target_spec, dtype=torch.float32)

    def audio_to_spectrogram(self, audio):
        return np.abs(stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))

# --- Model Definition ---
class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, output_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- Training Loop ---
def train_model():
    # Initialize dataset and dataloader
    dataset = MUSDBDataset(AUDIO_PATH, INSTRUMENT, n_fft=N_FFT, hop_length=HOP_LENGTH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = UNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for input_spec, target_spec in data_loader:
            input_spec = input_spec.unsqueeze(1)  # Add channel dimension
            target_spec = target_spec.unsqueeze(1)

            optimizer.zero_grad()
            output_spec = model(input_spec)
            loss = criterion(output_spec, target_spec)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(data_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), 'unet_model.pth')
    print("Model saved as 'unet_model.pth'")

# --- Evaluation Function ---
def evaluate(true_audio, predicted_audio):
    sdr, sir, sar, _ = bss_eval_sources(true_audio, predicted_audio)
    return sdr, sir, sar

if __name__ == "__main__":
    train_model()
