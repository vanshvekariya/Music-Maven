import torch
from torch.utils.data import DataLoader
from encoder import AudioEncoder
from dataset import SimCLRDataset, load_all_files
from loss import NTXentLoss

def main():
    GTZAN_PATH = "../data/genres"
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 3e-4
    EMBEDDING_DIM = 128

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    file_paths, labels = load_all_files(GTZAN_PATH)
    print(f"Total files: {len(file_paths)}")

    dataset = SimCLRDataset(file_paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        prefetch_factor=2
    )

    encoder = AudioEncoder(embedding_dim=EMBEDDING_DIM).to(device)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)

            z1 = encoder(x1)
            z2 = encoder(x2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    torch.save(encoder.state_dict(), "../outputs/simclr_encoder.pt")
    print("Encoder saved")


if __name__ == "__main__":
    main()