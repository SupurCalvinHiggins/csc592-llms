from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ACSDataset(Dataset):
    def __init__(self, data_path: Path, transform=None):
        self.data_path = data_path
        self.transform = transform

        airplane_paths = list(self.data_path.glob("airplanes/*.jpg"))
        car_paths = list(self.data_path.glob("cars/*.jpg"))
        ship_paths = list(self.data_path.glob("ships/*.jpg"))

        self.img_paths = airplane_paths + car_paths + ship_paths
        self.labels = (
            [0] * len(airplane_paths) + [1] * len(car_paths) + [2] * len(ship_paths)
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_train_loader(data_path: Path, batch_size: int, transform=None) -> DataLoader:
    dataset = ACSDataset(data_path, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def get_test_loader(data_path: Path, batch_size: int, transform=None) -> DataLoader:
    dataset = ACSDataset(data_path, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 224 * 224, 100)
        self.fc2 = nn.Linear(100, 3)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 3 * 224 * 224)
        x = F.relu(self.fc1(x))
        x = self.sm(self.fc2(x))
        return x


def main(batch_size: int = 128, epochs: int = 25) -> None:
    base_data_path = Path("../datasets/Dataset_PlanesCarsShips/")
    train_data_path = base_data_path / "train"
    test_data_path = base_data_path / "test"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_loader = get_train_loader(train_data_path, batch_size, transform)
    test_loader = get_test_loader(test_data_path, batch_size, transform)

    model = MLP().cuda()
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), momentum=0.9)

    for epoch in range(epochs):
        total_loss = torch.tensor(0.0, device="cuda")
        for images, labels in train_loader:
            labels, images = (
                labels.cuda(non_blocking=True),
                images.cuda(non_blocking=True),
            )
            opt.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            total_loss += loss

        total_loss = total_loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}]: train_loss = {total_loss}")

        total = torch.tensor(0, device="cuda")
        correct = torch.tensor(0, device="cuda")
        with torch.no_grad():
            for images, labels in test_loader:
                labels, images = (
                    labels.cuda(non_blocking=True),
                    images.cuda(non_blocking=True),
                )
                outputs = model(images)
                predicted_labels = outputs.argmax(dim=-1)
                total += predicted_labels.size(0)
                correct += (predicted_labels == labels).sum()

        total = total.item()
        correct = correct.item()
        print(f"Epoch [{epoch + 1}/{epochs}]: test_acc = {correct / total}")


if __name__ == "__main__":
    main()
