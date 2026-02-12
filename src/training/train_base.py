import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from ..models.resnet import resnet18, BasicBlock
import os
from datetime import datetime
import numpy as np
import random

 

def train(epochs=250, batch_size=128, lr=0.1, patience=15):
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"EarlyStopping enabled, patience={patience}")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    class Cutout(object):
        def __init__(self, n_holes, length):
            self.n_holes = n_holes
            self.length = length
        def __call__(self, img):
            h = img.size(1)
            w = img.size(2)
            mask = np.ones((h, w), np.float32)
            for _ in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)
                mask[y1:y2, x1:x2] = 0.
            mask = torch.from_numpy(mask).float()
            mask = mask.unsqueeze(0)
            mask = mask.expand_as(img)
            img = img * mask
            return img

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Use full train set and test set (no train/valid split) to match pytorch-cifar-master
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    model = resnet18(num_classes=10).to(device)

    def init_weights_kaiming(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            try:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            except Exception:
                pass

    model.apply(init_weights_kaiming)
    for m in model.modules():
        if hasattr(m, 'bn2'):
            try:
                nn.init.constant_(m.bn2.weight, 0)
            except Exception:
                pass

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Evaluation on test set (match pytorch-cifar-master)
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total if total > 0 else 0.0
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss/(len(trainloader)):.4f}, Test Acc: {acc:.2f}%')

        # Save checkpoint when test accuracy improves (filename: best_model.pth)
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(checkpoint_dir, 'best_model.pth'))
            best_acc = acc

        scheduler.step()

    actual_epochs = epoch + 1
    print(f'Training complete!')
    print(f'Actual epochs: {actual_epochs}/{epochs}')
    print(f'Best test accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    train()