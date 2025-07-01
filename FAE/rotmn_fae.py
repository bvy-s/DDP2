#  [1] Imports and Configuration
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import random
from torchvision.transforms import functional as func


# Configuration
class Config:
    batch_size = 128
    num_epochs = 5
    lr = 1e-4
    num_source_domains = 5  # 0°, 30°, 60°
    source_domains = [15, 30, 45, 60, 75]
    target_domains = [0, 90]       # Unseen domain
    num_classes = 10

config = Config()


#  [2] Create Rotated MNIST Dataset (Based on [3] and [5])
class RotatedMNIST(Dataset):
    def __init__(self, base_dataset, rotation_angles):
        self.base_dataset = base_dataset
        self.rotation_angles = rotation_angles
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        rot = random.choice(self.rotation_angles)
        img = func.rotate(img, rot)
        img = self.transform(img)
        domain = int(rot/15 - 1)  # Random domain from source
        return img, label, domain  # Returns (image, class_label, domain_label)

    def __len__(self):
        return len(self.base_dataset)
    

# Load base MNIST
train_base = torchvision.datasets.MNIST(root='./data', train=True, download=True)
test_base = torchvision.datasets.MNIST(root='./data', train=False, download=True)

train_dataset = RotatedMNIST(train_base, config.source_domains)
test_dataset = RotatedMNIST(test_base, config.target_domains)  # Unseen domain


# [3] Data Loaders with Stratified Sampling (Modified from [2])
train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle=True, num_workers=2)


#  [4] Model Architecture (Domain-Adversarial + Evidential)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(1600, 128)  # Adjusted for MNIST size

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.grl = GradientReversal.apply
        self.net = nn.Sequential(
            nn.Linear(128, 50), nn.ReLU(),
            nn.Linear(50, len(config.source_domains))
        )

    def forward(self, x, alpha=1.0):
        x = self.grl(x, alpha)
        return self.net(x)

class EvidentialHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, config.num_classes)

    def forward(self, x):
        evidence = torch.exp(self.fc(x))  # Ensures positive evidence
        return evidence + 1e-6  # Add epsilon for numerical stability
    


#  [5] FIXED-style Mixup (Adapted from [3] concept)
def fixed_mixup(x1, x2, y1, y2, d1, d2):
    """Mix samples from same class but different domains"""
    if y1 == y2 and d1 != d2:
        lam = np.random.beta(0.2, 0.2)
        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, y1, lam
    return None



#  [6] Loss Functions
class EvidentialLoss(nn.Module):
    def forward(self, alpha, targets):
        S = torch.sum(alpha, dim=1, keepdim=True)
        p = alpha / S

        # Bayesian risk with KL regularization (from [3] concepts)
        ce_loss = -torch.sum(targets * torch.log(p), dim=1).mean()
        kl_div = torch.digamma(S) - torch.digamma(alpha)
        kl_loss = torch.sum(kl_div, dim=1).mean()

        return ce_loss + 0.1 * kl_loss
    


#  [7] Training Loop
def train_model():
    feature_extractor = FeatureExtractor()
    domain_classifier = DomainClassifier()
    evidential_head = EvidentialHead()

    optimizer = torch.optim.Adam([
        {'params': feature_extractor.parameters()},
        {'params': domain_classifier.parameters()},
        {'params': evidential_head.parameters()}
    ], lr=config.lr)

    criterion = EvidentialLoss()

    for epoch in range(config.num_epochs):
        for images, labels, domains in train_loader:
            # Feature extraction
            features = feature_extractor(images)

            # Domain classification
            domain_preds = domain_classifier(features)
            domain_loss = nn.CrossEntropyLoss()(domain_preds, domains)

            # FIXED Mixup
            mixed_features, mixed_labels = [], []
            for i in range(0, len(images), 2):
                mix_result = fixed_mixup(features[i], features[i+1],
                                       labels[i], labels[i+1],
                                       domains[i], domains[i+1])
                if mix_result:
                    mixed_x, mixed_y, _ = mix_result
                    mixed_features.append(mixed_x)
                    mixed_labels.append(mixed_y)
                    

            # Evidential learning
            if mixed_features:
                mixed_features = torch.stack(mixed_features)
                mixed_labels = torch.tensor(mixed_labels, dtype=torch.long)
                alpha = evidential_head(mixed_features)
                evidential_loss = criterion(alpha,
                    nn.functional.one_hot(mixed_labels, config.num_classes).float())
            else:
                evidential_loss = 0

            # Total loss
            total_loss = 0.5 * domain_loss + evidential_loss

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Validation would go here
        print(f"Epoch {epoch+1}/{config.num_epochs} | Loss: {total_loss.item():.4f}")

    return feature_extractor, evidential_head



#  [8] Evaluation Metrics (From [3] and [5] concepts)
def evaluate(feature_extractor, evidential_head, loader):
    correct = 0
    total = 0
    uncertainties = []

    with torch.no_grad():
        for images, labels, _ in loader:
            features = feature_extractor(images)
            alpha = evidential_head(features)

            # Accuracy
            preds = torch.argmax(alpha, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            # Uncertainty
            S = torch.sum(alpha, dim=1)
            uncertainties.extend((config.num_classes / S).tolist())

    acc = correct / total
    avg_uncertainty = np.mean(uncertainties)
    print(f"Test Accuracy: {acc*100:.2f}% | Avg Uncertainty: {avg_uncertainty:.4f}")



# [9] Run Training
if __name__ == "__main__":
    feature_extractor, evidential_head = train_model()
    evaluate(feature_extractor, evidential_head, test_loader)