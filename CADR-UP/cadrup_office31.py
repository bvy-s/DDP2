import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import random
from collections import defaultdict

# --------------------------
# 1️⃣ Reproducibility
# --------------------------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# --------------------------
# 2️⃣ Domains & Config
# --------------------------
domains = ['amazon', 'dslr', 'webcam']
n_classes = 31
n_domains = len(domains)
train_domains = ['amazon', 'dslr']
domain_to_idx = {d: i for i, d in enumerate(train_domains)}
test_domain = 'webcam'

# --------------------------
# 3️⃣ Transforms & Loaders
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

domain_datasets = {d: datasets.ImageFolder(f'~/bhavyaaa/datasets/office31/{d}', transform=transform) for d in domains}
domain_loaders = {d: DataLoader(domain_datasets[d], batch_size=32, shuffle=True, num_workers=4) for d in domains}

# --------------------------
# 4️⃣ Feature Extractor for MMD
# --------------------------
class FeatureStats(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

feature_extractor = FeatureStats().cuda().eval()

def extract_domain_features(loader, feature_extractor, n_batches=10):
    features = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            feats = feature_extractor(x.cuda()).cpu().numpy()
            features.append(feats)
    return np.concatenate(features, axis=0)

train_domain_features = {d: extract_domain_features(domain_loaders[d], feature_extractor) for d in train_domains}

def compute_mmd(X, Y):
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    return XX.mean() + YY.mean() - 2 * XY.mean()

domain_graph = np.zeros((len(train_domains), len(train_domains)))
for i, d1 in enumerate(train_domains):
    for j, d2 in enumerate(train_domains):
        if i != j:
            domain_graph[i, j] = compute_mmd(train_domain_features[d1], train_domain_features[d2])
        else:
            domain_graph[i, j] = 0.0

print("Domain Graph (MMD):")
print(domain_graph)

# --------------------------
# 5️⃣ CADRUP Model
# --------------------------
class SharedBackbone(nn.Module):
    def __init__(self, out_dim=256, dropout_p=0.3):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, out_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        return x

class CausalHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.fc(x)

class StyleHead(nn.Module):
    def __init__(self, in_dim, n_domains):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_domains)
    def forward(self, x):
        return self.fc(x)

class Expert(nn.Module):
    def __init__(self, in_dim, n_classes, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(128, n_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class GatingNetwork(nn.Module):
    def __init__(self, in_dim, n_experts):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_experts)
    def forward(self, x, uncertainties):
        gate_logits = self.fc(x)
        gate_logits = gate_logits - uncertainties  # lower uncertainty → higher weight
        return F.softmax(gate_logits, dim=1)

class CADRUP(nn.Module):
    def __init__(self, n_classes, n_domains, n_experts=3, feat_dim=256, dropout_p=0.5):
        super().__init__()
        self.backbone = SharedBackbone(out_dim=feat_dim, dropout_p=dropout_p)
        self.causal_head = CausalHead(feat_dim, n_classes)
        self.style_head = StyleHead(feat_dim, n_domains)
        self.experts = nn.ModuleList([Expert(feat_dim, n_classes, dropout_p=dropout_p) for _ in range(n_experts)])
        self.gating = GatingNetwork(feat_dim, n_experts)
        self.n_experts = n_experts

    def forward(self, x, mc_dropout=False, n_mc=5):
        feats = self.backbone(x)
        causal_logits = self.causal_head(feats)
        style_logits = self.style_head(feats)

        if mc_dropout:
            self.train()
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

            expert_outputs = []
            for _ in range(n_mc):
                outputs = torch.stack([F.softmax(expert(feats), dim=1) for expert in self.experts])  # [n_experts, batch, n_classes]
                expert_outputs.append(outputs)
            expert_outputs = torch.stack(expert_outputs)  # [n_mc, n_experts, batch, n_classes]

            mean_outputs = expert_outputs.mean(0)
            std_outputs = expert_outputs.std(0).mean(-1)  # [n_experts, batch]
            uncertainties = std_outputs.T  # [batch, n_experts]
            experts_stack = mean_outputs.permute(1, 0, 2)  # [batch, n_experts, n_classes]
        else:
            experts_stack = torch.stack([F.softmax(expert(feats), dim=1) for expert in self.experts], dim=1)  # [batch, n_experts, n_classes]
            uncertainties = torch.zeros(x.size(0), self.n_experts, device=x.device)  # no uncertainty

        gates = self.gating(feats, uncertainties)  # [batch, n_experts]
        final_output = (gates.unsqueeze(-1) * experts_stack).sum(dim=1)

        return final_output, causal_logits, style_logits, uncertainties

# --------------------------
# 6️⃣ Pairwise style loss helper
# --------------------------
def compute_pairwise_style_loss(model, x1, x2):
    feats1 = model.backbone(x1)
    feats2 = model.backbone(x2)
    style_logits1 = model.style_head(feats1).mean(0)  # [n_domains]
    style_logits2 = model.style_head(feats2).mean(0)  # [n_domains]
    return F.mse_loss(style_logits1, style_logits2)

# --------------------------
# 7️⃣ Train loop with graph regularization
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CADRUP(n_classes=n_classes, n_domains=n_domains).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
ce_loss = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for d in train_domains:
        for x, y in domain_loaders[d]:
            x, y = x.to(device), y.to(device)
            domain_labels = torch.full((x.size(0),), domain_to_idx[d], dtype=torch.long, device=device)
            out, causal_logits, style_logits, uncertainties = model(x)

            loss_cls = ce_loss(out, y)
            loss_causal = ce_loss(causal_logits, y)
            loss_style = ce_loss(style_logits, domain_labels)

            loss = loss_cls + 0.5 * loss_causal + 0.2 * loss_style

            other_domains = [dd for dd in train_domains if dd != d]
            if other_domains:
                d2 = random.choice(other_domains)
                try:
                    x2, _ = next(iter(domain_loaders[d2]))
                except StopIteration:
                    continue
                x2 = x2.to(device)
                style_loss = compute_pairwise_style_loss(model, x, x2)
                i, j = domain_to_idx[d], domain_to_idx[d2]
                sim_weight = np.exp(-domain_graph[i, j])
                loss += 0.1 * sim_weight * style_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"Epoch {epoch} done. Loss = ", loss)

# --------------------------
# 8️⃣ Evaluate
# --------------------------
model.eval()
correct, total = 0, 0
uncertainties_list = []
with torch.no_grad():
    for x, y in domain_loaders[test_domain]:
        x, y = x.to(device), y.to(device)
        out, _, _, uncertainties = model(x, mc_dropout=True, n_mc=10)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        uncertainties_list.append(uncertainties.cpu().numpy())

print(f"Test accuracy on {test_domain}: {correct / total:.4f}")
print(f"Mean uncertainty: {np.mean(np.concatenate(uncertainties_list)):.4f}")
