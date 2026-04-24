"""
Self-Pruning Neural Network on CIFAR-10
Tredence AI Engineering Internship - Case Study

Author: [Your Name]
Description:
    Implements a feed-forward neural network with learnable gated weights
    that prune themselves during training via L1 sparsity regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────
# Part 1: PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable gate_scores.
    Each weight has a corresponding gate (sigmoid of gate_score) that
    multiplies the weight during the forward pass. Gates close to 0
    effectively prune the weight, making the network sparse.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight
        # Initialized near 1 so gates start open (sigmoid(2) ≈ 0.88)
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 2.0))

        # Weight initialization (Xavier uniform)
        nn.init.xavier_uniform_(self.weight)

    def get_gates(self) -> torch.Tensor:
        """Return gate values in [0, 1] via sigmoid of gate_scores."""
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Compute gates = sigmoid(gate_scores)
          2. pruned_weights = weight * gates  (element-wise)
          3. Output = x @ pruned_weights.T + bias
        Gradients flow through both `weight` and `gate_scores` automatically.
        """
        gates = self.get_gates()                         # shape: (out, in)
        pruned_weights = self.weight * gates             # element-wise product
        return F.linear(x, pruned_weights, self.bias)   # standard linear op


# ─────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (32x32 RGB → 10 classes).
    All linear layers use PrunableLinear so weights can be gated to zero.
    """

    def __init__(self):
        super().__init__()
        # Input: 32*32*3 = 3072
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)   # 10 CIFAR-10 classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)           # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def prunable_layers(self):
        """Yield all PrunableLinear sub-layers."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module


# ─────────────────────────────────────────────
# Part 2: Sparsity Regularization Loss
# ─────────────────────────────────────────────

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    Compute L1 norm of all gate values across every PrunableLinear layer.

    Why L1 encourages sparsity:
        The L1 penalty is |gate| for each gate. Gradients of L1 w.r.t. a
        positive gate are constant (+1), meaning every active gate always
        faces a fixed push toward 0. Unlike L2 (gradient ∝ value), L1 does
        not weaken as gates shrink — it keeps pushing until they reach 0.
        This property drives many gates to *exactly* zero, creating true
        sparsity rather than merely small values.
    """
    total = torch.tensor(0.0, requires_grad=True)
    for layer in model.prunable_layers():
        gates = layer.get_gates()          # values in (0, 1)
        total = total + gates.sum()        # L1 = sum of absolute values (all positive)
    return total


def total_loss(logits, targets, model, lam):
    """Total Loss = CrossEntropyLoss + λ * SparsityLoss"""
    ce_loss = F.cross_entropy(logits, targets)
    sp_loss = sparsity_loss(model)
    return ce_loss + lam * sp_loss, ce_loss.item(), sp_loss.item()


# ─────────────────────────────────────────────
# Part 3: Data Loading
# ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),   # CIFAR-10 mean
                             (0.2023, 0.1994, 0.2010)),  # CIFAR-10 std
    ])
    train_set = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    test_set  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, lam, device):
    model.train()
    total_ce, total_sp, correct, total = 0.0, 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss, ce, sp = total_loss(logits, labels, model, lam)
        loss.backward()
        optimizer.step()

        total_ce += ce * images.size(0)
        total_sp += sp * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_ce / total, total_sp / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return correct / total


def compute_sparsity(model, threshold=1e-2) -> float:
    """
    Percentage of weights whose gate value is below the threshold.
    A gate < threshold is considered 'pruned'.
    """
    pruned, total = 0, 0
    with torch.no_grad():
        for layer in model.prunable_layers():
            gates = layer.get_gates()
            pruned += (gates < threshold).sum().item()
            total  += gates.numel()
    return 100.0 * pruned / total if total > 0 else 0.0


def run_experiment(lam, epochs=15, device='cpu', batch_size=128):
    print(f"\n{'='*50}")
    print(f"  Training with λ = {lam}")
    print(f"{'='*50}")

    train_loader, test_loader = get_cifar10_loaders(batch_size)
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        ce, sp, train_acc = train_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            sparsity = compute_sparsity(model)
            print(f"  Epoch {epoch:2d} | CE: {ce:.4f} | SP: {sp:.4f} "
                  f"| TrainAcc: {train_acc*100:.1f}% | Sparsity: {sparsity:.1f}%")

    test_acc  = evaluate(model, test_loader, device)
    sparsity  = compute_sparsity(model)
    print(f"\n  → Final Test Accuracy : {test_acc*100:.2f}%")
    print(f"  → Final Sparsity Level: {sparsity:.2f}%")

    return model, test_acc, sparsity


# ─────────────────────────────────────────────
# Gate Distribution Plot
# ─────────────────────────────────────────────

def plot_gate_distribution(model, lam, filename="gate_distribution.png"):
    """
    Plot histogram of all gate values in the best model.
    A successful prune shows a large spike at 0 and another cluster away from 0.
    """
    all_gates = []
    with torch.no_grad():
        for layer in model.prunable_layers():
            gates = layer.get_gates().cpu().numpy().flatten()
            all_gates.extend(gates.tolist())

    all_gates = np.array(all_gates)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(all_gates, bins=100, color='steelblue', edgecolor='navy', alpha=0.85)
    ax.set_title(f"Gate Value Distribution  (λ = {lam})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Gate Value (sigmoid of gate_score)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.axvline(x=0.01, color='red', linestyle='--', label='Prune threshold (0.01)')
    ax.legend()
    ax.set_yscale('log')        # log scale so the 0-spike and the tail are both visible
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\nGate distribution plot saved → {filename}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Three lambda values: low, medium, high
    lambda_configs = [
        ("Low",    1e-5),
        ("Medium", 1e-4),
        ("High",   1e-3),
    ]

    results = []
    best_model, best_lam, best_acc = None, None, 0.0

    for label, lam in lambda_configs:
        model, test_acc, sparsity = run_experiment(lam, epochs=15, device=device)
        results.append((label, lam, test_acc * 100, sparsity))
        # Track best by accuracy
        if test_acc > best_acc:
            best_acc   = test_acc
            best_model = model
            best_lam   = lam

    # ── Results Table ──────────────────────────────
    print("\n\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Label':<8} {'Lambda':<12} {'Test Acc (%)':<15} {'Sparsity (%)'}")
    print("-"*60)
    for label, lam, acc, sp in results:
        print(f"  {label:<8} {lam:<12} {acc:<15.2f} {sp:.2f}")
    print("="*60)

    # ── Gate Distribution Plot ─────────────────────
    if best_model is not None:
        plot_gate_distribution(best_model, best_lam)
