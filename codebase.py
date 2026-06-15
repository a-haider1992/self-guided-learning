import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import pickle
from sklearn.mixture import GaussianMixture
from loss import CrossEntropyCosineLoss
from cosface import CosFace
import pdb
from gradcamV3 import GradCAM
from tqdm import tqdm
import random
import shutil
import logging
import warnings
import cv2
import math
import pandas as pd
from sklearn.decomposition import PCA
from torchvision.models import resnet50, resnet18, resnet34

warnings.filterwarnings("ignore", category=FutureWarning, message=".*non-full backward hook.*")

logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

LATENT_VCETOR_DIR = "Latent_vectors"

# ============================================================
# Dataset definitions (Oxford, Stanford, CIFAR10)
# ============================================================
class OxfordPetsDataset(Dataset):
    def __init__(self, data_dir, annotation_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        labels = []
        with open(annotation_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                base = parts[0]
                label_name = parts[1]
                img_name = base + ".jpg"

                labels.append(label_name)
                self.samples.append((img_name, label_name, base))

        self.classes = sorted(list(set(labels)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label_name, filename = self.samples[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[label_name]
        return image, label, filename

class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None, extensions=None):
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = extensions or (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        self.class_to_idx = {}
        self.class_names = []
        self.classes = 0
        self.samples = []

        self._build_index()

    def _is_valid_file(self, filename):
        return filename.lower().endswith(self.extensions)

    def _build_index(self):
        self.class_names = sorted(
            d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.classes = len(self.class_names)

        for cls_name in self.class_names:
            cls_dir = os.path.join(self.root_dir, cls_name)
            cls_idx = self.class_to_idx[cls_name]
            for fname in os.listdir(cls_dir):
                if self._is_valid_file(fname):
                    path = os.path.join(cls_dir, fname)
                    stem = os.path.splitext(fname)[0]
                    self.samples.append((path, cls_idx, stem))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label, filename = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, filename

# ============================================================
# CNN / Fine-tune models
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=(2,3))  # global avg pooling
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y

class PetCNN(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()
        
        # Block 1
        self.conv1a = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.se1 = SEBlock(32)

        # Block 2
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.se2 = SEBlock(64)

        # Block 3
        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.se3 = SEBlock(128)

        # Block 4
        self.conv4a = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4b = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.se4 = SEBlock(256)

        # GAP + FC
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.se1(x)
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.se2(x)
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.se3(x)
        x = self.pool3(x)

        # Block 4
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.se4(x)
        x = self.pool4(x)

        # GAP + FC
        x = self.gap(x).view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

class FineTuneModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.encoder_output = None

    def forward(self, x):
        z = self.base_model(x)
        self.encoder_output = z
        x = self.fc1(z)
        x = self.fc2(x)
        return x

    def get_encoder_output(self):
        return self.encoder_output

# ============================================================
# Losses
# ============================================================
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1.0, gamma=2.0):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         ce = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-ce)
#         loss = self.alpha * (1 - pt) ** self.gamma * ce
#         return loss.mean()

# ============================================================
# Self-Paced Curriculum Learning
# ============================================================
class SelfPacedCurriculum:
    def __init__(self, lambda_init=0.5, mu=0.1, lambda_max=10.0):
        self.lambda_sp = lambda_init
        self.mu = mu
        self.lambda_max = lambda_max

    def compute_v(self, losses):
        return (losses <= self.lambda_sp).float()

    def update_lambda(self):
        if self.lambda_sp < self.lambda_max:
            self.lambda_sp += self.mu

# ================================================================
# Active Self-paced learning
# ================================================================
class ActiveSelfPacedLearning:
    def __init__(
        self,
        lambda_init=0.5,
        lambda_growth=1.2,
        active_ratio=0.2,   # fraction of samples selected via uncertainty
        device="cuda"
    ):
        self.lambda_val = lambda_init
        self.lambda_growth = lambda_growth
        self.active_ratio = active_ratio
        self.device = device

    def update_lambda(self):
        """Gradually include harder samples"""
        self.lambda_val *= self.lambda_growth

    def compute_self_paced_mask(self, losses):
        """Easy samples (low loss)"""
        return (losses <= self.lambda_val)

    def compute_uncertainty(self, logits):
        """Entropy-based uncertainty"""
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy

    def compute_active_mask(self, logits):
        """Select top-k uncertain samples"""
        uncertainty = self.compute_uncertainty(logits)
        k = int(len(uncertainty) * self.active_ratio)

        if k == 0:
            return torch.zeros_like(uncertainty, dtype=torch.bool)

        _, indices = torch.topk(uncertainty, k)
        mask = torch.zeros_like(uncertainty, dtype=torch.bool)
        mask[indices] = True
        return mask

    def compute_combined_mask(self, losses, logits):
        """Union of easy + informative samples"""
        spl_mask = self.compute_self_paced_mask(losses)
        active_mask = self.compute_active_mask(logits)
        return spl_mask | active_mask

    def weighted_loss(self, losses, logits):
        mask = self.compute_combined_mask(losses, logits)

        if mask.sum() == 0:
            return losses.mean()

        return (losses * mask.float()).sum() / mask.sum()

def generate_latent_vectors(
    model,
    input_batch_tensor,
    input_batch_heatmaps,
    filenames,
    target_layer,
    predictions,
    ground_truth,
    save_root
):
    """
    Safe latent vector extraction.
    Does NOT affect training graph, optimizer, or model gradients.
    """

    latent_vecs = []
    if os.path.exists(os.path.join(save_root, "incorrect_cases")):
        shutil.rmtree(os.path.join(save_root, "incorrect_cases"))
    # --------------------------
    # Create hook (temporary)
    # --------------------------
    def hook_fn(module, inp, out):
        latent_vecs.append(out.detach().cpu())  # always detach, always CPU

    layer = dict(model.named_modules())[target_layer]
    hook_handle = layer.register_forward_hook(hook_fn)

    # --------------------------
    # Entire function runs without touching autograd graph
    # --------------------------
    model_was_training = model.training
    model.eval()  # no dropout/Bn randomness

    with torch.no_grad():
        pred_classes = predictions.detach().cpu().argmax(dim=1)  # DETACHED

        # Iterate samples in batch
        for input_tensor, heatmap, fname, gt, pred in zip(
            input_batch_tensor, input_batch_heatmaps, filenames, ground_truth, pred_classes
        ):
            latent_vecs.clear()

            gt = int(gt)
            pred = int(pred)

            # Choose save folder
            class_dir = os.path.join(
                save_root,
                "correct_cases" if gt == pred else "incorrect_cases",
                str(gt),
            )
            os.makedirs(class_dir, exist_ok=True)

            # ---------------------------------------------------
            # Prepare masked input safely
            # ---------------------------------------------------
            input_tensor = input_tensor.detach()  # no graph
            device = input_tensor.device
            C, H, W = input_tensor.shape

            heatmap_tensor = torch.as_tensor(heatmap, device=device, dtype=torch.float32)

            # pdb.set_trace()

            # Resize to input resolution
            heatmap_tensor = F.interpolate(
                heatmap_tensor.unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            # Normalize heatmap
            max_val = heatmap_tensor.max().item()
            if max_val < 1e-8:
                continue
            heatmap_tensor = heatmap_tensor / max_val

            # Apply heatmap as mask
            masked_input = (input_tensor * heatmap_tensor.unsqueeze(0)).detach()

            if torch.all(masked_input == 0):
                continue

            # ---------------------------------------------------
            # Forward pass ONLY to capture latent vectors
            # ---------------------------------------------------
            _ = model(masked_input.unsqueeze(0))  # no grad, no graph

            if len(latent_vecs) == 0:
                continue

            vec = latent_vecs[0].squeeze().numpy()

            # Save only valid vectors
            if np.all(np.isfinite(vec)):
                save_path = os.path.join(class_dir, f"{fname}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(vec, f)

    # --------------------------
    # CLEANUP
    # --------------------------
    hook_handle.remove()

    # Restore model training mode
    if model_was_training:
        model.train()

def gmm_em(
    X,
    n_components,
    max_iter=100,
    tol=1e-4,
    reg_covar=1e-3,  # increased default for stability
    device="cuda"
):
    """
    Gaussian Mixture Model using EM (Full Covariance, stable implementation).

    Args:
        X: torch.Tensor (N, D)
        n_components: number of mixture components
        max_iter: EM iterations
        tol: log-likelihood convergence tolerance
        reg_covar: covariance regularization
        device: "cuda" or "cpu"

    Returns:
        labels: (N,)
        means: (K, D)
        covariances: (K, D, D)
        weights: (K,)
    """
    X = X.to(device)
    N, D = X.shape
    K = n_components

    # ---- Initialize ----
    indices = torch.randperm(N)[:K]
    means = X[indices].clone()
    covariances = torch.stack([torch.eye(D, device=device) for _ in range(K)])
    weights = torch.ones(K, device=device) / K

    log_likelihood_old = None

    for iteration in range(max_iter):

        # ---- E-Step ----
        log_probs = []

        for k in range(K):
            diff = X - means[k]

            # Add regularization to ensure positive-definite
            cov = covariances[k] + reg_covar * torch.eye(D, device=device)

            # Ensure cov is positive definite
            eigvals = torch.linalg.eigvalsh(cov)
            if torch.any(eigvals <= 0):
                cov += 1e-3 * torch.eye(D, device=device)

            inv_cov = torch.linalg.inv(cov)
            log_det = torch.logdet(cov)

            exponent = -0.5 * torch.sum(diff @ inv_cov * diff, dim=1)

            log_prob = (
                torch.log(weights[k] + 1e-10)  # avoid log(0)
                - 0.5 * D * math.log(2 * math.pi)
                - 0.5 * log_det
                + exponent
            )
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs, dim=1)  # (N, K)

        log_sum = torch.logsumexp(log_probs, dim=1, keepdim=True)
        responsibilities = torch.exp(log_probs - log_sum)

        log_likelihood = log_sum.sum()

        # ---- Check Convergence ----
        if log_likelihood_old is not None:
            if torch.abs(log_likelihood - log_likelihood_old) < tol:
                break
        log_likelihood_old = log_likelihood

        # ---- M-Step ----
        Nk = responsibilities.sum(dim=0)  # (K,)

        # Reinitialize empty clusters
        for k in range(K):
            if Nk[k] < 1e-6:
                # Tiny cluster: reset mean, covariance, weight
                means[k] = X[torch.randint(0, N, (1,))]
                covariances[k] = torch.eye(D, device=device)
                weights[k] = 1.0 / N
                Nk[k] = 1.0
       
        weights = Nk / N
        means = (responsibilities.T @ X) / Nk.unsqueeze(1)

        new_covariances = []
        for k in range(K):
            diff = X - means[k]
            weighted = responsibilities[:, k].unsqueeze(1) * diff
            cov = (weighted.T @ diff) / Nk[k]
            cov += reg_covar * torch.eye(D, device=device)
            new_covariances.append(cov)

        covariances = torch.stack(new_covariances)

    labels = responsibilities.argmax(dim=1)
    return labels, means, covariances, weights

def plot_clusters(latent_arr, labels, case_name, class_label, epoch, result_dir):
    """
    PCA-based 2D cluster visualization
    """
    if latent_arr.shape[0] < 2:
        return  # nothing to plot

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(latent_arr)

    plt.figure(figsize=(5, 5))

    if labels is None:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    else:
        for lab in np.unique(labels):
            idx = labels == lab
            plt.scatter(
                reduced[idx, 0],
                reduced[idx, 1],
                label=f"Cluster {lab}",
                alpha=0.7
            )
        plt.legend()

    plt.title(f"{case_name} | Class {class_label} | Epoch {epoch}")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.tight_layout()

    os.makedirs(result_dir, exist_ok=True)
    fname = f"epoch_{epoch}_{case_name}_class_{class_label}.png"
    plt.savefig(os.path.join(result_dir, fname))
    plt.close()

def build_consensus(root_dir, num_classes, result_dir, epoch):
    """
    Build class-wise consensus vectors from saved latent vectors.
    Returns:
        { "Correct": {class_label: [vec]}, "Incorrect": {...} }
    """
    consensus = {"Correct": {}, "Incorrect": {}}

    for case_type in ["correct_cases", "incorrect_cases"]:
        case_path = os.path.join(root_dir, case_type)
        if not os.path.isdir(case_path):
            continue

        for class_label in os.listdir(case_path):
            class_dir = os.path.join(case_path, class_label)
            if not os.path.isdir(class_dir):
                continue

            latent_list = []

            # Load vectors for this class
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if not os.path.isfile(fpath):
                    continue

                with open(fpath, "rb") as f:
                    vec = pickle.load(f)

                if not np.all(np.isfinite(vec)):
                    logging.info(f"Ignoring invalid NaN/Inf vector: {fpath}")
                    continue

                if np.linalg.norm(vec) < 1e-8:
                    logging.info(f"Ignoring near-zero vector: {fpath}")
                    continue

                latent_list.append(vec.astype(np.float64))  # <-- FIX #1: force float64

            if len(latent_list) == 0:
                logging.info(f"No valid latent vectors in {class_dir}")
                continue

            latent_arr = np.stack(latent_list, axis=0)  # (N, D)
            N, D = latent_arr.shape

            latent_tensor = torch.from_numpy(latent_arr).float().cuda()

            # --------------------------
            # SAFE CONSENSUS GENERATION
            # --------------------------

            if N == 1:
                # Only one vector → consensus = the vector
                consensus_vec = latent_arr[0]

            else:
                # FIX #2: Choose a safe number of components
                safe_k = min(4, N - 1)  # must be <= N-1 for stability,
                if safe_k < 1:
                    safe_k = 1

                try:
                    labels, means, covs, weights = gmm_em(
                        latent_tensor,
                        n_components=safe_k,
                        max_iter=100,
                        device="cuda"
                    )

                    largest_cluster = torch.bincount(labels).argmax()
                    consensus_vec = means[largest_cluster].cpu().numpy()
                    # gmm = GaussianMixture(
                    #     n_components=safe_k,
                    #     covariance_type="full",
                    #     reg_covar=1e-4,              # <-- FIX #3: stabilize covariance
                    #     max_iter=200,
                    #     random_state=42
                    # )

                    # labels = gmm.fit_predict(latent_arr)
                    # centroids = gmm.means_

                    # # majority cluster
                    # largest_cluster = np.bincount(labels).argmax()
                    # consensus_vec = centroids[largest_cluster]

                except ValueError as e:
                    logging.warning(
                        f"GMM failed for class {class_label} ({case_type}). "
                        f"Falling back to mean. Error: {e}"
                    )

                    # FIX #4: fallback to simple mean
                    consensus_vec = latent_arr.mean(axis=0)

                # Store
                group = "Correct" if case_type == "correct_cases" else "Incorrect"
                consensus[group].setdefault(class_label, [])
                if case_type == "correct_cases":
                    consensus[group][class_label].append(consensus_vec)
                    # plot_clusters(
                    #     latent_arr=latent_arr,
                    #     labels=labels,
                    #     case_name=case_type,
                    #     class_label=class_label,
                    #     epoch=epoch,
                    #     result_dir=result_dir
                    # )
                else:
                    consensus[group][class_label] = latent_list

    return consensus

def compute_lossV2(consensus, batch_targets, eps=1e-8):
    device = batch_targets.device
    batch_size = batch_targets.size(0)

    total_loss = torch.zeros((), device=device)
    total_pairs = 0

    for i in range(batch_size):
        cls = str(batch_targets[i].item())

        if cls not in consensus["Correct"]:
            continue
        if cls not in consensus["Incorrect"]:
            continue
        if len(consensus["Correct"][cls]) == 0:
            continue
        if len(consensus["Incorrect"][cls]) == 0:
            continue

        correct_vec = torch.from_numpy(consensus["Correct"][cls][0]).to(device).view(-1)
        correct_vec = F.normalize(correct_vec, dim=0, eps=eps)

        for incorrect_vec_np in consensus["Incorrect"][cls]:
            incorrect_vec = torch.from_numpy(incorrect_vec_np).to(device).view(-1)
            incorrect_vec = F.normalize(incorrect_vec, dim=0, eps=eps)

            dist = torch.norm(correct_vec - incorrect_vec, p=2)
            dist = torch.clamp(dist, 0.0, 3.0)

            total_loss += dist
            total_pairs += 1

    if total_pairs > 0:
        total_loss = total_loss / total_pairs

    return total_loss

# ============================================================
# Training function (trainV2 with SPL)
# ============================================================
def trainV2(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    num_classes,
    latent_dir,
    result_dir,
    start_epoch,
    gradcam,
    epochs=5,
    lambda_sim=1.0,
    model_path="latest_model_to_load.pth",
    curriculum=None,
    aspl=None
):
    ce_loss_history = []
    total_loss_history = []

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        ce_running = 0.0
        start = time.time()

        time_model = time_gradcam = time_latent = time_consensus = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for x, y, filenames in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            t0 = time.time()
            outputs = model(x)
            time_model += time.time() - t0

            # ---------------- CE loss with SPL ----------------
            ce_losses = criterion(outputs, y)  # per-sample, reduction='none'
            
            # 1. SPCL
            # if curriculum is not None and epoch > 5:
            #     with torch.no_grad():
            #         v_star = curriculum.compute_v(ce_losses)
            #     ce_loss = (v_star * ce_losses).sum() / max(v_star.sum(), 1)
            # else:
            #     ce_loss = ce_losses.mean()

            # 2. ASPL
            #  epoch >= 10 (shallow aspl) epoch >= 5 (deep aspl)
            if epoch >= 10:
                ce_loss = aspl.weighted_loss(ce_losses, outputs)
            else:
                ce_loss = ce_losses.mean()

            # ce_loss = ce_losses.mean()
            ce_running += ce_loss.item()

            # ---------------- GradCAM + latent + sim_loss (optional) ----------------
            # if epoch % 10 == 0 and epoch >= 10 and epoch <= 30 and lambda_sim > 0.0:
            if epoch % 2 == 0 and epoch >= 1 and epoch <= 8 and lambda_sim > 0.0:
                t0 = time.time()
                cams = gradcam(x, y)
                time_gradcam += time.time() - t0

                t0 = time.time()
                latent_vectors = generate_latent_vectors(
                    model=model,
                    input_batch_tensor=x,
                    input_batch_heatmaps=cams,
                    filenames=filenames,
                    target_layer="fc1",
                    predictions=outputs,
                    ground_truth=y,
                    save_root=latent_dir
                )
                time_latent += time.time() - t0

                t0 = time.time()
                consensus = build_consensus(latent_dir, num_classes, result_dir, epoch)
                sim_loss = compute_lossV2(consensus, y)
                time_consensus += time.time() - t0
            else:
                sim_loss = 0.0

            # ---------------- Total loss ----------------
            loss = ce_loss + lambda_sim * sim_loss if lambda_sim > 0.0 else ce_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(
                ce=ce_loss.item(),
                sim=sim_loss if isinstance(sim_loss, float) else sim_loss.item(),
                total=loss.item()
            )

        # ---------------- End of epoch ----------------
        avg_loss = running_loss / len(train_loader)
        ce_loss_epoch_avg = ce_running / len(train_loader)
        total_loss_history.append(avg_loss)
        ce_loss_history.append(ce_loss_epoch_avg)

        if curriculum is not None:
            curriculum.update_lambda()

        print(f"Epoch {epoch+1}: CE={ce_loss_epoch_avg:.4f}, Total={avg_loss:.4f}")


# ============================================================
# Evaluation
# ============================================================
def evaluate(model, test_loader, criterion, device, class_names=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.mean().item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return accuracy, f1_macro

def save_model_final(model, checkpoint):
    torch.save(model.state_dict(), checkpoint)
    print(f"Saved model: {checkpoint}")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss

# ============================================================
# Main experiment logic
# ============================================================
def run_one_seed(seed, lambda_w, result_dir):
    set_seed(seed)

    # ----------------- Paths -------------------------
    data_dir = "datasets/oxford_pets/images"
    train_file = "datasets/oxford_pets/annotations/trainval.txt"
    test_file = "datasets/oxford_pets/annotations/test.txt"

    set_seed(seed)
    print("Running seed:", seed)

    # ----------------- data -----------------
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # ----- Strong augmentation for training -----
    dogs_train_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # zoom + crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # optional, if dataset allows
        transforms.RandomRotation(degrees=20),  # small rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),  # sometimes remove color
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),  # translation + shear
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # Cutout-style
    ])

    train_ds = OxfordPetsDataset(data_dir, train_file, transform)
    test_ds = OxfordPetsDataset(data_dir, test_file, transform)

    # train_ds = StanfordDogsDataset("CUB_200_2011/images/train", transform)
    # test_ds  = StanfordDogsDataset("CUB_200_2011/images/val", transform)

    # train_ds = StanfordDogsDataset("Stanford_dogs_dataset/train", dogs_train_aug)
    # test_ds  = StanfordDogsDataset("Stanford_dogs_dataset/val", dogs_train_aug)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=1)

    # num_classes = train_ds.classes
    num_classes = len(train_ds.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # base_model = resnet50(pretrained=True)
    model = PetCNN(num_classes).to(device)
    # for p in base_model.parameters():
    #     p.requires_grad = False
    # for p in base_model.layer4[-1].conv3.parameters():
    #     p.requires_grad = True
    # for p in base_model.layer4[-1].bn3.parameters():
    #     p.requires_grad = True
    # for p in base_model.fc.parameters():
    #     p.requires_grad = True

    # model = FineTuneModel(base_model, num_classes).to(device)
    # last_conv = model.base_model.layer4[-1].conv3

    last_conv = model.conv4b
    gradcam = GradCAM(model=model, target_layer=last_conv)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = FocalLoss()

    latent_dir = os.path.join(LATENT_VCETOR_DIR, str(seed))
    start_epoch = 0
    # for deep
    curriculum = SelfPacedCurriculum(lambda_init=0.3, mu=0.05, lambda_max=4.0)
    # for shallow
    # curriculum = SelfPacedCurriculum(lambda_init=1.0, mu=0.4, lambda_max=7.0)

    # Deep
    aspl = ActiveSelfPacedLearning(lambda_init=0.6, active_ratio=0.2)
    # Shallow
    # aspl = ActiveSelfPacedLearning(lambda_init=0.7, active_ratio=0.4)


    # epochs = 8 (deep spcl)
    # epcohs = 6 (shallow spcl)
    # epoch = 15 (shallow aspl)
    trainV2(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        num_classes,
        latent_dir,
        result_dir,
        start_epoch,
        gradcam,
        epochs=15,
        lambda_sim=lambda_w,
        curriculum=curriculum,
        aspl=aspl
    )

    save_model_final(model, f'Shallow_ASPL_FL_Oxford_{lambda_w}_seed={seed}.pth')

    acc, f1 = evaluate(model, test_loader, criterion, device)
    return seed, acc, f1

# ============================================================
# Main CLI entry
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()

    seed = args.seed
    mode = args.mode
    lambda_w = 1.0 if mode == "consensus" else 0.0
    result_dir = f"Shallow_ASPL_FL_Oxford_{mode}"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"seed_{seed}.txt")

    if os.path.exists(result_file):
        print(f"Seed {seed} already completed. Skipping.")
        return

    seed, acc, f1 = run_one_seed(seed, lambda_w, result_dir)

    with open(result_file, "w") as f:
        f.write(f"seed={seed}\n")
        f.write(f"accuracy={acc:.4f}\n")
        f.write(f"f1={f1:.4f}\n")

    print(f"Seed {seed} completed successfully.")

if __name__ == "__main__":
    main()
