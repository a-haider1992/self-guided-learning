import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None

    def _save_features(self, module, inp, out):
        self.features = out   # DO NOT detach

    def __call__(self, x, target_classes):
        """
        Grad-CAM that is SAFE for training:
        - no backward()
        - no grad accumulation
        - sim_loss can backprop
        """

        # ----------------------------
        # Hook only forward activations
        # ----------------------------
        hook = self.target_layer.register_forward_hook(self._save_features)

        # Forward pass (same graph as training)
        outputs = self.model(x)

        # Target scores
        B = outputs.size(0)
        scores = outputs[torch.arange(B), target_classes]

        # Compute gradients W.R.T. feature maps
        grads = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=self.features,
            retain_graph=True,     # needed for CE + sim backward
            create_graph=True      # REQUIRED if sim_loss trains model
        )[0]

        # Remove hook immediately
        hook.remove()

        # ----------------------------
        # Compute CAM
        # ----------------------------
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        # Normalize per sample (keeps graph)
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam   # Tensor, NOT numpy
