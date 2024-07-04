# neuralforge/multimodal/fusion.py
import torch
import torch.nn as nn

class MultiModalFusion(nn.Module):
    def __init__(self, modality_dims, fusion_dim):
        super(MultiModalFusion, self).__init__()
        self.modality_projections = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in modality_dims
        ])
        self.fusion = nn.Linear(fusion_dim * len(modality_dims), fusion_dim)

    def forward(self, modalities):
        projected = [proj(mod) for proj, mod in zip(self.modality_projections, modalities)]
        concatenated = torch.cat(projected, dim=1)
        fused = self.fusion(concatenated)
        return fused

class MultiModalClassifier(nn.Module):
    def __init__(self, modality_dims, fusion_dim, num_classes):
        super(MultiModalClassifier, self).__init__()
        self.fusion = MultiModalFusion(modality_dims, fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, modalities):
        fused = self.fusion(modalities)
        return self.classifier(fused)

# Usage example:
# modality_dims = [224*224*3, 128]  # Image and text dimensions
# fusion_dim = 256
# num_classes = 10
# model = MultiModalClassifier(modality_dims, fusion_dim, num_classes)
# image_input = torch.randn(32, 224*224*3)
# text_input = torch.randn(32, 128)
# output = model([image_input, text_input])