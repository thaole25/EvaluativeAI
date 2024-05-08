import torch
import torch.nn as nn


class PosthocLinearCBM(nn.Module):
    def __init__(self, concept_bank, backbone_name, idx_to_class=None, n_classes=5):
        """
        PosthocCBM Linear Layer.
        Takes an embedding as the input, outputs class-level predictions using only concept margins.
        Args:
            concept_bank (ConceptBank)
            backbone_name (str): Name of the backbone, e.g. clip:RN50.
            idx_to_class (dict, optional): A mapping from the output indices to the class names. Defaults to None.
            n_classes (int, optional): Number of classes in the classification problem. Defaults to 5.
        """
        super(PosthocLinearCBM, self).__init__()
        # Get the concept information from the bank
        self.backbone_name = backbone_name
        self.cavs = concept_bank.vectors
        self.intercepts = concept_bank.intercepts
        self.norms = concept_bank.norms
        self.names = concept_bank.concept_names.copy()
        self.n_concepts = self.cavs.shape[0]

        self.n_classes = n_classes
        # Will be used to plot classifier weights nicely
        # self.idx_to_class = idx_to_class if idx_to_class else {i: i for i in range(self.n_classes)}

        # A single linear layer will be used as the classifier
        self.classifier = nn.Linear(self.n_concepts, self.n_classes)

    def compute_dist(self, emb):
        # Computing the geometric margin to the decision boundary specified by CAV.
        margins = (torch.matmul(self.cavs, emb.T) + self.intercepts) / (self.norms)
        return margins.T

    def forward(self, emb, return_dist=False):
        x = self.compute_dist(emb)
        out = self.classifier(x)
        if return_dist:
            return out, x
        return out

    def forward_projs(self, projs):
        return self.classifier(projs)

    def trainable_params(self):
        return self.classifier.parameters()

    def classifier_weights(self):
        return self.classifier.weight

    def set_weights(self, weights, bias):
        self.classifier.weight.data = torch.tensor(weights).to(
            self.classifier.weight.device
        )
        self.classifier.bias.data = torch.tensor(bias).to(self.classifier.weight.device)
        return 1
