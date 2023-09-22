import torch
import torch.nn as nn


class WrapperModel(nn.Module):
    def __init__(self, model, classifier_name="classifier"):
        super().__init__()
        self.model = model
        self.classifier = get_named_submodule(self.model, classifier_name)
        set_named_submodule(self.model, classifier_name, nn.Identity())
        self.backbone_out = None
    
    def forward(self, x):
        self.backbone_out = self.model(x)
        return self.classifier(self.backbone_out)


def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)
