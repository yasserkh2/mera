# %% [markdown]
# Note about the source:
#
# > This notebook is copied from https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb
# which is mentioned in official repo https://github.com/facebookresearch/detr. The notebook contains `DETRdemo` and `DETRdemoNoDict`. Both are the same model; the only difference is that in `DETRdemoNoDict`, the output is a List instead of a Dictionary. We will use `DETRdemoNoDict` because it is better for exporting. In this current notebook, we only kept necessary parts to export the  `DETRdemoNoDict` as onnx.
# > Python 3.8 and PyTorch 1.12.1 was used for this export code.
# %%

import sys
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

torch.set_grad_enabled(False)


# ## DETR
# Here is a minimal implementation of DETR:
class DETRdemoNoDict(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(
        self,
        num_classes,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
        )

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )

        # propagate through the transformer
        h = self.transformer(
            pos + 0.1 * h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1)
        ).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return [self.linear_class(h), self.linear_bbox(h).sigmoid()]


detr_nodict = DETRdemoNoDict(num_classes=91)
print("Loading pretrained weight ...")
state_dict = torch.hub.load_state_dict_from_url(
    url="https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth",
    map_location="cpu",
    check_hash=True,
)
detr_nodict.load_state_dict(state_dict)
detr_nodict.eval()


# Input to the model
x = torch.randn(1, 3, 400, 600, requires_grad=False)
torch_out = detr_nodict(x)

out_path = Path("source_model_files/detr_600x400_nodict.onnx")
out_path.parent.mkdir(parents=True, exist_ok=True)
# Export the model
print("Exporting to ONNX ...")
torch.onnx.export(
    detr_nodict,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    out_path,  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=13,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
)

print(f"Success! Model exported to {out_path}")
