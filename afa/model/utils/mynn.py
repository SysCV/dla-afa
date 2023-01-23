"""AFA customized modules."""
from inplace_abn import ABN, InPlaceABNSync
from torch import nn

from afa.config import cfg


def bn_relu(ch: int, inplace: bool = True) -> nn.Sequential:
    """Batch Norm plus ReLU layer."""
    if cfg["inplace_abn"]:
        norm_layer = nn.Sequential(InPlaceABNSync(ch))
    else:
        norm_layer = nn.Sequential(
            nn.BatchNorm2d(ch), nn.ReLU(inplace=inplace)
        )

    return norm_layer


def initialize_weights(*models: nn.Sequential) -> None:
    """Initialize Model Weights."""
    for model in models:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (ABN, nn.BatchNorm2d)):
                m.weight.data.uniform_()
                if m.bias is not None:
                    m.bias.data.zero_()
