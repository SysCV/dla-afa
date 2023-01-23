"""DLA."""
from os.path import join
from typing import List, Optional, Union

import torch
from torch import nn
from torch.utils import model_zoo

from afa.model.utils.dla_utils import init_weights
from afa.utils.structures import ArgsType

WEB_ROOT = "http://dl.yf.io/dla/models"

imagenet_pretrained = {
    "dla34": "ba72cf86",
    "dla34+tricks": "24a49e58",
    "dla46_c": "2bfd52c3",
    "dla46x_c": "d761bae7",
    "dla60x_c": "b870c45c",
    "dla60": "24839fc4",
    "dla60x": "d15cacda",
    "dla102": "d94d9790",
    "dla102x": "ad62be81",
    "dla102x2": "262837b6",
    "dla169": "0914e092",
}


class BasicBlock(nn.Module):  # type: ignore
    """Basic block module."""

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1
    ) -> None:
        """Init."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(
        self, feat: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward."""
        if residual is None:
            residual = feat

        out = self.conv1(feat)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  # type: ignore
    """Bottleneck module."""

    expansion = 2

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1
    ) -> None:
        """Init."""
        super().__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(
            inplanes, bottle_planes, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottle_planes)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.conv3 = nn.Conv2d(
            bottle_planes, planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(
        self, feat: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward."""
        if residual is None:
            residual = feat

        out = self.conv1(feat)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):  # type: ignore
    """Bottleneck X module."""

    expansion = 2
    cardinality = 32

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1
    ) -> None:
        """Init."""
        super().__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(
            inplanes, bottle_planes, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottle_planes)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
            groups=cardinality,
        )
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.conv3 = nn.Conv2d(
            bottle_planes, planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(
        self, feat: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward."""
        if residual is None:
            residual = feat

        out = self.conv1(feat)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):  # type: ignore
    """Root module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        residual: bool,
    ) -> None:
        """Init."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *feat: torch.Tensor) -> torch.Tensor:
        """Forward."""
        children = feat
        x = self.conv(torch.cat(feat, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):  # type: ignore
    """Tree module."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        levels: int,
        block: nn.Module,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        level_root: bool = False,
        root_dim: int = 0,
        root_kernel_size: int = 1,
        dilation: int = 1,
        root_residual: bool = False,
    ) -> None:
        """Init."""
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(
                in_channels, out_channels, stride, dilation=dilation
            )
            self.tree2 = block(
                out_channels, out_channels, 1, dilation=dilation
            )
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
        if levels == 1:
            self.root = Root(
                root_dim, out_channels, root_kernel_size, root_residual
            )
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(
        self,
        feat: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        children: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward."""
        children = [] if children is None else children
        bottom = self.downsample(feat) if self.downsample else feat
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(feat, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):  # type: ignore
    """DLA module."""

    def __init__(
        self,
        levels: List[int],
        channels: List[int],
        num_classes: int = 1000,
        block: nn.Module = BasicBlock,
        residual_root: bool = False,
        return_levels: bool = False,
        pool_size: int = 7,
    ) -> None:
        """Init."""
        super().__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(
                3, channels[0], kernel_size=7, stride=1, padding=3, bias=False
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0]
        )
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2
        )
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
        )
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
        )

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(
            channels[-1],
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        init_weights(self.modules())

    @staticmethod
    def _make_conv_level(
        inplanes: int,
        planes: int,
        convs: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        """Make level conv net."""
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(
        self, feat: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward."""
        y = []
        x = self.base_layer(feat)
        for i in range(6):
            x = getattr(self, f"level{i}")(x)
            y.append(x)

        if not self.return_levels:
            x = self.avgpool(x)
            x = self.fc(x)
            y = x.view(x.size(0), -1)

        return y

    def load_pretrained_model(self, data_name: str, name: str) -> None:
        """Load pretrained model."""
        try:
            model_url = join(
                WEB_ROOT, data_name, f"{name}-{imagenet_pretrained[name]}.pth"
            )
        except KeyError as key_err:
            raise ValueError(
                f"{name} trained on {data_name} does not exist."
            ) from key_err
        self.load_state_dict(model_zoo.load_url(model_url))


def dla34(pretrained: Optional[str] = None, **kwargs: ArgsType) -> nn.Module:
    """DLA-34."""
    model = DLA(
        [1, 1, 1, 2, 2, 1],
        [16, 32, 64, 128, 256, 512],
        block=BasicBlock,
        **kwargs,
    )
    if pretrained is not None:
        model.load_pretrained_model(pretrained, "dla34+tricks")
    return model


def dla46_c(pretrained: Optional[str] = None, **kwargs: ArgsType) -> nn.Module:
    """DLA-46-C."""
    Bottleneck.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 2, 1],
        [16, 32, 64, 64, 128, 256],
        block=Bottleneck,
        **kwargs,
    )
    if pretrained is not None:
        model.load_pretrained_model(pretrained, "dla46_c")
    return model


def dla46x_c(
    pretrained: Optional[str] = None, **kwargs: ArgsType
) -> nn.Module:
    """DLA-X-46-C."""
    BottleneckX.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 2, 1],
        [16, 32, 64, 64, 128, 256],
        block=BottleneckX,
        **kwargs,
    )
    if pretrained is not None:
        model.load_pretrained_model(pretrained, "dla46x_c")
    return model


def dla60x_c(
    pretrained: Optional[str] = None, **kwargs: ArgsType
) -> nn.Module:
    """DLA-X-60-C."""
    BottleneckX.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 3, 1],
        [16, 32, 64, 64, 128, 256],
        block=BottleneckX,
        **kwargs,
    )
    if pretrained is not None:
        model.load_pretrained_model(pretrained, "dla60x_c")
    return model


def dla60(pretrained: Optional[str] = None, **kwargs: ArgsType) -> nn.Module:
    """DLA-60."""
    Bottleneck.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 3, 1],
        [16, 32, 128, 256, 512, 1024],
        block=Bottleneck,
        **kwargs,
    )
    if pretrained is not None:
        model.load_pretrained_model(pretrained, "dla60")
    return model


def dla60x(pretrained: Optional[str] = None, **kwargs: ArgsType) -> nn.Module:
    """DLA-X-60."""
    BottleneckX.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 3, 1],
        [16, 32, 128, 256, 512, 1024],
        block=BottleneckX,
        **kwargs,
    )
    if pretrained is not None:
        model.load_pretrained_model(pretrained, "dla60x")
    return model


def dla102(pretrained: Optional[str] = None, **kwargs: ArgsType) -> nn.Module:
    """DLA-102."""
    Bottleneck.expansion = 2
    model = DLA(
        [1, 1, 1, 3, 4, 1],
        [16, 32, 128, 256, 512, 1024],
        block=Bottleneck,
        residual_root=True,
        **kwargs,
    )
    if pretrained is not None:
        model.load_pretrained_model(pretrained, "dla102")
    return model


def dla102x(pretrained: Optional[str] = None, **kwargs: ArgsType) -> nn.Module:
    """DLA-X-102."""
    BottleneckX.expansion = 2
    model = DLA(
        [1, 1, 1, 3, 4, 1],
        [16, 32, 128, 256, 512, 1024],
        block=BottleneckX,
        residual_root=True,
        **kwargs,
    )
    if pretrained is not None:
        model.load_pretrained_model(pretrained, "dla102x")
    return model


def dla102x2(
    pretrained: Optional[str] = None, **kwargs: ArgsType
) -> nn.Module:
    """DLA-X-102 with 64 cardinality."""
    BottleneckX.cardinality = 64
    model = DLA(
        [1, 1, 1, 3, 4, 1],
        [16, 32, 128, 256, 512, 1024],
        block=BottleneckX,
        residual_root=True,
        **kwargs,
    )
    if pretrained is not None:
        model.load_pretrained_model(pretrained, "dla102x2")
    return model


def dla169(pretrained: Optional[str] = None, **kwargs: ArgsType) -> nn.Module:
    """DLA-169."""
    Bottleneck.expansion = 2
    model = DLA(
        [1, 1, 2, 3, 5, 1],
        [16, 32, 128, 256, 512, 1024],
        block=Bottleneck,
        residual_root=True,
        **kwargs,
    )
    if pretrained is not None:
        model.load_pretrained_model(pretrained, "dla169")
    return model
