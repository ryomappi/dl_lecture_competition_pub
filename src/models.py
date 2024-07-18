import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        # classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Dropout(0.25),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 64,
        p_drop: float = 0.25,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        residual = X
        
        X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X)
        X = F.gelu(self.batchnorm1(X))

        if self.in_dim == self.out_dim:
            X += residual
        else:
            X += self.conv0(residual)

        X = F.gelu(X)
        return self.dropout(X)

# https://www.jstage.jst.go.jp/article/pjsai/JSAI2020/0/JSAI2020_1I5GS202/_pdf/-char/jaを参考にした
class EEGNet(nn.Module):
    def __init__(
        self, 
        num_channels=64, 
        num_classes=2, 
        seq_len = 240,
        dropout_rate=0.5
    ):
        super(EEGNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, seq_len//2), padding=(0, 0))
        self.batchnorm1 = nn.BatchNorm2d(8)
        
        self.depthwiseconv = nn.Conv2d(8, 16, kernel_size=(64, 1), groups=8, padding=(0, 0))
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 2))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(1, 4), padding=(0, 0))
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(16 * 14, num_classes)  # 14は出力サイズの計算結果
        
    def forward(self, x):
        # 入力: (batch, channels, time)
        # リシェイプ: (batch, 1, channels, time)
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        x = self.depthwiseconv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)  # フラット化
        x = self.fc(x)
        
        return x