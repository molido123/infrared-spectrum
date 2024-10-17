import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from sklearn.metrics import mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConv(nn.Module):
    """
    This class represents a double convolutional layer.

    It consists of two convolutional layers with batch normalization and ReLU activation.
    """

    def __init__(self, in_ch: int, out_ch: int):
        """
        Initialize the DoubleConv layer.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """
        super(DoubleConv, self).__init__()

        # Define the convolutional layers with batch normalization and ReLU activation
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class ChannelAttention(nn.Module):
    """
    This class implements a channel attention module.

    It computes attention weights for each channel in the input tensor.

    Args:
        in_planes (int): The number of input channels.
        ratio (int, optional): The reduction ratio for the number of channels. Defaults to 4.
    """

    def __init__(self, in_planes: int, ratio: int = 4) -> None:
        """
        Initialize the ChannelAttention module.

        Args:
            in_planes (int): The number of input channels.
            ratio (int, optional): The reduction ratio for the number of channels. Defaults to 4.
        """
        super(ChannelAttention, self).__init__()

        # Define the average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Define the maximum pooling layer
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Define the first fully connected layer
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)

        # Define the ReLU activation function
        self.relu1 = nn.ReLU()

        # Define the second fully connected layer
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        # Define the sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial attention module.

    Applies spatial attention to the input tensor.
    """

    def __init__(self, kernel_size=3):
        """
        Initialize the SpatialAttention module.

        Args:
            kernel_size (int): Size of the convolutional kernel. Must be 3 or 7.
        """
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'

        # Set padding based on kernel size
        padding = 3 if kernel_size == 7 else 1

        # Define the convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=2,  # Number of input channels
            out_channels=1,  # Number of output channels
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        # Define the sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the SpatialAttention module.

        This function takes an input tensor `x` and applies spatial attention to it.
        It computes the average and maximum values across the second dimension of `x`
        and concatenates them along the second dimension. It then passes the concatenated
        tensor through a convolutional layer `self.conv1` and applies a sigmoid activation
        function to the output.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: The output tensor after applying spatial attention and convolutional layer.
        """
        # Compute the average and maximum values across the second dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate the average and maximum values along the second dimension
        x = torch.cat([avg_out, max_out], dim=1)

        # Pass the concatenated tensor through the convolutional layer
        x = self.conv1(x)

        # Apply the sigmoid activation function to the output
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=4, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class PLE(nn.Module):
    def __init__(self, n_expert=1, ple_hidden_dim=125, hidden_dim=[256, 64], dropouts=[0.1, 0.1], hidden_size=125,
                 num_task=3, output_size=1, expert_activation=None, hidden_size_gate=125, num_encoder_layers=2,
                 nhead=2):
        super(PLE, self).__init__()

        # 定义 Transformer Encoder 部分
        self.embedding_dim = 32  # 定义嵌入维度
        self.embedding_layer = nn.Conv1d(1, self.embedding_dim, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=nhead, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 定义卷积层和池化层（同之前的模型结构）
        self.conv1 = DoubleConv(self.embedding_dim, 32)  # 将输入调整为 embedding_dim 以适配 Transformer 的输出
        self.att1 = CBAM(32, 1, 3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = DoubleConv(32, 64)

        self.upsam1 = nn.ConvTranspose1d(64, 32, 3, stride=2)

        self.att2 = CBAM(64, 1, 3)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = DoubleConv(64, 128)

        self.upsam2 = nn.ConvTranspose1d(128, 32, 5, stride=4)

        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = DoubleConv(128, 256)

        self.upsam3 = nn.ConvTranspose1d(256, 32, 3, stride=9, padding=2)

        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = DoubleConv(256, 512)

        self.up6 = nn.ConvTranspose1d(512, 256, 3, stride=2)
        self.conv6 = DoubleConv(512, 256)

        self.upsam4 = nn.ConvTranspose1d(256, 32, 3, stride=9, padding=2)

        self.up7 = nn.ConvTranspose1d(256, 128, 3, stride=2)
        self.conv7 = DoubleConv(256, 128)

        self.upsam5 = nn.ConvTranspose1d(128, 32, 5, stride=4)

        self.up8 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)

        self.up9 = nn.ConvTranspose1d(64, 32, 3, stride=2)
        self.conv9 = DoubleConv(128, 32)
        self.conv10 = DoubleConv(64, 32)
        self.last_layer = nn.Conv1d(32, 1, 1)

        self.loss_fun = nn.L1Loss()
        self.num_task = num_task
        self.weights = torch.nn.Parameter(torch.ones(self.num_task).float())

        self.expert_activation = expert_activation
        self.experts = nn.ModuleList()

        for i in range(self.num_task + 1):  # +1 for shared experts
            expert_list = []
            for j in range(n_expert):
                expert_list.append(nn.Linear(hidden_size, ple_hidden_dim))
            self.experts.append(nn.ModuleList(expert_list))

        self.shared_experts = nn.ModuleList([nn.Linear(hidden_size, ple_hidden_dim) for _ in range(n_expert)])

        # Define task-specific towers
        self.towers = nn.ModuleList()
        for i in range(self.num_task):
            tower = nn.ModuleList()
            hid_dim = [ple_hidden_dim * (n_expert + 1)] + hidden_dim
            for j in range(len(hid_dim) - 1):
                tower.add_module('tower_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                tower.add_module('tower_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                tower.add_module('tower_dropout_{}'.format(j), nn.Dropout(dropouts[j]))
            tower.add_module('task_last_layer', nn.Linear(hid_dim[-1], output_size))
            self.towers.append(tower)

    def forward(self, x):
        x = self.embedding_layer(x)

        # 调整输入数据格式为 (sequence_length, batch_size, embedding_dim)
        x = x.permute(2, 0, 1)  # 调整维度顺序为 (sequence_length, batch_size, channels)

        # 使用 Transformer Encoder 进行特征提取
        x = self.transformer_encoder(x)

        # 调整回卷积层输入格式 (batch_size, embedding_dim, sequence_length)
        x = x.permute(1, 2, 0)

        # 开始进入卷积部分
        c1 = self.conv1(x)
        c11 = c1
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        part4 = self.upsam1(c2)
        c22 = c2
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        part5 = self.upsam2(c3)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        part6 = self.upsam3(c4)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        part2 = self.upsam4(c6)

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        part3 = self.upsam5(c7)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c22], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)

        concate = torch.cat([part2, part3, up_9, c11], dim=1)
        c9 = self.conv9(concate)
        x1 = self.last_layer(c9)
        x1 = x1.view(x1.size(0), -1)

        task_outputs = []
        for i in range(self.num_task):
            experts_output = []
            for expert in self.experts[i]:
                experts_output.append(expert(x1))
            shared_output = torch.stack([expert(x1) for expert in self.shared_experts], dim=2)

            task_expert_output = torch.cat(experts_output + [shared_output.mean(dim=2)], dim=1)
            for mod in self.towers[i]:
                task_expert_output = mod(task_expert_output)
            task_outputs.append(task_expert_output)

        return task_outputs
    def get_last_shared_layer(self):
        return self.last_layer

