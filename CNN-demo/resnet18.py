import torch
import torch.nn as nn
import torch.nn.functional as F

# 输入输出通道数/尺寸不变的残差模块
class BasicBlock1(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        """
        通道数/尺寸不变的残差块（stride固定为1）
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数（必须等于输入通道数）
        """
        # 为何要写(BasicBlock1,self)，这是为了兼容python2。豆包：https://www.doubao.com/thread/w344fd77daaaf8fa0
        super(BasicBlock1, self).__init__() 
        # 断言校验：强制输入输出通道数一致（避免维度不匹配）
        assert in_channels == out_channels, "BasicBlock1要求输入输出通道数必须相等！"

        # 第一层卷积：3×3，步长固定1，padding1（尺寸/通道不变）
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        # 第二层卷积：3×3，步长固定1，padding1（尺寸/通道不变）
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # inplace=True节省内存

    def forward(self, x_input):  # 避免用input（Python内置关键字）
        x = x_input
        x = self.norm1(self.conv1(x))
        x = self.relu(x)
        x = self.norm2(self.conv2(x))
        # 残差相加 + 最终ReLU
        x = x + x_input
        x = self.relu(x)
        return x

# 输入输出通道数变化、尺寸减半的残差模块
class BasicBlock2(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        """
        通道数翻倍、尺寸减半的残差块（stride固定为2）
        :param in_channels: 输入通道数（如64）
        :param out_channels: 输出通道数（如128，必须是输入的2倍）
        """
        super(BasicBlock2, self).__init__() 
        # 断言校验：强制输出通道数是输入的2倍（符合ResNet18下采样规则）
        assert out_channels == 2 * in_channels, "BasicBlock2要求输出通道数是输入的2倍！"
        # Shortcut分支：1×1卷积调整通道数+下采样（尺寸减半）
        self.conv0 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False
        )
        self.norm0 = nn.BatchNorm2d(out_channels)
        # 主分支第一层卷积：3×3，步长2（尺寸减半），padding1，通道数翻倍
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        # 主分支第二层卷积：3×3，步长1，padding1（尺寸/通道不变）
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # inplace=True节省内存

    def forward(self, x_input):
        x = x_input
        x = self.norm1(self.conv1(x))
        x = self.relu(x)
        x = self.norm2(self.conv2(x))
        # 残差相加 + 最终ReLU
        x = x + self.norm0(self.conv0(x_input))
        x = self.relu(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=10) -> None:
        super(ResNet18, self).__init__()
        # 输入层：7×7卷积 + BN + ReLU + 3×3最大池化（ResNet18标准配置）
        self.conv0 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.norm0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 中间层：按ResNet18标准命名（layer1-layer4，每层2个BasicBlock）
        # layer1：2个通道不变的块（64→64，尺寸56×56）
        self.layer1 = nn.Sequential(
            BasicBlock1(64, 64),
            BasicBlock1(64, 64)
        )
        # layer2：1个下采样块 + 1个通道不变块（64→128，尺寸28×28）
        self.layer2 = nn.Sequential(
            BasicBlock2(64, 128),
            BasicBlock1(128, 128)
        )
        # layer3：1个下采样块 + 1个通道不变块（128→256，尺寸14×14）
        self.layer3 = nn.Sequential(
            BasicBlock2(128, 256),
            BasicBlock1(256, 256)
        )
        # layer4：1个下采样块 + 1个通道不变块（256→512，尺寸7×7）
        self.layer4 = nn.Sequential(
            BasicBlock2(256, 512),
            BasicBlock1(512, 512)
        )
        # 输出层：自适应全局平均池化 + 全连接（适配任意输入尺寸）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 权重初始化
        # self.modules()的作用是递归遍历当前模型中所有的子模块， PyTorch 的nn.Module类内置的核心方法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入层
        x = self.maxpool0(self.relu0(self.norm0(self.conv0(x))))

        # 中间层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 输出层
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # (batch_size, 512, 1, 1) → (batch_size, 512)
        x = self.fc(x)

        return x
'''
# ===================== 测试代码（验证前向传播无报错） =====================
if __name__ == "__main__":
    # 1. 实例化模型（适配CIFAR10，num_classes=10）
    model = ResNet18(num_classes=10)
    print("ResNet18模型实例化成功！")
    
    # 2. 模拟输入（CIFAR10：batch_size=2，3通道，32×32）
    dummy_input = torch.randn(2, 3, 32, 32)
    print(f"\n模拟输入尺寸：{dummy_input.shape}")
    
    # 3. 前向传播测试
    with torch.no_grad():  # 禁用梯度，加快测试
        output = model(dummy_input)
    
    # 4. 验证输出尺寸
    print(f"模型输出尺寸：{output.shape}")  # 应为(2, 10)，符合分类数要求
    
    # 5. 打印各层输出尺寸（验证维度匹配）
    def print_layer_output(model, x):
        x = model.conv0(x)
        x = model.norm0(x)
        x = model.relu0(x)
        x = model.maxpool0(x)
        print(f"输入层后尺寸：{x.shape}")  # (2,64,8,8)
        
        x = model.layer1(x)
        print(f"layer1后尺寸：{x.shape}")  # (2,64,8,8)
        x = model.layer2(x)
        print(f"layer2后尺寸：{x.shape}")  # (2,128,4,4)
        x = model.layer3(x)
        print(f"layer3后尺寸：{x.shape}")  # (2,256,2,2)
        x = model.layer4(x)
        print(f"layer4后尺寸：{x.shape}")  # (2,512,1,1)
    
    print("\n各层输出尺寸验证：")
    print_layer_output(model, dummy_input)
'''