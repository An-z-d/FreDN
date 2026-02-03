import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletTransform2D(nn.Module):
    def __init__(self, wave_type='db4', level=1, device = 'cuda:0'):
        super(WaveletTransform2D, self).__init__()
        self.wave_type = wave_type
        self.level = level
        self.device = device

        # 获取小波滤波器
        wavelet = pywt.Wavelet(wave_type)
        H0, G0, _, _ = wavelet.filter_bank

        # 注册为buffer，确保设备一致性
        self.register_buffer('H0', torch.tensor(H0, dtype=torch.float32).to(self.device))
        self.register_buffer('G0', torch.tensor(G0, dtype=torch.float32).to(self.device))

        # 预先生成四个方向的滤波器
        self.filters = nn.ParameterList([
            nn.Parameter(torch.ger(self.H0, self.H0).view(1, 1, len(H0), len(H0)), requires_grad=False),  # LL
            nn.Parameter(torch.ger(self.G0, self.H0).view(1, 1, len(H0), len(H0)), requires_grad=False),  # LH
            nn.Parameter(torch.ger(self.H0, self.G0).view(1, 1, len(H0), len(H0)), requires_grad=False),  # HL
            nn.Parameter(torch.ger(self.G0, self.G0).view(1, 1, len(H0), len(H0)), requires_grad=False)]).to(self.device)  # HH

    def decompose_one_level(self, x):
        """执行单层二维小波分解"""
        k = len(self.H0)
        _, _, h, w = x.shape

        # 检查输入尺寸和滤波器长度是否满足要求
        if k % 2 != 0 or h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Filter length {k} or input size ({h}, {w}) must be even")

        p = int((k / 2) - 1)  # 计算填充量
        x_pad = F.pad(x, (p, p, p, p), mode='reflect')

        # 执行四个方向的卷积
        coeffs = []
        for fil in self.filters:
            coeff = F.conv2d(x_pad, fil, stride=2)
            coeffs.append(coeff)

        return coeffs  # [LL, LH, HL, HH]

    def forward(self, x):
        """
        执行多层二维小波变换
        输入: x [Batch, Channel, Height, Width]
        输出: [approx, detail_level, ..., detail_1]
             每个detail为字典 {'ad': LH, 'da': HL, 'dd': HH}
        """
        current_approx = x
        details = []

        for _ in range(self.level):
            # 单层分解
            LL, LH, HL, HH = self.decompose_one_level(current_approx)

            # 保存当前层的细节系数
            details.insert(0, {'ad': LH, 'da': HL, 'dd': HH})  # 插入列表头部保持层级顺序
            current_approx = LL  # 更新近似系数用于下一层分解

        return [current_approx] + details


class WaveletReconstruction2D(nn.Module):
    def __init__(self, wave_type='db4', device = 'cuda:0'):
        super(WaveletReconstruction2D, self).__init__()
        self.device = device
        self.grad = False
        wavelet = pywt.Wavelet(wave_type)
        _, _, H1, G1 = wavelet.filter_bank

        # 注册为buffer，确保设备一致性
        self.register_buffer('H1', torch.tensor(H1, dtype=torch.float32).to(self.device))
        self.register_buffer('G1', torch.tensor(G1, dtype=torch.float32).to(self.device))

        # 预先生成四个方向的滤波器
        self.filters = nn.ParameterList([
            nn.Parameter(torch.ger(self.H1, self.H1).view(1, 1, len(H1), len(H1)), requires_grad=self.grad),  # LL
            nn.Parameter(torch.ger(self.H1, self.G1).view(1, 1, len(H1), len(H1)), requires_grad=self.grad),  # LH
            nn.Parameter(torch.ger(self.G1, self.H1).view(1, 1, len(H1), len(H1)), requires_grad=self.grad),  # HL
            nn.Parameter(torch.ger(self.G1, self.G1).view(1, 1, len(H1), len(H1)), requires_grad=self.grad)]).to(self.device)  # HH

    def reconstruct_one_level(self, coeffs):
        """
        执行单层二维小波重构
        输入: coeffs [LL, LH, HL, HH]
        输出: 重构后的信号
        """
        k = len(self.H1)
        # 检查输入尺寸和滤波器长度是否满足要求
        # 检查输入尺寸和滤波器长度是否满足要求
        if k % 2 != 0 :
            raise ValueError(f"Filter length {k} must be even")
        #p = 0
        #p = int((k / 2) - 1)  # 计算填充量
        #coeffs_pad = [F.pad(co, (p, p, p, p), mode='reflect') for co in coeffs]
        LL_up, LH_up, HL_up, HH_up = [F.conv_transpose2d(coeffs[i], self.filters[i], stride=2, padding=3) for i in range(4)]
        # 将四个方向的信号相加
        reconstructed = LL_up + LH_up + HL_up + HH_up

        return reconstructed

    def forward(self, coeff_list):
        """
        执行多层二维小波重构
        输入: [approx, detail_level, ..., detail_1]
             每个detail为字典 {'ad': LH, 'da': HL, 'dd': HH}
        输出: x [Batch, Channel, Height, Width]
        """
        level = len(coeff_list) - 1
        current_approx = coeff_list[0]

        for i in range(level):
            # 获取当前层的细节系数
            detail = coeff_list[i + 1]
            LH, HL, HH = detail['ad'], detail['da'], detail['dd']

            # 执行单层重构
            coeffs = [current_approx, LH, HL, HH]
            current_approx = self.reconstruct_one_level(coeffs)

        return current_approx
