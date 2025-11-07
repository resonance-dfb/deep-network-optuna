import math
import sys

import torch
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets, transforms
import scienceplots
import torch.nn.functional as F

# 检测设备
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#Relu:Train Acc: 0.8000, Test Acc: 0.7200
# Gelu:sigma = 0.5 Train Acc: 0.8700, Test Acc: 0.8150
# Silu: sigma = 0.4 Train Acc: 0.8800, Test Acc: 0.8050
# mish : sigma = 0.5 Train Acc: 0.8600, Test Acc: 0.8200
# elu : sigma = 0.4 Train Acc: 0.8800, Test Acc: 0.8200

# ==========================================================
# 设置随机种子 - 确保实验可重复
# ==========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # 如果使用多GPU
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


# 设置随机种子
set_seed(42)


# ==========================================================
# 1. Load MNIST dataset (local, auto-download)
# ==========================================================
def load_mnist_local():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    x_tr = train_set.data.float().unsqueeze(1) / 255.0
    y_tr = train_set.targets
    x_te = test_set.data.float().unsqueeze(1) / 255.0
    y_te = test_set.targets

    # 移动到检测到的设备
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)

    return x_tr, y_tr, x_te, y_te


# 自定义激活函数类
class CustomActivation(nn.Module):
    def __init__(self, activation_type, sigma_init=None):
        super(CustomActivation, self).__init__()
        self.activation_type = activation_type

        # 为不同激活函数设置默认的sigma初值
        default_sigmas = {
            'gelu': 0.5,  # GELU的标准参数
            'silu': 0.4,  # SiLU/Swish
            'mish': 0.5,  # Mish
            'elu': 0.4,  # ELU
        }

        # 如果用户没有指定sigma_init，使用默认值
        if sigma_init is None:
            sigma_init = default_sigmas.get(activation_type, 0.3)

        # 为所有需要sigma的激活函数创建参数
        if activation_type in ['gelu', 'silu', 'elu', 'mish']:
            self.sigma = nn.Parameter(torch.tensor(sigma_init))
            print(f"Initialized {activation_type} with sigma = {sigma_init}")
        else:
            self.sigma = None

    def forward(self, input):
        if self.activation_type == 'relu':
            return F.relu(input)
        elif self.activation_type == 'gelu':
            # 标准GELU变体
            return input / 2 * (1 + torch.erf(input / math.sqrt(2) / self.sigma))

        elif self.activation_type == 'silu':
            # SiLU 激活函数 (也称为 Swish)
            return input * torch.sigmoid(input / self.sigma)

        elif self.activation_type == 'mish':
            # Mish: x * tanh(ln(1 + exp(x/sigma)))
            softplus = torch.log(1 + torch.exp(input / self.sigma))
            return input * torch.tanh(softplus)

        elif self.activation_type == 'elu':
            # ELU: x/sigma if x>0, (exp(x/sigma)-1) if x<=0
            condition = input > 0
            return torch.where(condition,
                               input / self.sigma,
                               (torch.exp(input / self.sigma) - 1))

        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")


class CNN_HERO(nn.Module):
    def __init__(self, a_type, n_iterations):
        super(CNN_HERO, self).__init__()
        self.a_type = a_type
        self.activation = CustomActivation(a_type)  # 统一使用CustomActivation

        # 其余层定义保持不变...
        self.layer1 = nn.Sequential(*([nn.Conv2d(1, 4, 3), self.activation, nn.BatchNorm2d(4)]))
        self.layer2 = nn.Sequential(*([nn.Conv2d(4, 8, 3), self.activation, nn.BatchNorm2d(8)]))
        # ... 其他层

        self.layer3 = nn.Sequential(*([nn.Conv2d(8, 16, 3), self.activation, nn.BatchNorm2d(16)]))

        self.layer4 = nn.Sequential(*([nn.Linear(5 * 5 * 16, 256), self.activation, nn.BatchNorm1d(256)]))

        self.layer5 = nn.Sequential(*([nn.Linear(256, 10)]))

        for m in self.modules():
            self.weight_init(m)

        self.pool_layer = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

        self.sigmas = th.zeros((7, n_iterations)).to(device)
        self.cost = []
        self.train_score = []  # 训练准确率
        self.test_score = []  # 测试准确率
        self.MI = th.zeros((n_iterations, 5, 2)).to(device)

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer2_p = self.pool_layer(layer2)
        layer3 = self.layer3(layer2_p)
        layer3_p = self.pool_layer(layer3)

        N, C, H, W = layer3_p.size()

        layer4 = self.layer4(layer3_p.view(N, -1))
        layer5 = self.layer5(layer4)

        return [layer5, layer4, layer3, layer2, layer1]

    def train_model(self, x, y, model, optimizer=th.optim.SGD):
        optimizer = optimizer(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        model.train()

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output[0], y)
        loss.backward()
        optimizer.step()

        self.cost.append(loss.item())
        return

    def predict(self, x, y, model, is_train=True):
        model.eval()
        output = model(x)
        y_hat = th.argmax(self.softmax(output[0]), 1)
        score = th.eq(y, y_hat).sum().float() / x.size(0)
        if is_train:
            self.train_score.append(score.item())
        else:
            self.test_score.append(score.item())
        return

    def dist_mat(self, x):
        try:
            x = th.from_numpy(x)
        except TypeError:
            x = x

        if len(x.size()) == 4:
            x = x.view(x.size()[0], -1)

        dist = th.norm(x[:, None] - x, dim=2)
        return dist

    def entropy(self, *args):
        for idx, val in enumerate(args):
            if idx == 0:
                k = val.clone()
            else:
                k *= val

        k /= k.trace()
        eigv = th.linalg.eigh(k)[0].abs()
        return -(eigv * (eigv.log2())).sum()

    def kernel_mat(self, x, k_x, k_y, sigma=None, epoch=None, idx=None):
        d = self.dist_mat(x)
        if sigma is None:
            # 修改这里：使用.to(device)而不是.cuda()
            if epoch > 40:
                sigma_vals = th.linspace(0.1, 10 * d.mean().item(), 50).to(device)
            else:
                sigma_vals = th.linspace(0.1, 10 * d.mean().item(), 75).to(device)
            L = []
            for sig in sigma_vals:
                k_l = th.exp(-d ** 2 / (sig ** 2)) / d.size(0)
                L.append(self.kernel_loss(k_x, k_y, k_l, idx))

            if epoch == 0:
                self.sigmas[idx + 1, epoch] = sigma_vals[L.index(max(L))]
            else:
                self.sigmas[idx + 1, epoch] = 0.9 * self.sigmas[idx + 1, epoch - 1] + 0.1 * sigma_vals[L.index(max(L))]

            sigma = self.sigmas[idx + 1, epoch]

        return th.exp(-d ** 2 / (sigma ** 2))

    def kernel_loss(self, k_x, k_y, k_l, idx):
        b = 1.0
        beta = [b, b, b, b, b]

        L = th.norm(k_l)
        Y = th.norm(k_y) ** beta[idx]
        X = th.norm(k_x) ** (1 - beta[idx])

        LY = th.trace(th.matmul(k_l, k_y)) ** beta[idx]
        LX = th.trace(th.matmul(k_l, k_x)) ** (1 - beta[idx])

        return 2 * th.log2((LY * LX) / (L * Y * X))

    def one_hot(self, y):
        # 修改这里：移除gpu参数，直接使用device
        try:
            y = th.from_numpy(y)
        except TypeError:
            pass

        y_1d = y
        y_hot = th.zeros((y.size(0), th.max(y).int() + 1)).to(device)

        for i in range(y.size(0)):
            y_hot[i, y_1d[i].int()] = 1

        return y_hot

    def compute_mi(self, x, y, model, current_iteration):
        model.eval()
        data = self.forward(x)
        data.reverse()
        data[-1] = self.softmax(data[-1])
        data.insert(0, x)
        data.append(self.one_hot(y))  # 现在只需要一个参数

        k_x = self.kernel_mat(data[0], [], [], sigma=th.tensor(8.0).to(device))
        k_y = self.kernel_mat(data[-1], [], [], sigma=th.tensor(0.1).to(device))

        k_list = [k_x]
        for idx_l, val in enumerate(data[1:-1]):
            k_list.append(
                self.kernel_mat(val.reshape(data[0].size(0), -1), k_x, k_y, epoch=current_iteration, idx=idx_l))
        k_list.append(k_y)

        e_list = [self.entropy(i) for i in k_list]
        j_XT = [self.entropy(k_list[0], k_i) for k_i in k_list[1:-1]]
        j_TY = [self.entropy(k_i, k_list[-1]) for k_i in k_list[1:-1]]

        for idx_mi, val_mi in enumerate(e_list[1:-1]):
            self.MI[current_iteration, idx_mi, 0] = e_list[0] + val_mi - j_XT[idx_mi]
            self.MI[current_iteration, idx_mi, 1] = e_list[-1] + val_mi - j_TY[idx_mi]

    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if self.a_type == 'relu':
                init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'leaky_relu':
                init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'tanh':
                g = init.calculate_gain('tanh')
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'sigmoid':
                g = init.calculate_gain('sigmoid')
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            else:
                # 对于其他激活函数，使用默认的Xavier初始化
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data, 0)

    def make_batches(self, N, batch_size):
        idx = random.sample(range(0, N), N)
        for i in range(0, N, batch_size):
            yield idx[i:i + batch_size]


# 主程序
x_tr, y_tr, x_te, y_te = load_mnist_local()
N = 1
batch_size_tr = 100
batch_size_te = 200
epochs = 50
tr_size = 1000
te_size = 1000  # 测试集大小
n_iterations = (tr_size // batch_size_tr) * epochs
current_iteration = 0
activation = 'elu'

# 从测试集中随机选择一部分作为验证集
te_indices = random.sample(range(len(x_te)), te_size)
x_te_subset = x_te[te_indices]
y_te_subset = y_te[te_indices]

for n in range(N):
    model = CNN_HERO(activation, n_iterations).to(device)
    for epoch in range(epochs):
        batches_tr = list(model.make_batches(tr_size, batch_size_tr))
        for idx_tr in batches_tr:
            x_tr_b, y_tr_b = x_tr[idx_tr], y_tr[idx_tr]

            # 从测试集中随机选择batch进行验证
            idx_te = random.sample(range(te_size), batch_size_te)
            x_te_b, y_te_b = x_te_subset[idx_te], y_te_subset[idx_te]

            model.train_model(x_tr_b, y_tr_b, model)
            with th.no_grad():
                # 训练集准确率
                model.predict(x_tr_b, y_tr_b, model, is_train=True)
                # 测试集准确率
                model.predict(x_te_b, y_te_b, model, is_train=False)
                model.compute_mi(x_te_b, y_te_b, model, current_iteration)
                current_iteration += 1
        print(f"Run: {n}, Epoch: {epoch}, Cost: {model.cost[-1]:.4f}, "
              f"Train Acc: {model.train_score[-1]:.4f}, Test Acc: {model.test_score[-1]:.4f}")

# 可视化结果
xy1 = model.MI.cpu().detach().numpy().reshape(1, n_iterations, 5, 2)

# 第一张图：互信息散点图
plt.figure(1, figsize=(10, 6))
plt.style.use('ggplot')
c_lab = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']

for m in range(N):
    for j in range(5):
        plt.scatter(xy1[m, :, j, 0], xy1[m, :, j, 1], cmap=c_lab[j], c=np.arange(0, xy1.shape[1], 1),
                    edgecolor=c_lab[j][:-1], s=30)

for j in range(5):
    plt.scatter(xy1[0, -1, j, 0], xy1[0, -1, j, 1], c=c_lab[j][:-1], label='Layer {}'.format(j + 1), s=30)

plt.legend(facecolor='white')
plt.xlabel('MI(X,T)')
plt.ylabel('MI(T,Y)')
plt.title('Mutual Information Scatter Plot')
plt.tight_layout()
plt.savefig('MI_.jpg', dpi=300, bbox_inches='tight')
plt.show()

# 第二张图：准确率曲线
plt.figure(2, figsize=(10, 6))
plt.style.use('ggplot')
plt.plot(model.train_score, label='Train Accuracy')
plt.plot(model.test_score, label='Test Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy over Iterations')
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_.jpg', dpi=300, bbox_inches='tight')
plt.show()

# 第三张图：Sigma曲线
plt.figure(3, figsize=(10, 6))
plt.style.use('ggplot')
plt.plot(model.sigmas.cpu().detach().numpy().T)
plt.xlabel('Iteration')
plt.ylabel('Sigma Values')
plt.title('Sigma Values over Iterations')
plt.tight_layout()
plt.savefig('sigma_values_.jpg', dpi=300, bbox_inches='tight')
plt.show()

np.savez_compressed('sigma_cnn_train_elu.npz', a=model.sigmas.cpu().detach().numpy().T)