# -*- coding: utf-8 -*-
# @Time    : 2024/5/13 00:50
# @Author  : seenli


import torch
from torch import nn
from d2l import torch as d2l


class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)


n_train = 50  # 训练样本数
x_train, sorted_indices = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本
print("x_train.shape", x_train.shape)
print("x_train", x_train)
print("sorted_indices", sorted_indices)

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
print("y_train", y_train)
x_test = torch.arange(0, 5, 0.1)  # 测试样本
print("x_test", x_test)
y_truth = f(x_test)  # 测试样本的真实输出
print("y_truth", y_truth)
n_test = len(x_test)  # 测试样本数
print("n_test", n_test)

def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    d2l.plt.show()

# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

print("X_tile.shape", X_tile.shape)
print("X_tile", X_tile)
print("Y_tile.shape", Y_tile.shape)
print("Y_tile", Y_tile)
print("keys.shape", keys.shape)
print("keys", keys)
print("values.shape", values.shape)
print("values", values)

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

epoch_list = []
l_sum_list = []

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    epoch_list.append(epoch+1)
    l_sum_list.append(l.sum())
    animator.add(epoch + 1, float(l.sum()))
    # d2l.plt.show()


# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)