# -*- coding: utf-8 -*-
# @Time    : 2024/5/13 14:19
# @Author  : seenli


import torch

# queries 的形状：(2, 3, 4)
queries = torch.ones(2, 3, 4)

# keys 的形状：(2, 5, 4)
keys = torch.ones(2, 5, 4) * 2

print("queries.shape", queries.shape)
print("queries", queries)
print("keys.shape", keys.shape)
print("keys", keys)


queries = queries.unsqueeze(2)  # 形状变为 (2, 3, 1, 4)
keys = keys.unsqueeze(1)        # 形状变为 (2, 1, 5, 4)
print("queries.shape", queries.shape)
print("keys.shape", keys.shape)

features = queries + keys  # 形状为 (2, 3, 5, 4)
print("features.shape", features.shape)
print("features", features)

print()