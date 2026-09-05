import torch

x = torch.tensor(
    [
        [1.0, 0.0],  # Token 0
        [1.0, 1.0],  # Token 1
    ]
)

# 直接让Q、K、V等于输入
q = x
k = x
v = x

# 每个Query与每个Key 计算点积
scores = q @ k.T

# 将每行分数转换成注意力权重
weights = torch.softmax(scores, dim=-1)

# 使用权重汇总 Value 向量
output = weights @ v

print("x:")
print(x)

print("\nscores:")
print(scores)

print("\nweights:")
print(weights)

print("\noutput:")
print(output)
