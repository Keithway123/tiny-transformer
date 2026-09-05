import torch
import math
from torch import nn


class SingleHeadCausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()  # 初始化父类nn.Module 的内部管理机制，使PyTorch能等级子模块和参数。

        self.embedding_dim = embedding_dim

        self.q_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        self.k_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        self.v_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, sequence_length, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim},"
                f"but received {embedding_dim}"
            )

        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.k_projection(x)

        scores = q @ k.transpose(-2, -1)
        scaled_scores = scores / math.sqrt(self.embedding_dim)

        causal_mask = (
            torch.tril(  # 一个bool Tenser提供位置规则 用来描述那些位置允许或禁止关注。
                torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                    device=x.device,
                )
            )
        )
        print(causal_mask)

        masked_scores = scaled_scores.masked_fill(  # 按照规则修改scores
            ~causal_mask,
            float("-inf"),
        )
        print(masked_scores)

        attention_weights = torch.softmax(masked_scores, dim=-1)
        attention_output = attention_weights @ v

        return attention_output, attention_weights


def main() -> None:
    torch.manual_seed(0)

    batch_size = 2
    sequence_size = 4
    embedding_dim = 8

    x = torch.randn(
        batch_size,
        sequence_size,
        embedding_dim,
    )

    attention = SingleHeadCausalSelfAttention(embedding_dim)

    output, weights = attention(x)

    print("output.shape:", output.shape)
    print("weights.shape:", weights.shape)
    print("\nweights[0]:")
    print(weights[0])


if __name__ == "__main__":
    main()
