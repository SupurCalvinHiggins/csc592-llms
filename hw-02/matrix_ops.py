import sys

import torch


def main():
    a = torch.arange(2 * 2).reshape(2, 2)
    print(a)
    b = torch.arange(2 * 2).reshape(2, 2)
    print(b)
    c = a * b
    print(c)
    e = torch.arange(2 * 3).reshape(2, 3)
    print(e)
    f = torch.matmul(a, e)  # matrix multiplication
    print(f)
    f1 = a @ e
    print(f1)
    f1 = torch.unsqueeze(f, dim=0)  # add a dimension in the beginning
    print(f1.shape)
    f1t = torch.transpose(f1, 1, 2)
    print(f1t.shape)

    f1t = torch.transpose(f1, 1, 2)
    print(f1t)
    tensor1 = torch.randn(10, 3, 4)
    tensor2 = torch.randn(10, 4, 5)
    res = torch.matmul(tensor1, tensor2)
    print(res.shape)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
