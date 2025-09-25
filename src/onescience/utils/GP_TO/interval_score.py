import torch


def interval_score(Yu, Yl, Y, alpha=0.05):
    out = Yu - Yl
    out += (Y > Yu).to(torch.int64) * 2 / alpha * (Y - Yu)
    out += (Y < Yl).to(torch.int64) * 2 / alpha * (Yl - Y)

    accuracy = torch.sum(out > (Yu - Yl)).to(torch.float64)
    accuracy /= len(out)

    return torch.mean(out), accuracy
