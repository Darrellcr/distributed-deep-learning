import os

import torch
import torch.distributed as dist
from torch import nn, optim
from torchvision import models

def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return rank, world_size


def cleanup():
    dist.destroy_process_group()


def train():
    rank, world_size = setup()

    # Load a predefined ResNet model
    model = models.resnet50(pretrained=True)
    model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Example dummy data
    inputs = torch.randn(32, 3, 224, 224).to(rank)
    labels = torch.randint(0, 1000, (32,)).to(rank)

    # Forward & Backward Pass
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    cleanup()
    


def main():
    train()  


if __name__ == "__main__":
    main()