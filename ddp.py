import os

import torch
import torch.distributed as dist
from torch import nn, optim
from torchvision import models

def setup():
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    
    world_size = dist.get_world_size()

    return local_rank, global_rank, world_size


def cleanup():
    dist.destroy_process_group()


def train():
    local_rank, global_rank, world_size = setup()

    # Load a predefined ResNet model
    model = models.resnet50(pretrained=True)
    model = model.to(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Example dummy data
    inputs = torch.randn(32, 3, 224, 224).to(local_rank)
    labels = torch.randint(0, 1000, (32,)).to(local_rank)

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