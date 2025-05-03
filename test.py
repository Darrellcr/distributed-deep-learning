import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

# Initialize process group
dist.init_process_group(backend="nccl")  # Use "nccl" for GPUs
device_mesh = init_device_mesh(
    "cuda", mesh_shape=(3, 2), mesh_dim_names=("dp", "pp")
)  # Initialize device mesh


# Check GPU availability
if torch.cuda.is_available():
    device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
else:
    device = torch.device("cpu")

print(f"I am worker {dist.get_rank()} of {dist.get_world_size()} on {device}!")
device_name = torch.cuda.get_device_name(dist.get_rank() % torch.cuda.device_count())
print(device_name)

# Create a tensor on GPU and perform all_reduce
a = torch.tensor([dist.get_rank()], device=device)
print(device_mesh['dp'])
dist.all_reduce(a, group=device_mesh.get_group("pp"))
print(f"all_reduce output (on {device}) = {a.item()}")

dist.destroy_process_group()
