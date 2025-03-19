import torch
import torch.distributed as dist

# Initialize process group
dist.init_process_group(backend="nccl")  # Use "nccl" for GPUs

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
else:
    device = torch.device("cpu")

print(f"I am worker {dist.get_rank()} of {dist.get_world_size()} on {device}!")

# Create a tensor on GPU and perform all_reduce
a = torch.tensor([dist.get_rank()], device=device)
dist.all_reduce(a)
print(f"all_reduce output (on {device}) = {a.item()}")

dist.destroy_process_group()
