import torch
import torch.distributed as dist
import os
from pathlib import Path


def setup():
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)


def cleanup():
    dist.destroy_process_group()


def measure_communication_time():
    tensor_size = 1024 * 1024 
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()

    if global_rank == 0:
        tensor = torch.randn(tensor_size, dtype=torch.float32, device=f"cuda:{local_rank}")
        ack = torch.zeros(1, dtype=torch.float32, device=f"cuda:{local_rank}")

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        dist.send(tensor, dst=1)
        dist.recv(ack, src=1)
        end_time.record()

        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        return elapsed_time
    elif global_rank == 1:
        tensor = torch.zeros(tensor_size, dtype=torch.float32, device=f"cuda:{local_rank}")
        ack = torch.ones(1, dtype=torch.float32, device=f"cuda:{local_rank}")

        dist.recv(tensor, src=0)
        dist.send(ack, dst=0)

        torch.cuda.synchronize()



def main():
    setup()
    job_id = os.getenv("TORCHX_JOB_ID", "local").split("/")[-1]
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Job ID: {job_id}, GPU: {gpu_name}, Rank: {dist.get_rank()}, Local Rank: {dist.get_rank() % torch.cuda.device_count()}")

    # elapsed_time = measure_communication_time()
    if dist.get_rank() == 0:
        with open("/mnt/dcornelius/training_logs/communication_time.csv", "a") as f:
            for i in range(1000):
                elapsed_time = measure_communication_time()
                f.write(f"{job_id},{i},{elapsed_time}\n")
    elif dist.get_rank() == 1:
        for i in range(1000):
            measure_communication_time()

    cleanup()

if __name__ == "__main__":
    main()