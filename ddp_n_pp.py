import os
import random

import numpy as np
import pandas as pd
import torch
from torch import optim, nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.pipelining import SplitPoint, ScheduleGPipe, PipelineStage
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import models, io
from tqdm import tqdm


def setup():
    dist.init_process_group(backend='nccl')
    device_mesh = init_device_mesh(
        'cuda', mesh_shape=(3, 2), mesh_dim_names=('dp', 'pp'))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    return device_mesh


def cleanup():
    dist.destroy_process_group()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Normalize:
    def __call__(self, img):
        img = img / 255.

        return img.float()


class AptosDataset(Dataset):
    def __init__(self, csv_file, root_dir, filename_col, label_col='diagnosis', transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.filename_col = filename_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = str(self.metadata.loc[idx, self.filename_col])
        img_path = os.path.join(self.root_dir, img_name + '.png')
        image = io.read_image(img_path)
        if self.transform:
            image = self.transform(image)

        label = self.metadata.loc[idx, self.label_col]

        return image, label
    

def custom_loss_fn(output: torch.Tensor, target: torch.Tensor):
    print(f"Output shape: {output.shape}")
    print(f"Target shape: {target.shape}")
    return F.cross_entropy(output, target)


class Trainer:
    def __init__(
        self,
        model_stages: list[torch.nn.Module],
        train_data: DataLoader,
        optimizers: list[torch.optim.Optimizer],
        device_mesh: DeviceMesh,
        num_microbatches: int = 4,
        save_every: int = 0,
        snapshot_path: str = '',
    ) -> None:
        self.global_rank = dist.get_rank()
        self.local_rank = self.global_rank % torch.cuda.device_count()
        self.device = torch.device(f'cuda:{self.local_rank}')
        self.model_stages = model_stages
        self.train_data: DataLoader[torch.Tensor] = train_data
        self.optimizers = optimizers
        self.device_mesh = device_mesh
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.num_microbatches = num_microbatches
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model_stage = self.model_stages[self.local_rank]
        self.model_stage.to(self.device)
        self.model_stage = DDP(self.model_stage, process_group=self.device_mesh['dp'].get_group())

        self.model_stage = PipelineStage(
            self.model_stage,
            stage_index=self.local_rank,
            num_stages=len(self.model_stages),
            device=self.device,
            group=self.device_mesh['pp'].get_group(),
        )

        # pp_size = self.device_mesh['pp'].size()
        # loss_fn = F.cross_entropy if self.local_rank == 0 else None
        self.schedule = ScheduleGPipe(
            self.model_stage,
            n_microbatches=self.num_microbatches,
            # loss_fn=F.cross_entropy,
            loss_fn=custom_loss_fn,
        )

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(
            f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _run_batch(self, source, targets):
        # self.optimizer.zero_grad()
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        if self.local_rank == 0:
            self.schedule.step(source)
        elif self.local_rank == len(self.model_stages) - 1:
            self.schedule.step(target=targets)
        else:
            self.schedule.step()

        # loss = F.cross_entropy(output, targets)
        # loss.backward()
        
        # self.optimizer.step()
        for optimizer in self.optimizers:
            optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        print(f"[GPU{self.global_rank}] Starting Epoch {epoch}")
        print('batch size:', b_sz)
        for source, targets in tqdm(self.train_data):
            source = source.to('cuda:0')
            targets = targets.to('cuda:1')
            self._run_batch(source, targets)
        print(
            f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
    
    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            # if self.local_rank == 0 and self.save_every != 0 and epoch % self.save_every == 0:
            #     self._save_snapshot(epoch)


def main():
    device_mesh = setup()
    set_seed(42)

    dataset = AptosDataset(
        csv_file="/mnt/dcornelius/preprocessed-aptos/train.csv",
        root_dir="/mnt/dcornelius/preprocessed-aptos/train_images",
        filename_col="new_id_code",
        label_col="diagnosis",
        transform=Normalize(),
    )
    sampler = DistributedSampler(dataset)
    data_loader = DataLoader(dataset, batch_size=8, sampler=sampler, drop_last=True)
    # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = models.resnet50()
    print('model initialized')
    stage1 = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
    )
    stage2 = nn.Sequential(
        model.layer3,
        model.layer4,
        model.avgpool,
        nn.Flatten(),
        model.fc,
    )
    model_stages = [stage1, stage2]
    optimizers = [optim.Adam(stage.parameters()) for stage in model_stages]

    trainer = Trainer(
        model_stages=model_stages,
        train_data=data_loader,
        optimizers=optimizers,
        device_mesh=device_mesh,
        num_microbatches=4,
    )
    trainer.train(max_epochs=1)

    cleanup()


if __name__ == "__main__":
    main()
