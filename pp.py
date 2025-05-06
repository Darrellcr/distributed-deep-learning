import csv
from datetime import datetime
import os
import random
from typing import Union, Iterable

import numpy as np
import pandas as pd
import torch
from torch import optim, nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.pipelining import ScheduleGPipe, PipelineStage
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, io


CHECKPOINT_DIR = "/mnt/dcornelius/checkpoints/pp"


def setup():
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)


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


class AppState(Stateful):
    def __init__(self, epoch: int, model: nn.Module, optimizer: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]], global_rank):
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.global_rank = global_rank

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            f"model{self.global_rank}": model_state_dict,
            f"optim{self.global_rank}": optimizer_state_dict,
            "epoch": torch.tensor(self.epoch),
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict[f"model{self.global_rank}"],
            optim_state_dict=state_dict[f"optim{self.global_rank}"]
        )
        self.epoch = state_dict["epoch"].item()
    

class Trainer:
    def __init__(
        self,
        model_stages: list[torch.nn.Module],
        train_data: DataLoader,
        num_microbatches: int = 4,
        save_every: int = 0,
        snapshot_path: str = '',
    ) -> None:
        self.job_id = os.getenv("TORCHX_JOB_ID", "local").split("/")[-1]
        self.global_rank = dist.get_rank()
        self.local_rank = self.global_rank % torch.cuda.device_count()
        self.device = torch.device(f'cuda:{self.local_rank}')

        self.model_stages = model_stages
        self.train_data: DataLoader[torch.Tensor] = train_data
        self.save_every = save_every
        self.epochs_run = 0
        self.epoch_losses = []
        self.snapshot_path = snapshot_path
        self.num_microbatches = num_microbatches

        self.model_stage = self.model_stages[self.local_rank]
        self.model_stage.to(self.device)
        self.optimizer = optim.Adam(self.model_stage.parameters(), betas=(0.9, 0.999))
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.pipeline_stage = PipelineStage(
            self.model_stage,
            stage_index=self.local_rank,
            num_stages=len(self.model_stages),
            device=self.device,
        )

        self.schedule = ScheduleGPipe(
            self.pipeline_stage,
            n_microbatches=self.num_microbatches,
            loss_fn=F.cross_entropy,
        )

    def _load_snapshot(self, snapshot_path):
        state_dict = {f"app": AppState(self.epochs_run, self.model_stage, self.optimizer)}
        dcp.load(state_dict, checkpoint_id=f"{snapshot_path}")
        self.epochs_run = state_dict[f"app"].epoch
        
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch: int):
        state_dict = {f"app": AppState(epoch, self.model_stage, self.optimizer, self.global_rank)}
        save_path = CHECKPOINT_DIR + f"/{self.job_id}/epoch_{epoch}"
        os.makedirs(save_path, exist_ok=True)

        dcp.save(state_dict, checkpoint_id=save_path)

        print(
            f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()

        if self.local_rank == 0:
            self.schedule.step(source)
        elif self.local_rank == len(self.model_stages) - 1:
            losses = []
            self.schedule.step(target=targets, losses=losses)
            self.epoch_losses.append(losses)
        else:
            self.schedule.step()

        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Starting Epoch {epoch}")
        print('batch size:', b_sz)
        for source, targets in self.train_data:
            source = source.to('cuda:0')
            targets = targets.to(f'cuda:{torch.cuda.device_count() - 1}')
            self._run_batch(source, targets)
        print(
            f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
    
    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)

            if self.local_rank == len(self.model_stages) - 1:
                loss = torch.mean(torch.tensor(self.epoch_losses, device=self.device))
                self._log_loss(loss, epoch)

                self.epoch_losses = []

            # is_start_epoch = epoch == 0 or epoch == self.epochs_run
            # if self.save_every != 0 and not is_start_epoch and epoch % self.save_every == 0:
            self._save_snapshot(epoch)

    def _log_loss(self, loss, epoch):
        with open("/mnt/dcornelius/training_logs/loss.csv", "a") as f:
            writer = csv.writer(f)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([now, self.job_id, self.global_rank, self.local_rank, epoch, loss])
            

def main():
    setup()
    seed = 42
    set_seed(seed)

    dataset = AptosDataset(
        csv_file="/mnt/dcornelius/preprocessed-aptos/train.csv",
        root_dir="/mnt/dcornelius/preprocessed-aptos/train_images",
        filename_col="new_id_code",
        label_col="diagnosis",
        transform=Normalize(),
    )
    g = torch.Generator()
    g.manual_seed(seed)
    data_loader = DataLoader(dataset, batch_size=16, drop_last=True, shuffle=True, generator=g)

    model = models.densenet121()
    print("model initialized")
    stage1 = nn.Sequential(
        model.features.conv0,
        model.features.norm0,
        model.features.relu0,
        model.features.pool0,
        model.features.denseblock1,
        model.features.transition1,
        model.features.denseblock2,
        model.features.transition2,
    )
    stage2 = nn.Sequential(
        model.features.denseblock3,
        model.features.transition3,
        model.features.denseblock4,
        model.features.norm5,
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        model.classifier,
    )
    model_stages = [stage1, stage2]
    # optimizers = [optim.Adam(stage.parameters()) for stage in model_stages]

    trainer = Trainer(
        model_stages=model_stages,
        train_data=data_loader,
        num_microbatches=4,
        save_every=2,
        # snapshot_path=f"{CHECKPOINT_DIR}/pp-hb4wjkl5l3x36/epoch_2",
    )
    trainer.train(max_epochs=1)

    cleanup()


if __name__ == "__main__":
    main()