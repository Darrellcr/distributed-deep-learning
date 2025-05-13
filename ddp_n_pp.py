import csv
from datetime import datetime
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Union, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import torch
from torch import optim, nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.pipelining import SplitPoint, ScheduleGPipe, PipelineStage
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import models, io
from tqdm import tqdm


CHECKPOINT_DIR = "/mnt/dcornelius/checkpoints/ddpnpp"


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
    torch.backends.cudnn.benchmark = False


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
    def __init__(self, epoch: int, model: nn.Module, optimizer: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]], local_rank: int):
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.local_rank = local_rank

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            f"model{self.local_rank}": model_state_dict,
            f"optim{self.local_rank}": optimizer_state_dict,
            "epoch": torch.tensor(self.epoch),
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict[f"model{self.local_rank}"],
            optim_state_dict=state_dict[f"optim{self.local_rank}"]
        )
        self.epoch = state_dict["epoch"].item()
    

class Trainer:
    def __init__(
        self,
        model_stages: list[torch.nn.Module],
        train_data: DataLoader,
        test_data: DataLoader,
        OptimizerClass: type[optim.Optimizer],
        device_mesh: DeviceMesh,
        num_microbatches: int = 4,
        snapshot_job_id: str = None,
        snapshot_epoch: int = None,
    ) -> None:
        self.job_id = os.getenv("TORCHX_JOB_ID", "local").split("/")[-1]
        self.global_rank = dist.get_rank()
        self.local_rank = self.global_rank % torch.cuda.device_count()
        self.device = torch.device(f'cuda:{self.local_rank}')
        self.device_mesh = device_mesh

        self.model_stages = model_stages
        self.train_data: DataLoader[torch.Tensor] = train_data
        self.test_data: DataLoader[torch.Tensor] = test_data
        self.epochs_run = 0
        self.epoch_losses = []
        self.num_microbatches = num_microbatches

        self.model_stage = self.model_stages[self.local_rank]
        self.model_stage.to(self.device)
        self.optimizer = OptimizerClass(self.model_stage.parameters())
        self.model_stage = DDP(self.model_stage, process_group=self.device_mesh.get_group('dp'))
        self.is_last_stage = self.local_rank == len(self.model_stages) - 1

        self.snapshot_job_id = snapshot_job_id
        snapshot_path = "" if snapshot_job_id is None else CHECKPOINT_DIR + f"/{snapshot_job_id}/epoch_{snapshot_epoch}"
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.pipeline_stage = PipelineStage(
            self.model_stage,
            stage_index=self.local_rank,
            num_stages=len(self.model_stages),
            device=self.device,
            group=self.device_mesh.get_group('pp'),
        )

        self.schedule = ScheduleGPipe(
            self.pipeline_stage,
            n_microbatches=self.num_microbatches,
            loss_fn=F.cross_entropy,
        )

        self.schedule_inference = ScheduleGPipe(
            self.pipeline_stage,
            n_microbatches=self.num_microbatches,
        )

        self.best_qwk = -1

    def _load_snapshot(self, snapshot_path):
        state_dict = {"app": AppState(self.epochs_run, self.model_stage, self.optimizer, self.local_rank)}
        dcp.load(state_dict, checkpoint_id=f"{snapshot_path}")
        self.epochs_run = state_dict["app"].epoch + 1

        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        state_dict = {"app": AppState(epoch, self.model_stage, self.optimizer, self.local_rank)}
        save_path = CHECKPOINT_DIR + f"/{self.job_id}/epoch_{epoch}"
        os.makedirs(save_path, exist_ok=True)

        dcp.save(state_dict, checkpoint_id=save_path)

        print(f"Epoch {epoch} | Saved snapshot to {save_path}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()

        if self.local_rank == 0:
            self.schedule.step(source)
        elif self.is_last_stage:
            losses = []
            self.schedule.step(target=targets, losses=losses)
            self.epoch_losses.append(losses)
        else:
            self.schedule.step()
        
        self.optimizer.step()

    def _run_batch_inference(self, source):
        if self.local_rank == 0:
            output = self.schedule_inference.step(source)
        else:
            output = self.schedule_inference.step()

        return output

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
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
            start_time = perf_counter()
            self._run_epoch(epoch)
            end_time = perf_counter()
            print(f"Epoch {epoch} | Time: {end_time - start_time:.2f}s")

            if self.is_last_stage: 
                loss = torch.mean(torch.tensor(self.epoch_losses, device=self.device))
                dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=self.device_mesh.get_group('dp'))
                
                if self.device_mesh.get_group('dp').rank() == 0:
                    self._log_metric("epoch_time", end_time - start_time, epoch)
                    self._log_metric("loss", loss.item(), epoch)

            self.epoch_losses = []

            self.model_stage.eval()
            save_model = torch.tensor(self._evaluate(epoch), device=self.device)
            self.model_stage.train()
            dist.broadcast(save_model, src=self.device_mesh.mesh[0][-1])

            if save_model:
                print(f"Epoch {epoch} | Saving snapshot")
                self._save_snapshot(epoch)

    @torch.no_grad()
    def _evaluate(self, epoch):
        dist.barrier()
        self.test_data.sampler.set_epoch(epoch)

        local_targets = None
        local_output = None
        for source, targets in self.test_data:
            source = source.to('cuda:0')
            targets = targets.to(f'cuda:{torch.cuda.device_count() - 1}')
            output = self._run_batch_inference(source)
            
            if self.is_last_stage:
                if local_targets is None:
                    local_targets = torch.clone(targets)
                else:
                    local_targets = torch.cat((local_targets, targets), dim=0)
                if local_output is None:
                    local_output = torch.clone(output)
                else:
                    local_output = torch.cat((local_output, output), dim=0)

        if self.is_last_stage:
            all_targets = [torch.zeros_like(local_targets, device=self.device) for _ in range(self.device_mesh.get_group('dp').size())]
            all_output = [torch.zeros_like(local_output, device=self.device) for _ in range(self.device_mesh.get_group('dp').size())]
            dist.all_gather(all_targets, local_targets, group=self.device_mesh.get_group('dp'))
            dist.all_gather(all_output, local_output, group=self.device_mesh.get_group('dp'))
            all_targets = torch.cat(all_targets, dim=0)
            all_output = torch.cat(all_output, dim=0)

            loss = F.cross_entropy(all_output, all_targets)
            print(f"Epoch {epoch} | Validation Loss: {loss.item():.4f}")

            print('all_targets')
            print(all_targets)
            print('all_output')
            print(all_output)
            if self.device_mesh.get_group('dp').rank() == 0:
                self._log_metric("val_loss", loss.item(), epoch)
                all_targets = all_targets.detach().cpu().numpy()
                all_output = all_output.detach().cpu().numpy()
                all_output = np.argmax(all_output, axis=1)

                qwk = cohen_kappa_score(all_targets, all_output, weights='quadratic')
                print(f"Epoch {epoch} | Validation QWK: {qwk:.4f}")
                self._log_metric("qwk", qwk, epoch)

                if qwk > self.best_qwk:
                    self.best_qwk = qwk
                    print(f"Epoch {epoch} | New Best Validation QWK: {self.best_qwk:.4f}")
                    return True

        return False

    def _log_metric(self, metric, value, epoch):
        with open(f"/mnt/dcornelius/training_logs/{metric}.csv", "a") as f:
            writer = csv.writer(f)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_start_job_id = self.snapshot_job_id if self.snapshot_job_id else self.job_id
            writer.writerow([now, self.job_id, self.global_rank, self.local_rank, epoch, value])
            

def main():
    device_mesh = setup()
    seed = 42
    set_seed(seed)
    dataset_dir = Path("/mnt/dcornelius/preprocessed-aptos")

    train_dataset = AptosDataset(
        csv_file=(dataset_dir / "train.csv"),
        root_dir=(dataset_dir / "train_images"),
        filename_col="new_id_code",
        label_col="diagnosis",
        transform=Normalize(),
    )
    train_sampler = DistributedSampler(train_dataset, drop_last=True, shuffle=True, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=14, sampler=train_sampler, drop_last=True, shuffle=False)

    test_dataset = AptosDataset(
        csv_file=(dataset_dir / "test.csv"),
        root_dir=(dataset_dir / "test_images"),
        filename_col="id_code",
        label_col="diagnosis",
        transform=Normalize(),
    )
    test_sampler = DistributedSampler(test_dataset, drop_last=True, shuffle=False, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=14, sampler=test_sampler, drop_last=True, shuffle=False)

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    print('model initialized')
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

    trainer = Trainer(
        model_stages=model_stages,
        train_data=train_loader,
        test_data=test_loader,
        OptimizerClass=optim.Adam,
        device_mesh=device_mesh,
        num_microbatches=7,
        # snapshot_job_id="ddpnpp-cpjxq2lxc5sntd",
        # snapshot_epoch=1,
    )
    trainer.train(max_epochs=50)

    cleanup()


if __name__ == "__main__":
    main()
