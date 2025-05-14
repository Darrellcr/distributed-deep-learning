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
from torch.distributed.pipelining import ScheduleGPipe, PipelineStage, pipeline
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
        OptimizerClass: type[optim.Optimizer],
        train_data: DataLoader,
        test_data: DataLoader,
        num_microbatches: int = 4,
        snapshot_job_id: str = None,
        snapshot_epoch: int = None,
    ) -> None:
        self.job_id = os.getenv("TORCHX_JOB_ID", "local").split("/")[-1]
        self.global_rank = dist.get_rank()
        self.local_rank = self.global_rank % torch.cuda.device_count()
        self.device = torch.device(f'cuda:{self.local_rank}')

        self.model_stages = model_stages
        self.train_data: DataLoader[torch.Tensor] = train_data
        self.test_data: DataLoader[torch.Tensor] = test_data
        self.epochs_run = 0
        self.epoch_losses = []
        self.snapshot_job_id = snapshot_job_id
        snapshot_path = "" if snapshot_job_id is None else f"{CHECKPOINT_DIR}/{snapshot_job_id}/epoch_{snapshot_epoch}"
        self.num_microbatches = num_microbatches

        self.model_stage = self.model_stages[self.local_rank]
        self.model_stage.to(self.device)
        self.optimizer = OptimizerClass(self.model_stage.parameters())
        self.is_last_stage = self.global_rank == len(self.model_stages) - 1  
       
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

        self.schedule_inference = ScheduleGPipe(
            self.pipeline_stage,
            n_microbatches=self.num_microbatches,
        )

        self.best_qwk = -1

    def _load_snapshot(self, snapshot_path):
        state_dict = {f"app": AppState(self.epochs_run, self.model_stage, self.optimizer, self.global_rank)}
        dcp.load(state_dict, checkpoint_id=f"{snapshot_path}")
        self.epochs_run = state_dict[f"app"].epoch + 1
        
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch: int):
        state_dict = {f"app": AppState(epoch, self.model_stage, self.optimizer, self.global_rank)}
        save_path = CHECKPOINT_DIR + f"/{self.job_id}/epoch_{epoch}"
        os.makedirs(save_path, exist_ok=True)

        dcp.save(state_dict, checkpoint_id=save_path)

        print(
            f"Epoch {epoch} | Training snapshot saved at {save_path}")

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

        named_parameters = self.model_stage.named_parameters()
        gradients = next(named_parameters)[1].grad
        print(f"max: {torch.max(gradients)}, min: {torch.min(torch.abs(gradients))}")
        self.optimizer.step()

    def _run_batch_inference(self, source):
        if self.local_rank == 0:
            output = self.schedule_inference.step(source)
        else:
            output = self.schedule_inference.step()

        return output
        
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
            start_time = perf_counter()
            self._run_epoch(epoch)
            end_time = perf_counter()

            if self.is_last_stage:
                print(f"Epoch {epoch} | Time: {end_time - start_time:.2f}s")
                self._log_metric("epoch_time", end_time - start_time, epoch)
                loss = torch.mean(torch.tensor(self.epoch_losses, device=self.device))
                print(f"Epoch {epoch} | Loss: {loss.item()}")
                self._log_metric("loss", loss.item(), epoch)

                self.epoch_losses = []

            self.model_stage.eval()
            save_model = torch.tensor(self._evaluate(epoch), device=self.device)
            self.model_stage.train()
            dist.broadcast(save_model, src=len(self.model_stages) - 1)

            if save_model:
                print(f"Saving model at epoch {epoch}")
                self._save_snapshot(epoch)

    @torch.no_grad()
    def _evaluate(self, epoch: int):
        merged_targets = None
        merged_output = None
        for source, targets in self.test_data:
            source = source.to('cuda:0')
            targets = targets.to(f'cuda:{torch.cuda.device_count() - 1}')
            output = self._run_batch_inference(source)

            if self.is_last_stage:
                if merged_targets is None:
                    merged_targets = torch.clone(targets)
                else:
                    merged_targets = torch.cat((merged_targets, targets), dim=0)
                if merged_output is None:
                    merged_output = torch.clone(output)
                else:
                    merged_output = torch.cat((merged_output, output), dim=0)

        if self.is_last_stage:
            loss = F.cross_entropy(merged_output, merged_targets)
            print(f"Epoch {epoch} | Validation Loss: {loss}")
            self._log_metric("val_loss", loss.item(), epoch)
            merged_output = merged_output.detach().cpu().numpy()
            merged_targets = merged_targets.detach().cpu().numpy()
            merged_output = np.argmax(merged_output, axis=1)

            qwk = cohen_kappa_score(merged_targets, merged_output, weights='quadratic')
            print(f"Epoch {epoch} | New Best Validation QWK: {qwk}")

            self._log_metric("qwk", qwk, epoch)
            if qwk > self.best_qwk:
                self.best_qwk = qwk
                print(f"Best validation QWK: {self.best_qwk}")
                return True
            
        return False

    def _log_metric(self, metric, value, epoch):
        with open(f"/mnt/dcornelius/training_logs/{metric}.csv", "a") as f:
            writer = csv.writer(f)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_start_job_id = self.snapshot_job_id if self.snapshot_job_id else self.job_id
            writer.writerow([now, self.job_id, self.global_rank, self.local_rank, model_start_job_id, epoch, value])
    
            

def main():
    setup()
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
    test_dataset = AptosDataset(
        csv_file=(dataset_dir / "test.csv"),
        root_dir=(dataset_dir / "test_images"),
        filename_col="id_code",
        label_col="diagnosis",
        transform=Normalize(),
    )
    
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=30, drop_last=True, shuffle=True, generator=g
    )
    test_loader = DataLoader(
        test_dataset, batch_size=30, drop_last=True, shuffle=True
    )

    # model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    # print("model initialized")
    # stage1 = nn.Sequential(
    #     model.features.conv0,
    #     model.features.norm0,
    #     model.features.relu0,
    #     model.features.pool0,
    #     model.features.denseblock1,
    #     model.features.transition1,
    #     model.features.denseblock2,
    #     model.features.transition2,
    # )
    # stage2 = nn.Sequential(
    #     model.features.denseblock3,
    #     model.features.transition3,
    #     model.features.denseblock4,
    #     model.features.norm5,
    #     nn.ReLU(inplace=True),
    #     nn.AdaptiveAvgPool2d((1, 1)),
    #     nn.Flatten(),
    #     model.classifier,
    # )
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
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

    trainer = Trainer(
        model_stages=model_stages,
        train_data=train_loader,
        test_data=test_loader,
        OptimizerClass=optim.Adam,
        num_microbatches=6,
        # snapshot_job_id=,
        # snapshot_epoch=,
    )
    trainer.train(max_epochs=10)

    cleanup()


if __name__ == "__main__":
    main()