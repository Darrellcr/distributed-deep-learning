import csv
from datetime import datetime
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Iterable, Union

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
import torch
from torch import optim, nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler
from torchvision import models, io
from tqdm import tqdm


CHECKPOINT_DIR = "/mnt/dcornelius/checkpoints/ddp"


def setup():
    dist.init_process_group(backend="nccl")
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
    def __init__(self, epoch: int, model: nn.Module, optimizer: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]]):
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            f"model": model_state_dict,
            f"optim": optimizer_state_dict,
            "epoch": torch.tensor(self.epoch),
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict[f"model"],
            optim_state_dict=state_dict[f"optim"]
        )
        self.epoch = state_dict["epoch"].item()


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        OptimizerClass: type[optim.Optimizer],
        snapshot_job_id: str = None,
        snapshot_epoch: int = None,
    ) -> None:
        self.job_id = os.getenv("TORCHX_JOB_ID", "local").split("/")[-1]
        self.global_rank = dist.get_rank()
        self.local_rank = self.global_rank % torch.cuda.device_count()
        self.device = torch.device(f'cuda:{self.local_rank}')

        self.train_data: DataLoader[torch.Tensor] = train_data
        self.test_data: DataLoader[torch.Tensor] = test_data
        self.model = model.to(self.device)
        self.optimizer = OptimizerClass(self.model.parameters())
        self.epochs_run = 0
        self.epoch_losses = []
        self.training_output = None
        self.training_targets = None
        self.best_qwk = -1

        self.model = DDP(self.model, device_ids=[self.local_rank])

        self.snapshot_job_id = snapshot_job_id
        snapshot_path = '' if snapshot_job_id is None else CHECKPOINT_DIR + f"/{snapshot_job_id}/epoch_{snapshot_epoch}"
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

    def _load_snapshot(self, snapshot_path):
        state_dict = {"app": AppState(self.epochs_run, self.model, self.optimizer)}
        dcp.load(state_dict, checkpoint_id=f"{snapshot_path}")
        self.epochs_run = state_dict["app"].epoch + 1
        
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        state_dict = {"app": AppState(self.epochs_run, self.model, self.optimizer)}
        save_path = CHECKPOINT_DIR + f"/{self.job_id}/epoch_{epoch}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        dcp.save(state_dict, checkpoint_id=save_path)
        print(f"Epoch {epoch} | Saved snapshot to {save_path}")

    def _run_batch(self, source, targets, step=None):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)

        argmax_output = torch.argmax(output, dim=1)
        if self.training_output is None:
            self.training_output = torch.clone(argmax_output)
        else:
            self.training_output = torch.cat((self.training_output, argmax_output), dim=0)
        if self.training_targets is None:
            self.training_targets = torch.clone(targets)
        else:
            self.training_targets = torch.cat((self.training_targets, targets), dim=0)
        self.epoch_losses.append(loss)

        loss.backward()
        # if self.global_rank == 0:
        #     self._log_gradient(step)
        
        self.optimizer.step()

    def _run_batch_inference(self, source, targets):
        output = self.model(source)
        return output

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        print(f"[GPU{self.global_rank}] Starting Epoch {epoch}")
        for step, (source, targets) in enumerate(self.train_data):
            source = source.to(self.device)
            targets = targets.to(self.device)
            self._run_batch(source, targets, step=step)

        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            start_time = perf_counter()
            self._run_epoch(epoch)
            end_time = perf_counter()
            print(f"Epoch {epoch} | Time: {end_time - start_time:.2f}s")

            all_training_output = [torch.zeros_like(self.training_output, device=self.device) for _ in range(dist.get_world_size())]
            all_training_targets = [torch.zeros_like(self.training_targets, device=self.device) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=all_training_output, tensor=self.training_output)
            dist.all_gather(tensor_list=all_training_targets, tensor=self.training_targets)
            all_training_output = torch.cat(all_training_output, dim=0)
            all_training_targets = torch.cat(all_training_targets, dim=0)

            all_training_output = all_training_output.detach().cpu().numpy()
            all_training_targets = all_training_targets.detach().cpu().numpy()
            train_accuracy = accuracy_score(all_training_targets, all_training_output)
            self.training_output = None
            self.training_targets = None

            if self.global_rank == 0:
                self._log_metric("train_accuracy", train_accuracy, epoch)
                print(f"Epoch {epoch} | Training Accuracy: {train_accuracy}")

            loss = torch.mean(torch.tensor(self.epoch_losses, device=self.device))
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            print(f"Epoch {epoch} | Loss: {loss.item()}")
            self.epoch_losses = []

            if self.global_rank == 0:
                self._log_metric("epoch_time", end_time - start_time, epoch)
                self._log_metric("loss", loss.item(), epoch)
            
            self.model.eval()
            save_model = torch.tensor(self._evaluate(epoch), device=self.device)
            self.model.train()
            dist.broadcast(save_model, src=0)

            # if save_model:
            #     print(f"Saving snapshot at epoch {epoch}")
            #     self._save_snapshot(epoch)

    @torch.no_grad()
    def _evaluate(self, epoch):
        self.test_data.sampler.set_epoch(epoch)
        local_targets = None
        local_output = None
        for source, targets in self.test_data:
            source = source.to(self.device)
            targets = targets.to(self.device)
            output= self._run_batch_inference(source, targets)

            if local_output is None:
                local_output = torch.clone(output)
            else:
                local_output = torch.cat((local_output, output), dim=0)
            if local_targets is None:
                local_targets = torch.clone(targets)
            else:
                local_targets = torch.cat((local_targets, targets), dim=0)
        
        all_output = [torch.zeros_like(local_output, device=self.device) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=all_output, tensor=local_output)
        all_output = torch.cat(all_output, dim=0)

        all_targets = [torch.zeros_like(local_targets, device=self.device) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=all_targets, tensor=local_targets)
        all_targets = torch.cat(all_targets, dim=0)

        loss = F.cross_entropy(all_output, all_targets)
        print(f"Epoch {epoch} | Validation Loss: {loss.item()}")

        if self.global_rank == 0:
            self._log_metric("val_loss", loss.item(), epoch)
            all_output = all_output.detach().cpu().numpy()
            all_targets = all_targets.detach().cpu().numpy()
            all_output = np.argmax(all_output, axis=1)

            qwk = cohen_kappa_score(all_targets, all_output, weights="quadratic")
            weighted_f1 = f1_score(all_targets, all_output, average="weighted")
            accuracy = accuracy_score(all_targets, all_output)
            print(f"Epoch {epoch} | Validation QWK: {qwk}")
            print(f"Epoch {epoch} | Validation Weighted F1: {weighted_f1}")
            print(f"Epoch {epoch} | Validation Accuracy: {accuracy}")

            self._log_metric("qwk", qwk, epoch)
            self._log_metric("weighted_f1", weighted_f1, epoch)
            self._log_metric("val_accuracy", accuracy, epoch)

            if qwk > self.best_qwk:
                self.best_qwk = qwk
                print(f"New Best Validation QWK: {self.best_qwk}")
                return True
            
        return False

    def _log_metric(self, metric, value, epoch):
        with open(f"/mnt/dcornelius/training_logs/{metric}.csv", "a") as f:
            writer = csv.writer(f)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_start_job_id = self.snapshot_job_id if self.snapshot_job_id else self.job_id
            writer.writerow([now, self.job_id, self.global_rank, self.local_rank, model_start_job_id, epoch, value])

    def _log_gradient(self, step):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f"/mnt/dcornelius/training_logs/gradient.csv", "a") as f:
            writer = csv.writer(f)
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if param.grad is None: 
                    continue
                
                min_grad = param.grad.abs().min()
                mean_grad = param.grad.abs().mean()
                max_grad = param.grad.abs().max()
                percentile25th = torch.quantile(param.grad.abs(), 0.25)
                median_grad = param.grad.abs().median()
                percentile75th = torch.quantile(param.grad.abs(), 0.75)
                std_grad = param.grad.abs().std()
                row = [now, self.job_id, self.global_rank, self.local_rank, step, i, name, min_grad.item(), mean_grad.item(), max_grad.item(), percentile25th.item(), median_grad.item(), percentile75th.item(), std_grad.item()]
                writer.writerow(row)


def main():
    setup()
    seed = 42
    set_seed(seed)

    dataset_dir = Path("/mnt/dcornelius/preprocessed-aptos")
    batch_size = 15
    train_dataset = AptosDataset(
        csv_file=(dataset_dir / "train.csv"),
        root_dir=(dataset_dir / "train_images"),
        filename_col="new_id_code",
        label_col="diagnosis",
        transform=Normalize(),
    )
    train_sampler = SequentialSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)

    test_dataset = AptosDataset(
        csv_file=(dataset_dir / "test.csv"),
        root_dir=(dataset_dir / "test_images"),
        filename_col="id_code",
        label_col="diagnosis",
        transform=Normalize(),
    )
    test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=True, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False)
    
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 5)

    trainer = Trainer(
        model=model,
        train_data=train_loader,
        test_data=test_loader,
        OptimizerClass=optim.Adam,
        # snapshot_job_id=,
        # snapshot_epoch=,
    )
    trainer.train(max_epochs=10)
    
    cleanup()


if __name__ == "__main__":
    main()
