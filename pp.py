import csv
from datetime import datetime
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Union, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
import torch
from torch import optim, nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.pipelining import ScheduleGPipe, pipeline, SplitPoint, build_stage, Pipe
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler
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
        pipe: Pipe,
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

        self.train_data: DataLoader[torch.Tensor] = train_data
        self.test_data: DataLoader[torch.Tensor] = test_data
        self.epochs_run = 0
        self.epoch_losses = []
        self.training_output = None
        self.training_targets = None
        self.snapshot_job_id = snapshot_job_id
        snapshot_path = "" if snapshot_job_id is None else f"{CHECKPOINT_DIR}/{snapshot_job_id}/epoch_{snapshot_epoch}"
        self.num_microbatches = num_microbatches

        self.pipe = pipe
        self.stage_mod = pipe.get_stage_module(self.global_rank)
        self.stage_mod.to(self.device)
        self.optimizer = OptimizerClass(self.stage_mod.parameters())
        self.is_last_stage = self.global_rank == pipe.num_stages - 1  

        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        stage = build_stage(self.stage_mod, self.global_rank, self.pipe.info(), self.device)

        self.schedule = ScheduleGPipe(
            stage,
            n_microbatches=self.num_microbatches,
            loss_fn=F.cross_entropy,
        )

        stage = build_stage(self.stage_mod, self.global_rank, self.pipe.info(), self.device)
        self.schedule_inference = ScheduleGPipe(
            stage,
            n_microbatches=self.num_microbatches,
        )

        self.best_qwk = -1

    def _load_snapshot(self, snapshot_path):
        state_dict = {f"app": AppState(self.epochs_run, self.stage_mod, self.optimizer, self.global_rank)}
        dcp.load(state_dict, checkpoint_id=f"{snapshot_path}")
        self.epochs_run = state_dict[f"app"].epoch + 1
        
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch: int):
        state_dict = {f"app": AppState(epoch, self.stage_mod, self.optimizer, self.global_rank)}
        save_path = CHECKPOINT_DIR + f"/{self.job_id}/epoch_{epoch}"
        os.makedirs(save_path, exist_ok=True)

        dcp.save(state_dict, checkpoint_id=save_path)

        print(
            f"Epoch {epoch} | Training snapshot saved at {save_path}")

    def _run_batch(self, source, targets, step=None):
        self.optimizer.zero_grad()

        if self.global_rank == 0:
            self.schedule.step(source)
        elif self.is_last_stage:
            losses = []
            output = self.schedule.step(target=targets, losses=losses)

            argmax_output = torch.argmax(output, dim=1)
            if self.training_output is None:
                self.training_output = torch.clone(argmax_output)
            else:
                self.training_output = torch.cat((self.training_output, argmax_output), dim=0)
            if self.training_targets is None:
                self.training_targets = torch.clone(targets)
            else:
                self.training_targets = torch.cat((self.training_targets, targets), dim=0)
            self.epoch_losses.append(losses)
        else:
            self.schedule.step()
        
        self.optimizer.step()

    def _run_batch_inference(self, source):
        if self.global_rank == 0:
            output = self.schedule_inference.step(source)
        else:
            output = self.schedule_inference.step()

        return output
        
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Starting Epoch {epoch}")
        print('batch size:', b_sz)
        for step, (source, targets) in enumerate(self.train_data):
            if self.global_rank == 0:
                source = source.to(self.device)
            if self.is_last_stage:
                targets = targets.to(self.device)
            self._run_batch(source, targets, step=step)
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
                self.epoch_losses = []

                training_output = self.training_output.detach().cpu().numpy()
                training_targets = self.training_targets.detach().cpu().numpy()
                training_accuracy = accuracy_score(training_targets, training_output)
                print(f"Epoch {epoch} | Training Accuracy: {training_accuracy}")
                self.training_output = None
                self.training_targets = None
                
                self._log_metric("train_accuracy", training_accuracy, epoch)
                self._log_metric("loss", loss.item(), epoch)

            self.stage_mod.eval()
            self.schedule
            save_model = torch.tensor(self._evaluate(epoch), device=self.device)
            self.stage_mod.train()
            dist.broadcast(save_model, src=self.pipe.num_stages - 1)

            # if save_model:
            #     print(f"Saving model at epoch {epoch}")
            #     self._save_snapshot(epoch)

    @torch.no_grad()
    def _evaluate(self, epoch: int):
        merged_targets = None
        merged_output = None
        for source, targets in self.test_data:
            if self.global_rank == 0:
                source = source.to(self.device)
            if self.is_last_stage:
                targets = targets.to(self.device)
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
            weighted_f1 = f1_score(merged_targets, merged_output, average='weighted')
            accuracy = accuracy_score(merged_targets, merged_output)
            print(f"Epoch {epoch} | Validation Accuracy: {accuracy}")
            print(f"Epoch {epoch} | Validation Weighted F1: {weighted_f1}")
            print(f"Epoch {epoch} | Validation QWK: {qwk}")

            self._log_metric("val_accuracy", accuracy, epoch)
            self._log_metric("weighted_f1", weighted_f1, epoch)
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
            row = [now, self.job_id, self.global_rank, self.local_rank, model_start_job_id, epoch, value]
            writer.writerow(row)
    
    def _log_gradient(self, step):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f"/mnt/dcornelius/training_logs/gradients{self.global_rank}.csv", "a") as f:
            writer = csv.writer(f)
            for i, (name, param) in enumerate(self.stage_mod.named_parameters()):
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
    
    batch_size = 30
    sampler = SequentialSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, sampler=sampler
    )
    sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, drop_last=True, sampler=sampler
    )

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    features_in = model.classifier.in_features
    model.classifier = nn.Linear(features_in, 5)

    num_microbatches = 1
    input_sample = next(iter(train_loader))[0][:batch_size//num_microbatches]
    pipe = pipeline(
        model,
        mb_args=(input_sample,),
        split_spec={
            "features.denseblock3.denselayer1": SplitPoint.BEGINNING,
        }
    )
    print("model initialized")

    trainer = Trainer(
        pipe=pipe,
        train_data=train_loader,
        test_data=test_loader,
        OptimizerClass=optim.Adam,
        num_microbatches=num_microbatches,
        # snapshot_job_id=,
        # snapshot_epoch=,
    )
    trainer.train(max_epochs=10)

    cleanup()


if __name__ == "__main__":
    main()