import csv
from datetime import datetime
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Iterable, Union

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
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


CHECKPOINT_DIR = "/mnt/dcornelius/checkpoints/single"


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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_data: DataLoader[torch.Tensor] = train_data
        self.test_data: DataLoader[torch.Tensor] = test_data
        self.model = model.to(self.device)
        self.optimizer = OptimizerClass(self.model.parameters())
        self.epochs_run = 0
        self.epoch_losses = []
        self.training_output = None
        self.training_targets = None
        self.best_qwk = -1

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

    def _run_batch(self, source, targets, step):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)

        self._log_metric("loss", loss.item(), step)

        loss.backward()
        self.optimizer.step()

    def _run_batch_inference(self, source, targets):
        output = self.model(source)
        return output

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"Starting Epoch {epoch}")
        for step, (source, targets) in enumerate(self.train_data):
            source = source.to(self.device)
            targets = targets.to(self.device)
            self._run_batch(source, targets, step)
            if step == 30: break
        print(f"Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            start_time = perf_counter()
            self._run_epoch(epoch)
            end_time = perf_counter()
            print(f"Epoch {epoch} | Time: {end_time - start_time:.2f}s")

            # loss = torch.mean(torch.tensor(self.epoch_losses, device=self.device))
            # print(f"Epoch {epoch} | Loss: {loss.item()}")
            # self.epoch_losses = []

            # training_output = self.training_output.detach().cpu().numpy()
            # training_targets = self.training_targets.detach().cpu().numpy()
            # training_accuracy = accuracy_score(training_targets, training_output)
            # print(f"Epoch {epoch} | Training Accuracy: {training_accuracy}")
            # self.training_output = None
            # self.training_targets = None

            # self._log_metric("train_accuracy", training_accuracy, epoch)
            # self._log_metric("epoch_time", end_time - start_time, epoch)
            # self._log_metric("loss", loss.item(), epoch)
            
            # self.model.eval()
            # save_model = torch.tensor(self._evaluate(epoch), device=self.device)
            # self.model.train()

            # if save_model:
            #     print(f"Saving snapshot at epoch {epoch}")
            #     self._save_snapshot(epoch)

    @torch.no_grad()
    def _evaluate(self, epoch):
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
        
        loss = F.cross_entropy(local_output, local_targets)
        print(f"Epoch {epoch} | Validation Loss: {loss.item()}")


        self._log_metric("val_loss", loss.item(), epoch)
        local_output = local_output.detach().cpu().numpy()
        local_targets = local_targets.detach().cpu().numpy()
        local_output = np.argmax(local_output, axis=1)

        accuracy = accuracy_score(local_targets, local_output)
        weighted_f1 = f1_score(local_targets, local_output, average="weighted")
        qwk = cohen_kappa_score(local_targets, local_output, weights="quadratic")

        print(f"Epoch {epoch} | Validation Accuracy: {accuracy}")
        print(f"Epoch {epoch} | Validation Weighted F1: {weighted_f1}")
        print(f"Epoch {epoch} | Validation QWK: {qwk}")

        self._log_metric("val_accuracy", accuracy, epoch)
        self._log_metric("weighted_f1", weighted_f1, epoch)
        self._log_metric("qwk", qwk, epoch)

        if qwk > self.best_qwk:
            self.best_qwk = qwk
            print(f"New Best Validation QWK: {self.best_qwk}")
            return True
            
        return False

    def _log_metric(self, metric, value, epoch):
        with open(f"/mnt/dcornelius/training_logs/perstep/{metric}.csv", "a") as f:
            writer = csv.writer(f)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_start_job_id = self.snapshot_job_id if self.snapshot_job_id else self.job_id
            writer.writerow([now, self.job_id, 0, 0, model_start_job_id, epoch, value])


def main():
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
    sampler = SequentialSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=30, num_workers=2, sampler=sampler)

    test_dataset = AptosDataset(
        csv_file=(dataset_dir / "test.csv"),
        root_dir=(dataset_dir / "test_images"),
        filename_col="id_code",
        label_col="diagnosis",
        transform=Normalize(),
    )
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False, num_workers=2)
    
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
    trainer.train(max_epochs=1)


if __name__ == "__main__":
    main()
