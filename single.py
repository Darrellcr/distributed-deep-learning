import csv
from datetime import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import io, models


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
    

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int = 0,
        snapshot_path: str = '',
    ) -> None:
        self.job_id = os.getenv("TORCHX_JOB_ID", "local")
        self.train_data: DataLoader[torch.Tensor] = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        # if os.path.exists(snapshot_path):
        #     print("Loading snapshot")
        #     self._load_snapshot(snapshot_path)

        self.model = model

    # def _load_snapshot(self, snapshot_path):
    #     loc = f"cuda:{self.local_rank}"
    #     snapshot = torch.load(snapshot_path, map_location=loc)
    #     self.model.load_state_dict(snapshot["MODEL_STATE"])
    #     self.epochs_run = snapshot["EPOCHS_RUN"]
    #     print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"Starting Epoch {epoch}")
        for source, targets in self.train_data:
            self._run_batch(source, targets)
        print(f"Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")

    # def _save_snapshot(self, epoch):
    #     snapshot = {
    #         "MODEL_STATE": self.model.module.state_dict(),
    #         "EPOCHS_RUN": epoch,
    #     }
    #     torch.save(snapshot, self.snapshot_path)
    #     print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            self._log_loss(loss, epoch)

            # if self.local_rank == 0 and self.save_every != 0 and epoch % self.save_every == 0:
            #     self._save_snapshot(epoch)
    
    def _log_loss(self, loss, epoch):
        with open("/mnt/dcornelius/training_logs/loss.csv", "a") as f:
            writer = csv.writer(f)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([now, self.job_id, 0, 0, epoch, loss])


def main():
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
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, generator=g)

    model = models.densenet121()
    print('model initialized')

    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(
        model=model,
        train_data=data_loader,
        optimizer=optimizer,
    )
    trainer.train(max_epochs=3)


if __name__ == "__main__":
    main()