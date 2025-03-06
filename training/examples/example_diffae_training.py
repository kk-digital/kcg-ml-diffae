import os, sys
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as VF
from torchvision.transforms import RandomResizedCrop
from PIL import Image
from tqdm.auto import tqdm

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from diffae.templates import ffhq256_autoenc, LitModel, WarmupLR, ema
from diffae.choices import ModelMeanType, LossType, OptimizerType, TrainMode
from diffae.model.nn import mean_flat


class RandomImageDataset(Dataset):
    """ Generates random images for testing purposes. Replace with actual dataset. """

    def __init__(self, num_images=1000, image_size=2):
        self.num_images = num_images
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        image = Image.fromarray(np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8))
        return self.transform(image) * 2. - 1.  # Normalize to [-1, 1]


class DiffAETrainingPipeline:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32  # Change to torch.bfloat16 if needed

        # Load Model Configuration
        self.conf = config
        self.batch_size = 2
        self.num_epochs = 2
        self.gradient_accumulation_steps = 1
        self.max_train_steps = 7  # Adjust based on dataset

        # Model Initialization
        self.model = LitModel(self.conf).to(self.device, dtype=self.dtype)
        self.ema_model = self.model.ema_model.to(self.device, dtype=self.dtype)

        # Dataset and Dataloader
        self.dataset = RandomImageDataset(num_images=10000, image_size=self.conf.model_conf.image_size)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)

        # Optimizer & Scheduler
        self.optim = torch.optim.AdamW(self.model.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)

        if self.conf.warmup > 0:
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=WarmupLR(self.conf.warmup))

        # Training Progress
        self.progress_bar = tqdm(range(self.max_train_steps), desc="Steps")

    def train_step(self, x_start):
        """Perform one training step"""
        t, weight = self.model.T_sampler.sample(x_start.shape[0], device=self.device)
        noise = torch.randn_like(x_start)
        x_t = self.model.sampler.q_sample(x_start, t, noise=noise).to(dtype=x_start.dtype)

        model_output = self.model.model.forward(
            x=x_t.detach(),
            t=self.model.sampler._scale_timesteps(t),
            x_start=x_start.detach()
        ).pred

        target = noise
        loss = mean_flat((target - model_output) ** 2).mean()

        return loss

    def train(self):
        """Train the DiffAE model"""
        self.model.train()
        step, epoch = 0, 1
        data_iter = iter(self.dataloader)
        losses = []

        while step < self.max_train_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                torch.save(self.model.state_dict(), f"checkpoint_epoch{epoch}.pth")
                epoch += 1
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            self.optim.zero_grad()
            x_start = batch.to(self.device, dtype=self.dtype)
            loss = self.train_step(x_start)

            loss.backward()
            losses.append(loss.item())

            if step % self.gradient_accumulation_steps == 0:
                if hasattr(self.model, 'on_before_optimizer_step'):
                    #self.model.on_before_optimizer_step(self.optim, 0)
                    self.model.on_before_optimizer_step(self.optim)
                self.optim.step()
                self.sched.step()
                ema(self.model.model, self.ema_model, self.conf.ema_decay)

            if step % 1000 == 0:  # Save checkpoint every 1000 steps
                torch.save(self.model.state_dict(), f"checkpoint_step{step}.pth")

            step += 1
            self.progress_bar.update(1)

        print("Training complete.")


if __name__ == "__main__":
    print('Started')
    conf = ffhq256_autoenc()  # Load the default DiffAE config
    print('Started 2')
    pipeline = DiffAETrainingPipeline(conf)
    print('Started 3')
    pipeline.train()
