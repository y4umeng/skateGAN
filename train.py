import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed
from perceptual_loss import PerceptualLoss
from data import skate_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_encoder_path', type=str, default=None)
    parser.add_argument('--pretrained_decoder_path', type=str, default=None)
    parser.add_argument('--vgg19_path', type=str, default='./networks/imagenet-vgg-verydeep-19.mat')
    parser.add_argument('--output_model_path', type=str, default='vit-t-classifier.pt')
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--val_path', type=str, default=None)  

    args = parser.parse_args()

    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    transform = Compose([ToTensor(), Normalize(0.5, 0.5)]) 
    train_dataset = skate_data(args.train_path, device, transform=transform)
    val_dataset = skate_data(args.val_path, device, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    
    model = skateGAN('', batch_size, device)
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'pretrain-cls'))

    if args.pretrained_encoder_path is not None:
        model.encoder = torch.load(args.pretrained_encoder_path, map_location='cpu') 
        print(f"Loaded pretrained encoder from {args.pretrained_encoder_path}")
    if args.pretrained_decoder_path is not None:
        model.decoder = torch.load(args.pretrained_decoder_path, map_location='cpu')  
        print(f"Loaded pretrained decoder from {args.pretrained_decoder_path}")

    loss_fn = PerceptualLoss(args.vgg19_path)

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    best_val_loss = 0
    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img_curr, img_prev in tqdm(iter(train_dataloader)):
            step_count += 1
            img_curr = img_curr.to(device)
            img_prev = img_prev.to(device)
            img_gen = model(img_curr, img_prev)
            loss, _ = loss_fn(img_gen, img_curr)
            loss.backward()

            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        print(f'In epoch {e}, average training loss is {avg_train_loss}.')

        model.eval()
        with torch.no_grad():
            losses = []
            for img_curr, img_prev in tqdm(iter(val_dataloader)):
                img_curr = img_curr.to(device)
                img_prev = img_prev.to(device)
                img_gen = model(img_curr, img_prev)
                loss, _ = loss_fn(img_gen, img_curr)
                losses.append(loss.item())
            avg_val_loss = sum(losses) / len(losses)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}.')  

        if avg_val_loss > best_val_loss:
            best_val_loss = avg_val_loss
            print(f'saving best model with acc {best_val_loss} at {e} epoch!')       
            torch.save(model, args.output_model_path)

        writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
