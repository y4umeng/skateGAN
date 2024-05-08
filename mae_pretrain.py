import argparse
import math
import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed
from data import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='checkpoints/pretrain128.pt')

    args = parser.parse_args()

    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_devices = torch.cuda.device_count() 
    print(f"DEVICE: {device}")
    print(f"Num devices: {num_devices}")
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size
    transform = Compose([ToTensor(), Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    inv_normalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.255])
    train_dataset = skate_data_pretrain(['data/batb1k/frames128'], device, transform=transform)
    # val_dataset = skate_data_pretrain(['data/batb1k/frames'], device, transform=transform)
    val_dataset = train_dataset
    print(f'Batch size {load_batch_size}')
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    

    model = MAE_ViT(mask_ratio=args.mask_ratio, image_size=128, patch_size=8).to(device)
    if num_devices > 1:
        model = nn.DataParallel(model)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        # writer.add_scalar('mae_pretrain_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average training loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i] for i in range(16)]) # [16, 3, 32, 32]
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            img = inv_normalize(img) 
            torchvision.utils.save_image(img.cpu(), f"logs/val128.jpg")
        
        ''' save model '''
        torch.save(model, args.model_path)