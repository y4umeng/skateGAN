import argparse
import math
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize, ColorJitter, RandomAffine
from tqdm import tqdm

from model import *
from utils import setup_seed
from data import *

'''
Pretraining MAE on image reconstruction with both real and synthetic data
Modified from https://github.com/IcarusWizard/MAE/blob/main/mae_pretrain.py
'''

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
    parser.add_argument('--model_path', type=str, default='checkpoints/pretrain128_2.pt')
    parser.add_argument('-load', action='store_true')

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

    # transforms
    real_transform = Compose([ToTensor()])
    synth_transform = Compose([Add_Legs('data/batb1k/leg_masks128')]) 
    shared_transform = Compose([ColorJitter(0.3, 0.3, 0.3), 
                                RandomAffine(degrees=0, translate=(0.3,0.3)),
                                Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]) 
    inv_normalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.255])
    
    # create dataloader
    real_train_dataset = skate_data_pretrain(['data/batb1k/frames128'], transform=real_transform)
    synth_train_dataset = skate_data_synth_pretrain('data/batb1k/synthetic_frames128', 'data/batb1k/backgrounds128', transform=synth_transform)
    combined_dataset = skate_data_combined(real_train_dataset, synth_train_dataset, shared_transform)
    val_dataset = combined_dataset
    print(f'Batch size: {load_batch_size}.')
    dataloader = torch.utils.data.DataLoader(combined_dataset, load_batch_size, shuffle=True, num_workers=4)
    
    # create mae
    if args.load:
        print(f"Loading pretrained MAE from {args.model_path}.")
        model = torch.load(args.model_path)
    else:
        print(f"Initializing new untrained MAE.")
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
            val_img = torch.stack([val_dataset[int(random.random() * len(val_dataset))] for i in range(16)]) # [16, 3, 32, 32]
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            img = inv_normalize(img) 
            torchvision.utils.save_image(img.cpu(), f"logs/val128_random.jpg")
            print("Saved img")
            
        ''' save model '''
        torch.save(model, args.model_path)
        