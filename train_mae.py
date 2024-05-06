import argparse
import math
import time
import torch
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import *
from utils import setup_seed, Add_Legs
from data import skate_data
from torchvision.transforms import Compose, ToTensor, ColorJitter, Normalize, RandomAffine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_encoder_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None) 
    parser.add_argument('--output_model_path', type=str, default='checkpoints/skateMAE_epoch_')

    args = parser.parse_args()

    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_devices = torch.cuda.device_count() 
    print(f"Device: {device}")
    print(f"Num GPUs: {num_devices}")

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_transform = Compose([ToTensor(), 
                               Add_Legs('data/batb1k/leg_masks'), 
                               RandomAffine(degrees=0, translate=(0.3,0.3)), 
                               ColorJitter(0.3, 0.3, 0.3), 
                               Normalize(0.5, 0.5)])
    val_transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    train_dataset = skate_data('data/batb1k/synthetic_frames', 'data/batb1k/synthetic_frame_poses_FIXED.csv', device, train_transform)
    val_dataset = skate_data('data/batb1k/test_synthetic_frames', 'data/batb1k/test_synthetic_frame_poses_FIXED.csv', device, val_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=2)
    print(f'Batch size: {load_batch_size}')
    if args.model_path is not None:
        model = torch.load(args.model_path, map_location=device)
        model = model.to(device)
        print(f"Loading pretrained model from {args.model_path}")
    elif args.pretrained_encoder_path is not None:
        model = torch.load(args.pretrained_encoder_path, map_location='cpu')
        print(f"Loading encoder from {args.pretrained_encoder_path}")
        model = skateMAE(model.encoder, embed_dim=124).to(device)
        # writer = SummaryWriter(os.path.join('logs', 'cifar10', 'pretrain-cls'))
    else:
        model = skateMAE(MAE_ViT().encoder, embed_dim=124).to(device)
        # writer = SummaryWriter(os.path.join('logs', 'cifar10', 'scratch-cls'))

    # if num_devices > 1:
    #     model = nn.DataParallel(model)

    loss_fn = torch.nn.MSELoss()
    weights = torch.tensor([10.0, 1.0, 1.0], device=device)
    acc_fn = lambda pred, label: torch.mean((torch.round(pred.detach()) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)
    
    best_val_acc = 0
    best_val_loss = float('inf')
    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        time_start = time.time()
        for img, dist_label, elev_label, azim_label, _ in tqdm(iter(train_dataloader)):
            img = img.to(device)
            dist_label = dist_label.to(device)
            elev_label = elev_label.to(device)
            azim_label = azim_label.to(device)

            step_count += 1
            dist_preds, elev_preds, azim_preds = model(img)
            loss =  loss_fn(dist_preds.squeeze(), dist_label) * weights[0] + \
                    loss_fn(elev_preds.squeeze(), elev_label) * weights[1] + \
                    loss_fn(azim_preds.squeeze(), azim_label) * weights[2]
            acc = torch.mean(torch.stack((acc_fn(elev_preds, elev_label), acc_fn(azim_preds, azim_label))))
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
            
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, dist_label, elev_label, azim_label, _ in tqdm(iter(val_dataloader)):
                img = img.to(device)
                dist_label = dist_label.to(device)
                elev_label = elev_label.to(device)
                azim_label = azim_label.to(device)

                dist_preds, elev_preds, azim_preds = model(img)
                loss =  loss_fn(dist_preds.squeeze(), dist_label) * weights[0] + \
                    loss_fn(elev_preds.squeeze(), elev_label) * weights[1] + \
                    loss_fn(azim_preds.squeeze(), azim_label) * weights[2] 
                acc = torch.mean(torch.stack((acc_fn(elev_preds, elev_label), acc_fn(azim_preds, azim_label))))
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')
            print(f'Epoch time: {time.time()-time_start} seconds')  

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f'saving best model with val loss {best_val_loss} and acc {avg_val_acc} at {e} epoch!')       
            torch.save(model, f'{args.output_model_path}{e}_valloss_{best_val_loss}_TRANSLATIONS.pt')

        # writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
        # writer.add_scalars('cls/acc', {'train' : avg_train_acc, 'val' : avg_val_acc}, global_step=e)