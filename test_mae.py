import argparse
import math
import torch
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed
from data import skate_data
# python test_mae --model_path #SBATCH -N 1 -n 4 --gres=gpu:1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--model_path', type=str, default='')

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

    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    test_dataset = skate_data('data/batb1k/test_synthetic_frames', 'data/batb1k/test_synthetic_frame_poses.csv', device, transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, load_batch_size, shuffle=False, num_workers=2)
    
    print(f'Batch size: {load_batch_size}')
    model = torch.load(args.model_path, map_location='cpu')
    model = model.to(device)

    loss_fn = torch.nn.MSELoss()
    acc_fn = lambda pred, label: torch.mean((torch.round(pred.detach()) == label).float())

    model.eval()
    with torch.no_grad():
        losses = []
        acces = []
        for img, dist_label, elev_label, azim_label in tqdm(iter(test_dataloader)):
            img = img.to(device)
            dist_label = dist_label.to(device)
            elev_label = elev_label.to(device)
            azim_label = azim_label.to(device)
            dist_preds, elev_preds, azim_preds = model(img)
            loss = loss_fn(dist_preds.squeeze(), dist_label) + loss_fn(elev_preds.squeeze(), elev_label) + loss_fn(azim_preds.squeeze(), azim_label)
            acc = torch.mean(torch.stack((acc_fn(dist_preds, dist_label), acc_fn(elev_preds, elev_label), acc_fn(azim_preds, azim_label))))
            losses.append(loss.detach().item())
            acces.append(acc.item())
        avg_val_loss = sum(losses) / len(losses)
        avg_val_acc = sum(acces) / len(acces)
        print(f'Average test loss is {avg_val_loss}, average test acc is {avg_val_acc}.')  

