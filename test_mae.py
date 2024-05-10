import argparse
import math
import torch
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed
from data import skate_data_synth_test
# python test_mae --model_path #SBATCH -N 1 -n 4 --gres=gpu:1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=128)
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

    transform = Compose([ToTensor(), Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    dataset = skate_data_synth_test('data/batb1k/test_synthetic_frames128', 'data/batb1k/poses128.csv', transform)
    dataloader = torch.utils.data.DataLoader(dataset, load_batch_size, shuffle=False, num_workers=2)
    
    print(f'Batch size: {load_batch_size}')
    model = torch.load(args.model_path, map_location='cpu').module
    model = model.to(device)

    loss_fn = torch.nn.MSELoss()
    weights = [1.0, 1.0, 1.0]
    acc_fn = lambda pred, label: torch.mean((torch.round(pred.detach()) == label).float())

    model.eval()
    with torch.no_grad():
        losses = []
        dist_losses = []
        elev_losses = []
        azim_losses = []
        acces = []
        for img, dist_label, elev_label, azim_label, _ in tqdm(iter(dataloader)):
            N = img.shape[0]
            img = img.to(device)
            dist_label = dist_label.to(device)
            elev_label = elev_label.to(device)
            azim_label = azim_label.to(device)

            dist_preds, elev_preds, azim_preds = model(img)
            dist_loss = loss_fn(dist_preds.squeeze(), dist_label) * weights[0] 
            elev_loss = loss_fn(elev_preds.squeeze(), elev_label) * weights[1] 
            azim_loss = loss_fn(azim_preds.squeeze(), azim_label) * weights[2] 
            loss = dist_loss + elev_loss + azim_loss
            
            acc = torch.mean(torch.stack((acc_fn(elev_preds, elev_label), acc_fn(azim_preds, azim_label))))
            losses.append(loss.item() / N)
            dist_losses.append(dist_loss.item() / N)
            elev_losses.append(elev_loss.item() / N)
            azim_losses.append(azim_loss.item() / N)
            acces.append(acc.item() / N)

        avg_val_loss = sum(losses) / len(losses)
        avg_val_acc = sum(acces) / len(acces)
        avg_dist_loss = sum(dist_losses) / len(dist_losses)
        avg_elev_loss = sum(elev_losses) / len(elev_losses)
        avg_azim_loss = sum(azim_losses) / len(azim_losses)
        print(f'Average test loss is {avg_val_loss}, average test acc is {avg_val_acc}.')
        print(f'Dist: {avg_dist_loss}, Elev: {avg_elev_loss}, Azim: {avg_azim_loss}')

