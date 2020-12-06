import logging
import os
import sys
import tempfile
from glob import glob

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

import monai
from monai.data import ArrayDataset, create_test_image_2d
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, AddChannel, AsDiscrete, Compose, LoadImage, RandRotate90, RandSpatialCrop, ScaleIntensity, ToTensor, LoadNumpy
from monai.visualize import plot_2d_or_3d_image
import numpy as np
from tutils import *

# ---------------  hyper-params -------------------------
output_dir = tdir("output/seg-monai-simple/", generate_name())
writer = SummaryWriter(tdir(output_dir, "summary"))
max_epoch = 200

train_imtrans = Compose(
    [
        LoadNumpy(data_only=True),
        ScaleIntensity(),
        AddChannel(),
        RandSpatialCrop((96, 96), random_size=False),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ToTensor(),
    ]
)
train_segtrans = Compose(
    [
        LoadNumpy(data_only=True),
        AddChannel(),
        RandSpatialCrop((96, 96), random_size=False),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ToTensor(),
    ]
)


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # --------------- Dataset  ---------------
    
    datadir1 = "/home1/quanquan/datasets/lsw/benign_65/fpAML_55/slices/"
    image_files = np.array([x.path for x in os.scandir(datadir1+"image") if x.name.endswith(".npy")])
    label_files = np.array([x.path for x in os.scandir(datadir1+"label") if x.name.endswith(".npy")])
    image_files.sort()
    label_files.sort()
    # --- ??? what's up ???
    train_files = [{"img":img, "seg":seg} for img, seg in zip(image_files[:-20], label_files[:-20])]
    val_files   = [{"img":img, "seg":seg} for img, seg in zip(image_files[-20:], label_files[-20:])]
    # print("files", train_files[:20])
    # print(val_files)
   
    val_imtrans = Compose([LoadNumpy(data_only=True), ScaleIntensity(), AddChannel(), ToTensor()])
    val_segtrans = Compose([LoadNumpy(data_only=True), AddChannel(), ToTensor()])
    
    # define array dataset, data loader
    check_ds = ArrayDataset(image_files, train_imtrans, label_files, train_segtrans)
    check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
    im, seg = monai.utils.misc.first(check_loader)
    print(im.shape, seg.shape)
    
    # create a training data loader
    train_ds = ArrayDataset(image_files[:-20], train_imtrans, label_files[:-20], train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    # create a validation data loader
    val_ds = ArrayDataset(image_files[-20:], val_imtrans, label_files[-20:], val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    
    # ---------------  model  ---------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    # ---------------  loss function  ---------------
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    # writer = SummaryWriter(logdir=tdir(output_dir, "sumamry"))
    
    # -------------------  Training ----------------------
    for epoch in range(max_epoch):
        # print("-" * 10)
        # print(f"epoch {epoch + 1}/{10}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"\r\t Training batch: {step}/{epoch_len}, \ttrain_loss: {loss.item():.4f}\t", end="")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"\n\tepoch {epoch + 1} \taverage loss: {epoch_loss:.4f}")
        
        # -------------------  Save Model  ----------------------
        if epoch % 5 == 0:
            def get_lr(optimizer):
                for param_group in optimizer.param_groups:
                    return float(param_group['lr'])
                  
            state = {'epoch': epoch + 1,
                     'lr': get_lr(optimizer),
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict()
                     }
            torch.save(state, tfilename(output_dir, "model", "{}_{}.pkl".format("lsw_monai_simple", epoch)))
    
        # -------------------  Evaluation  -----------------------
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    roi_size = (96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = post_trans(val_outputs)
                    # value, _ = dice_metric(y_pred=val_outputs, y=val_labels)
                    value = dice_metric(y_pred=val_outputs, y=val_labels)
                    metric_count += len(value)
                    metric_sum += value.item() * len(value)
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation2d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
        
        
if __name__ == "__main__":
    main()