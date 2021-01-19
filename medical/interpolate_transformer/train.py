import torch
import numpy as np
import torchvision
import os 
import sys
import argparse

from torch.utils.data import DataLoader, DistributedSampler
from torch import nn

from model import IPT, SimpleImageDecoder
from dataloader import Kits19
from transformers import Transformer
from position_encoding import PositionEmbeddingLearned
from resnet import ResNet_trans
from tutils import *
import wandb
from config import args
# !wandb login

# model_dir
# backbone = torchvision.models.resnet18(pretrained=True) # backbone = torchvision.models.resnet50(pretrained=True)
backbone            = ResNet_trans()
transformer         = Transformer(d_model=args.hidden_dim, nhead=args.nheads, num_encoder_layers=args.enc_layers, num_decoder_layers=args.dec_layers, dim_feedforward=args.dim_feedforward)
position_encoding   = PositionEmbeddingLearned(args.hidden_dim // 2)
query_embed         = nn.Embedding(args.num_queries, args.hidden_dim)
image_decoder       = SimpleImageDecoder(input_dim=256, sr_size=(512,22), times=2)
n_parameters = sum(p.numel() for p in backbone.parameters()             if p.requires_grad); print('backbone: number of params:', n_parameters)
n_parameters = sum(p.numel() for p in transformer.parameters()          if p.requires_grad); print('transformer: number of params:', n_parameters)
n_parameters = sum(p.numel() for p in position_encoding.parameters()    if p.requires_grad); print('position_encoding: number of params:', n_parameters)
n_parameters = sum(p.numel() for p in query_embed.parameters()          if p.requires_grad); print('query_embed: number of params:', n_parameters)

backbone.cuda()
transformer.cuda()
position_encoding.cuda()
query_embed.cuda()
image_decoder.cuda()

# image_decoder = SimpleImageDecoder()
# model = IPT(backbone=backbone, transformer=transformer, image_decoder=image_decoder)

optimizer       = torch.optim.AdamW(backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler    = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
criterion       = torch.nn.L1Loss()

kwargs = {'num_workers': 8, 'pin_memory': True}
dataset_train = Kits19(load_mod="sr_x2_slice", datadir="/home1/quanquan/datasets/kits19/resampled_data")
dataset_train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)

p("Start training") 
for epoch in range(0, args.epochs+1):
    loss_total = 0.
    backbone.train()
    datalen = len(dataset_train)
    for batch_idx, data in enumerate(dataset_train_loader):
        tinput, target = data[0].cuda(), data[1].cuda()
        # import ipdb; ipdb.set_trace()
        optimizer.zero_grad()
        output      = backbone(tinput)
        b,c,m,n     = output.size()
        mask        = torch.zeros((b,m,n), dtype=torch.bool).cuda()
        pos         = position_encoding(output)
        # import ipdb; ipdb.set_trace()
        output2, _ = transformer(output, mask, query_embed.weight, pos)
        # torch.einsum('(b c n*n) -> ()')
        output_feature = image_decoder(output2[-1])
        loss = criterion(output_feature, target)
        loss.backward()
        optimizer.step()
        # import ipdb; ipdb.set_trace()
        loss_total += loss.item()
        print(f"\r\t Epoch[{(epoch+1):3d}/{args.epochs:3d}] \t\tLoss: {loss_total*1./(batch_idx+1):.10f} \t\t Bs: [{batch_idx:5d}/{datalen:5d}]", end="")
    
    
    # ----------------  Records and Eval  ----------------------------------------------
    modelname = args.modelname
    print(f"Basic loss: {loss_total*1./datalen} or {loss_total*1./batch_idx+1}")
    wandb.log({"Basic loss": loss_total*1./datalen })
    if epoch % 10 == 0 and epoch != 0:
        torch.save(backbone.state_dict(), tfilename(args.output_dir, "model", f"backbone_{modelname}_{epoch}.pkl"))
        torch.save(transformer.state_dict(), tfilename(args.output_dir, "model", f"transformer_{modelname}_{epoch}.pkl"))
        torch.save(position_encoding.state_dict(), tfilename(args.output_dir, "model", f"position_enc_{modelname}_{epoch}.pkl"))
        torch.save(query_embed.state_dict(), tfilename(args.output_dir, "model", f"query_embed_{modelname}_{epoch}.pkl"))
        torch.save(image_decoder.state_dict(), tfilename(args.output_dir, "model", f"imagedecoder_{modelname}_{epoch}.pkl"))

    lr_scheduler.step()
