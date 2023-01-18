import os,sys
import torch
import numpy as np
import MinkowskiEngine as ME

from engine_lightning import LitEngineResNetSparse
from lartpcdataset import lartpcDatasetSparse

DEVICE = torch.device("cuda")
BATCHSIZE=1
dataset = lartpcDatasetSparse( root="data/",device = DEVICE)
batchdata = next(iter(dataset))
print(len(batchdata))
print(batchdata[0].shape)
print(batchdata[1].shape)
print(batchdata[2].shape)

loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)

coords = batchdata[0].type(torch.LongTensor).to(DEVICE) # coords
feats = batchdata[1].unsqueeze(1).type(torch.FloatTensor).to(DEVICE)  # feats
truth = batchdata[2].type(torch.LongTensor).to(DEVICE)
print("coords: ",coords.dtype,coords.shape)
print("feats: ",feats.dtype,feats.shape)
print("truth: ",truth.dtype,truth.shape)

c = batchdata[2]

coords_v = [coords]
feats_v  = [feats]

coords, feats = ME.utils.sparse_collate(coords_v, feats_v )
sparsebatch = [ ME.SparseTensor(features=feats, coordinates=coords) ]

engine = LitEngineResNetSparse( dataset, dataset ).to(DEVICE)
print(engine.model)

optimizer = torch.optim.AdamW( engine.parameters(), lr=0.001, weight_decay=0.01 )

NITERS = 100

for istep in range(NITERS):

    optimizer.zero_grad()
    
    print("ITER ",istep)
    out = engine.model( sparsebatch[0] )
    #print("out: ",out.shape)
    #print("out min: ",out.features.min())
    #print("out max: ",out.features.max())

    loss = loss_fn( out.features, truth )
    print("  loss: ",loss.detach())

    loss.backward()
    optimizer.step()


