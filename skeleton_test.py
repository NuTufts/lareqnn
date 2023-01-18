import os,sys
import torch
import numpy as np
import MinkowskiEngine as ME

from engine_lightning import LitEngineResNetSparse
from lartpcdataset import lartpcDatasetSparse

DEVICE = torch.device("cuda")
BATCHSIZE=4
dataset = lartpcDatasetSparse( root="data/",device = DEVICE)
loader = torch.utils.data.DataLoader( dataset, collate_fn=ME.utils.batch_sparse_collate, batch_size=BATCHSIZE )
batchdata = next(iter(loader))
print(batchdata)
if False:
    # set to true to just check data loader
    sys.exit(0)

loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)

engine = LitEngineResNetSparse( dataset, dataset ).to(DEVICE)
print(engine.model)

optimizer = torch.optim.AdamW( engine.parameters(), lr=1.0e-2, weight_decay=1.0e-5 )

#NITERS = 1   # batchsize=1
NITERS = 1000 # batchsize=4

for istep in range(NITERS):

    optimizer.zero_grad()

    st = ME.SparseTensor( features=batchdata[1].unsqueeze(1).type(torch.FloatTensor).to(DEVICE),
                          coordinates=batchdata[0].type(torch.IntTensor).to(DEVICE) )
    
    print("ITER ",istep)
    out = engine.model( st )
    #print("out: ",out.shape)
    #print("out min: ",out.features.min())
    #print("out max: ",out.features.max())

    loss = loss_fn( out.features, batchdata[2].type(torch.LongTensor).to(DEVICE) )
    print("  loss: ",loss.detach())

    loss.backward()
    optimizer.step()


