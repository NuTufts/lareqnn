# Skeleton code to run the code in a simpler configuration

import os,sys
import torch
import numpy as np
import MinkowskiEngine as ME
from tqdm import trange, tqdm
import MEresnet

#from engine_lightning import LitEngineResNetSparse
from lartpcdataset import lartpcDatasetSparse, PreProcess



config = dict(
            train_datapath = "../PTrain",
            test_datapath = "../PilarDataTest",
            model = "Resnet18",
            batch_size = 4,
            lr = 1e-2,
            weight_decay = 1e-2,
            grad_batches = 1,
            epochs = 1000,
            pin_memory = True,
            grad_clip = 0.5,
            steps_per_epoch = 100,
            normalize = True, 
            clip = True, 
            sqrt = True, 
            norm_mean = 0.65, 
            norm_std = 0.57, 
            clip_min = -1.0, 
            clip_max = 1.0
        )

PreP = PreProcess(config["normalize"],
                                    config["clip"],
                                    config["sqrt"],
                                    config["norm_mean"],
                                    config["norm_std"],
                                    config["clip_min"],
                                    config["clip_max"]
                                    )



DEVICE = torch.device("cuda")
dataset = lartpcDatasetSparse( root=config["train_datapath"],sqrt=True,device = DEVICE)
loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn = ME.utils.batch_sparse_collate,
        #pin_memory=True,
        num_workers=8)
#torch.utils.data.DataLoader( dataset, collate_fn=ME.utils.batch_sparse_collate, batch_size=BATCHSIZE )
losses = np.array([])
classes = dataset.class_to_idx
invclasses = {v: k for k, v in classes.items()}

for i in range(25):
    
    print(f"\nBatch {i}")
    
    batchdata = next(iter(loader))

    coords, feats, labels = batchdata # data batch, labels

    coords = coords.type(torch.IntTensor).to(DEVICE)
    feats = PreP(feats.unsqueeze(1).type(torch.FloatTensor).to(DEVICE))
  
#    print(feat)
 #   print(feats.cpu())
    #     st = ME.SparseTensor(coordinates=coords.to(DEVICE), features=feats.unsqueeze(dim=-1).float().to(DEVICE))
    if [invclasses[int(i)] for i in batchdata[2].cpu()][0]!="Proton":
        continue
    st = ME.SparseTensor( features=feats,
                             coordinates=coords )

    
    truelabel = [invclasses[int(i)] for i in batchdata[2].cpu()]
    
    #code for splitting array
    coordsloc = st.coordinates[:,0].cpu()  # Get coordinates
    
    splitloc = np.where(np.diff(coordsloc,prepend=np.nan))[0] # Find locations where coordinates change
    
    splitfeatures = np.split(st.features.cpu(),splitloc[1:]) # split feature array for each batch
    
    print("min",[i.min() for i in splitfeatures])
    print("max",[i.max() for i in splitfeatures])
    
    if False:
        # set to true to just check data loader
        sys.exit(0)

    loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)

    #engine = LitEngineResNetSparse(config, dataset, dataset ).to(DEVICE)
    #print(engine.model)
    model = MEresnet.ResNet18(in_channels=1, out_channels=5, D=3).to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    #optimizer = torch.optim.AdamW( engine.parameters(), lr=config["lr"], weight_decay=config["weight_decay"] )

    #NITERS = 1   # batchsize=1
    NITERS = config["epochs"] # batchsize=4

    losssmall = np.array([])
    
    pbar = tqdm(range(NITERS))
    
    for istep in pbar:
        optimizer.zero_grad()

        
        #print("ITER ",istep)
        out = model( st )
        #print("out: ",out.shape)
        #print("out min: ",out.features.min())
        #print("out max: ",out.features.max())
        maxname = "test"
        maxvalue = 0.5
        for name, param in model.named_parameters():
            if param.requires_grad:
                maxdata = torch.max(torch.abs(param.data))
                if maxdata > maxvalue:
                    maxvalue = maxdata
                    maxname = name
                #print (name, param.data)
        
        if istep == 0:
            print(f"initial {out.features.cpu().detach().numpy()}")
            predictedlabel = [invclasses[int(i)] for i in out.features.argmax(1).cpu().detach()]
            print(f"True {truelabel}, Pred {predictedlabel}")
        
        
        loss = loss_fn( out.features, batchdata[2].type(torch.LongTensor).to(DEVICE) )
        #print(out.features)
        
        
        pbar.set_description(f"ITER {istep}, Loss {loss.detach():.3f}, {maxname} = {maxvalue:.3f}" )
        losssmall = np.append(losssmall,loss.detach().cpu().numpy())
#         #print("  loss: ",loss.detach())
        

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(engine.model.parameters(), 1.0)
        optimizer.step()
    losses=np.append(losses,losssmall)
    
    print(f"final {out.features.cpu().detach().numpy()}")
    predictedlabel = [invclasses[int(i)] for i in out.features.argmax(1).cpu()]
    print(f"True {truelabel}, Pred {predictedlabel}")
    
#     print(st)
#     print(st.features)
    
    #for istep in range(NITERS):

    
    

    
    
    
    
    
    
    
    
np.save("losses.npy",losses)
