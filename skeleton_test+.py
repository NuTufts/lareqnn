# More in depth skeleton test file to print out all model weights, layers, and values at running for troubleshooting
# TODO get layers from model instead of listing them

import os,sys
import torch
import numpy as np
import MinkowskiEngine as ME
from tqdm import trange, tqdm

from engine_lightning import LitEngineResNetSparse
from lartpcdataset import lartpcDatasetSparse, PreProcess
import lovely_tensors as lt

config = dict(
            train_datapath = "../PTrain",
            test_datapath = "../PilarDataTest",
            model = "ResNet34",
            batch_size = 64,
            lr = 1e-3,
            weight_decay = 1e-2,
            grad_batches = 1,
            epochs = 200,
            pin_memory = False,
            grad_clip = 0.5,
            steps_per_epoch = 100,
            normalize = False, 
            clip = True, 
            sqrt = True, 
            norm_mean = 0.08, 
            norm_std = 1.14, 
            clip_min = 0.0, 
            clip_max = 1.0
        )


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
layers = ["conv1","layer1","layer2","layer3","layer4","conv5","glob_pool","final"]
# layers = ["conv1","layer1","layer2","layer3","layer4","glob_pool","final"]

DEBUG = False
PRINT_OUTPUT = False
PRINT_LABEL_TEXTS = False
PRINT_MIN_MAX = False

DEVICE = torch.device("cuda")
dataset = lartpcDatasetSparse( root=config["train_datapath"],device = DEVICE)
loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn = ME.utils.batch_sparse_collate,
        #pin_memory=True,
        num_workers=8)
classes = dataset.class_to_idx
invclasses = {v: k for k, v in classes.items()}
PreP = PreProcess(config["normalize"],
                  config["clip"],
                  config["sqrt"],
                  config["norm_mean"],
                  config["norm_std"],
                  config["clip_min"],
                  config["clip_max"])
print(invclasses)


def train_one_batch(batchdata,losses):
    coords, feats, labels = batchdata # data batch, labels

#     if [invclasses[int(i)] for i in batchdata[2].cpu()][0]!="Proton":
#         return losses
    coords = coords.type(torch.IntTensor).to(DEVICE)
    feats = PreP(feats.unsqueeze(1).type(torch.FloatTensor).to(DEVICE))
    print(feats.shape)
    st = ME.SparseTensor( features=feats,
                             coordinates=coords)

    truelabel = [invclasses[int(i)] for i in labels.cpu()]

    #code for splitting array
    coordsloc = st.coordinates[:,0].cpu()  # Get coordinates

    splitloc = np.where(np.diff(coordsloc,prepend=np.nan))[0] # Find locations where coordinates change

    splitfeatures = np.split(st.features.cpu(),splitloc[1:]) # split feature array for each batch

    
    if PRINT_MIN_MAX:
        print("min",[i.min() for i in splitfeatures])
        print("max",[i.max() for i in splitfeatures])

    if False:
        # set to true to just check data loader
        sys.exit(0)

    loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)

    engine = LitEngineResNetSparse(config, dataset, dataset ).to(DEVICE)
    #print(engine.model)
    if DEBUG:
        for layer in layers:
            eval(f"engine.model.{layer}.register_forward_hook(get_activation('{layer}'))")

    #optimizer = torch.optim.SGD(engine.parameters(), lr=config["lr"])
    optimizer = torch.optim.AdamW( engine.parameters(), lr=config["lr"], weight_decay=config["weight_decay"] )

    #NITERS = 1   # batchsize=1
    NITERS = config["epochs"] # batchsize=4
    
    losses = train_one_epoch(st,labels,loss_fn,engine,optimizer,NITERS,losses)
    
    return losses


def train_one_epoch(st,labels,loss_fn,engine,optimizer,NITERS,losses):
    
    
    losssmall = np.array([])
    
    
    pbar = tqdm(range(NITERS))
    for istep in pbar:
        losssmall, out = train_one_step(istep,st,labels,loss_fn,engine,optimizer,losssmall,pbar)
    
    
    losses=np.append(losses,losssmall)
    
    
    if PRINT_OUTPUT:
        print(f"final {out.features.cpu().numpy()}")
        
    
    truelabel = [invclasses[int(i)] for i in labels.cpu()]
    
    predictedlabel = [invclasses[int(i)] for i in out.features.argmax(1).cpu()]
        
        
    if PRINT_LABEL_TEXTS:    
        print(f"True {truelabel},\n Pred {predictedlabel}")
    similarity = compute_confusion_matrix(labels.cpu(),out.features.argmax(1).cpu())
    print(f"correct = {np.trace(similarity)}/{similarity.sum()},conf=\n{similarity}")
    return losses

    
    

def train_one_step(istep,st,labels,loss_fn,engine,optimizer,losssmall,pbar):
    optimizer.zero_grad()

    #print("ITER ",istep)
    out = engine.model( st )
    #print(activation)
    if DEBUG:
        for layer in layers:
            print(layer,lt.lovely(activation[layer].features.cpu(), depth=1))
    #print("out: ",out.shape)
    #print("out min: ",out.features.min())
    #print("out max: ",out.features.max())
    
    maxname = "test"
    maxvalue = 0.5
    for name, param in engine.model.named_parameters():
        if param.requires_grad:
            if DEBUG:
                print(name,lt.lovely(param.data))
            maxdata = torch.max(torch.abs(param.data))
            if maxdata > maxvalue:
                maxvalue = maxdata
                maxname = name
            #print (name, param.data)

    if istep == 0:
        true_label = labels.cpu().detach()
        pred_label = out.features.cpu().detach().numpy()
        
        
        if PRINT_OUTPUT:
            print(f"initial {pred_label}")
            
        true_label_name = [invclasses[int(i)] for i in true_label]
        pred_label_name = [invclasses[int(i)] for i in pred_label.argmax(1)]
        
        if PRINT_LABEL_TEXTS:
            print(f"True {truelabel},\n Pred {predictedlabel}")
        
        similarity = compute_confusion_matrix(true_label,pred_label.argmax(1))
        print(f"initial_correct = {np.trace(similarity)}/{similarity.sum()},conf=\n{similarity}")


    loss = loss_fn( out.features, labels.type(torch.LongTensor).to(DEVICE) )
    #print(out.features)


    pbar.set_description(f"ITER {istep}, Loss {loss.detach():.3f}, {maxname} = {maxvalue:.3f}" )
    losssmall = np.append(losssmall,loss.detach().cpu().numpy())
#         #print("  loss: ",loss.detach())


    loss.backward()
    #torch.nn.utils.clip_grad_norm_(engine.model.parameters(), config["grad_clip"])
    optimizer.step()
    return losssmall, out.detach()



def compute_confusion_matrix(true, pred):
    '''Computes a confusion matrix using numpy for two np.arrays
  true and pred.

  Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"

  However, this function avoids the dependency on sklearn.'''

    K = len(classes) # Number of classes 
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result



if __name__ == "__main__":
    
    DEVICE = torch.device("cuda")
    dataset = lartpcDatasetSparse( root=config["train_datapath"],device = DEVICE)
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

    for i in range(20):

        print(f"\nBatch {i}")

        batchdata = next(iter(loader))

        losses = train_one_batch(batchdata,losses)



    #     print(st)
    #     print(st.features)

    #for istep in range(NITERS):    







    np.save("losses.npy",losses)
