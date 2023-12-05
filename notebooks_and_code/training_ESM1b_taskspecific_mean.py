import esm
import sys
import math
import time
import copy
import random
import logging
import argparse
import numpy as np
import pandas as pd
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('.\\additional_code')
from util.process import *
from util.constants import *

CURRENT_DIR = os.getcwd()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
fhandler = logging.FileHandler(filename='log_TSP_lr4_1e-6_phylo.txt', mode='a')
logger.addHandler(fhandler)

# Check if multiple GPUs are available
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_ids = list(range(torch.cuda.device_count()))
    gpus = len(device_ids)
    logging.info(str(gpus) + ' GPUs detected')
else:
    device = torch.device('cpu')
    logging.info('Using CPU as device')
   
    
class FullModel(nn.Module):
    def __init__(self, in_dim, model):
        super(FullModel, self).__init__()
        self.model = model
        self.fc1 = nn.Linear(in_dim,256)
        self.fc2 = nn.Linear(256,32)
        self.fc3 = nn.Linear(32,1)
    
    def forward(self, data, subs, rep_layer, gpu, use_cuda):
        output = self.model(data, repr_layers=[rep_layer])
        output = output["representations"][rep_layer]
        if use_cuda:
            output = output.cuda(gpu, non_blocking=True)
        output = torch.mean(output, 1)
        x = torch.cat((output,subs), dim=1)
        if use_cuda:
            x = x.cuda(gpu, non_blocking=True)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
    
def train(gpu, model_full, data_loader, optimizer, use_cuda, args):
    model_full.train()
    total_loss = 0.0
    start_time = time.time()
    
    if args.esm_msa:
        rep_layer = 33
    else:
        rep_layer = 12
    
    criterion = nn.BCELoss()

    for batch_idx, (labels, strs, toks, substrates, bindings) in enumerate(data_loader):
        n_seq = toks.size(0)
        logging.info(f" {gpu}: Training {batch_idx + 1} of {len(data_loader)} batches ({n_seq} sequences)")

        data = toks
        subs = torch.Tensor([[int(c) for c in s] for s in substrates])
    
        if use_cuda:
            data = data.cuda(gpu, non_blocking=True)
            subs = subs.cuda(gpu, non_blocking=True)
    
        # Get prediction for the ESM model
        output = model_full(data, subs, rep_layer, gpu, use_cuda)

        # Calculate loss
        optimizer.zero_grad()
        output_flat = output.reshape(-1)
        target_flat = torch.Tensor(bindings).reshape(-1)
        if use_cuda:
            output_flat = output_flat.cuda(gpu, non_blocking=True)
            target_flat = target_flat.cuda(gpu, non_blocking=True)

        loss = criterion(output_flat, target_flat)
        
        loss.backward()	
        optimizer.step()

#        for name, param in model_full.named_parameters():
#            if param.grad is None:
#                print(name)

        avg_loss = loss.item()
        total_loss += avg_loss
    return total_loss / len(data_loader)

def evaluate(gpu, model_full, data_loader, use_cuda, args):
    model_full.eval()
    val_loss = 0.
    correct_pred = 0.
    count_samples = 0
    
    if args.esm_msa:
        rep_layer = 33
    else:
        rep_layer = 12
    
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for batch_idx, (labels, strs, toks, substrates, bindings) in enumerate(data_loader):
            n_seq = toks.size(0)
            logging.info(f" {gpu}: Evaluating {batch_idx + 1} of {len(data_loader)} batches ({n_seq} sequences)")
            
            data = toks
            subs = torch.Tensor([[int(c) for c in s] for s in substrates])

            if use_cuda:
                data = data.cuda(gpu, non_blocking=True)
                subs = subs.cuda(gpu, non_blocking=True)

            # Get prediction for the ESM model
            output = model_full(data, subs, rep_layer, gpu, use_cuda)
            
            # Calculate loss
            output_flat = output.reshape(-1)
            target_flat = torch.Tensor(bindings).reshape(-1)
            if use_cuda:
                output_flat = output_flat.cuda(gpu, non_blocking=True)
                target_flat = target_flat.cuda(gpu, non_blocking=True)

            loss = criterion(output_flat, target_flat)
            
            true = target_flat.detach().cpu().numpy()
            pred = np.round(output_flat.detach().cpu().numpy())
            correct_pred += np.sum(true == pred)
            count_samples += len(true)
            
                        
            avg_loss = loss.item()
            val_loss += avg_loss
    return(val_loss / len(data_loader), correct_pred / count_samples)


def train_eval(gpu, args):
    if args.use_cuda:
        logging.info(f" {gpu}: Running train_eval, DDP GPU rank {gpu}")
    else:
        logging.info(f" {gpu}: Running train_eval on CPU")
        
    ## Vars ##================================
    
    token_per_batch = 1024*2
    
    load_subs = False
    load_esm = False
    save_models = True
    args.esm_msa = True             # True: uses ESM model, False: uses MSA model. Bert = MSA or ESM
    PATH_load_bert = 'models/model.pkl'

    ##========================================
        
    alphabet = Alphabet(proteinseq_toks['toks'])
    if args.esm_msa:
        model, _ = esm.pretrained.esm1b_t33_650M_UR50S()
        logging.info(f" {gpu}: Using ESM bert model")
    else:
        model, _ = esm.pretrained.esm_msa1_t12_100M_UR50S()
        logging.info(f" {gpu} Using MSA bert model")
    
    d_subs = 1024
    if args.esm_msa:
        d_model = 1280
    else:
        d_model = 768

    model_full = FullModel(d_model+d_subs, model)
    
        
    if load_esm:
        if args.use_cuda:
            model_full.load_state_dict(torch.load(PATH_load_bert))
        else:
            model_full.load_state_dict(torch.load(PATH_load_bert), map_location=torch.device('cpu'))


    logging.info(f" {gpu}: Models loaded!!")

    
    training_dataframe =join(CURRENT_DIR, "data", "ESM1b_training",  'train_data_ESM_training.pkl')
    validation_dataframe =join(CURRENT_DIR, "data", "ESM1b_training", 'validation_data_ESM_training.pkl')

    train_df = pd.read_pickle(training_dataframe)
    test_df = pd.read_pickle(validation_dataframe)
    test_df = test_df.loc[test_df["type"] == "exp"]
    
    train_df = train_df.sample(frac = 1)

    train_dataset = FastaSubsDataset.from_df(data =train_df)
    val_dataset = FastaSubsDataset.from_df(data = test_df)

    train_batches = train_dataset.get_batch_indices(token_per_batch, extra_toks_per_seq=1)
    val_batches = val_dataset.get_batch_indices(token_per_batch, extra_toks_per_seq=1)
    
    logging.info(f' {gpu}: Read train data with {len(train_dataset)} sequences')
    logging.info(f' {gpu}: Read val data with {len(val_dataset)} sequences')
    
    
    # Dropping extra batches from the dataset
    ws = args.world_size
    if len(train_batches) % ws != 0:
        num_samples = math.ceil((len(train_batches) - ws) / ws)
    else:
        num_samples = math.ceil(len(train_batches) / ws)
    train_total_size = num_samples * ws
    logging.info("Selected {} of {} train batches, dropped {}/{}".format(train_total_size, 
                        len(train_batches), len(train_batches)-train_total_size, len(train_batches)))

    if len(val_batches) % ws != 0:
        num_samples = math.ceil((len(val_batches) - ws) / ws)
    else:
        num_samples = math.ceil(len(val_batches) / ws)
    val_total_size = num_samples * ws
    logging.info("Selected {} of {} val batches, dropped {}/{}".format(val_total_size, 
                        len(val_batches), len(val_batches)-val_total_size, len(val_batches)))

    ## set require_grad = False for unused params ##
    nograd = {
        "contact_head.regression.weight",
        "contact_head.regression.bias", 
        "lm_head.bias", "lm_head.dense.weight", 
        "lm_head.dense.bias", "lm_head.layer_norm.weight", 
        "lm_head.layer_norm.bias"
    }
    logging.info(f" {gpu}: Not computing gradients for:")
    for name, param in model.named_parameters():
        if name in nograd:
            logging.info(f" {gpu}: {name}")
            param.requires_grad = False
            
            
    params = list(model_full.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    if args.use_cuda:
        rank = args.nr * args.gpus + gpu
        setup(rank, args.world_size)
        torch.manual_seed(0)
        torch.cuda.set_device(gpu)
        model_full.cuda(gpu)
        # Wrap the model and load dataset
        model_full = DDP(model_full,device_ids=[gpu])
    else:
        rank = gpu

    # Take indexes starting from {rank} till end with a gap of {world_size}
    train_batch = train_batches[rank:train_total_size:args.world_size]
    random.shuffle(train_batch)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=alphabet.get_batch_converter(for_subs=True), 
        batch_sampler=train_batch
    )


    # Take indexes starting from {rank} till end with a gap of {world_size}
    val_batch = val_batches[rank:val_total_size:args.world_size]

    val_loader = torch.utils.data.DataLoader(
        val_dataset, collate_fn=alphabet.get_batch_converter(for_subs=True), 
        batch_sampler=val_batch
    )
    
    best_model = model_full
    best_val_loss = float('inf')

    # Train model
    for epoch in range(args.epochs):
        PATH_save_bert = '/gpfs/project/alkro105/ESM/models/model_ESM_binary_A100_epoch_' +str(epoch) + '_mean_ts.pkl'
        start_time = time.time()
        # train
        train_loss = train(gpu, model_full,
                           train_loader, optimizer, args.use_cuda, args)
        # evaluate
        val_loss, val_acc = evaluate(gpu, model_full,
                            val_loader, args.use_cuda, args)

        logging.info('-' * 100)
        logging.info('| Device: {:2d} | End of epoch: {:3d} | Time taken: {:5.2f}s |\n'
                '| Val loss: {:2.5f} | Train loss: {:4.5f} | Val accuracy {:2.5f} |'
                    .format(rank, epoch, (time.time() - start_time), val_loss, train_loss, val_acc))
        logging.info('-' * 100)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model_full
            if save_models and gpu == 0:
                torch.save(best_model.state_dict(), PATH_save_bert)
        scheduler.step()
    
    if args.use_cuda:
        cleanup()
        
        
if __name__ == '__main__':
    sys.argv = ['']
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes, goes from 0 to args.nodes-1')
    parser.add_argument('--epochs', default=100, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=4*1e-6, type=float,
                        help='learning rate for optimizer')
    parser.add_argument('--world_size', default=10, type=int,
                        help='world_size')
    parser.add_argument('--use_cuda', default=True, type=bool,
                        help='Use cuda')
    parser.add_argument('--esm_msa', default=True, type=bool,
                        help='ESM(True) or MSA(False) as the BERT layer')
    
    args = parser.parse_args()    
    if(device == torch.device('cpu')):
        args.use_cuda = False
    else:
        args.use_cuda = True
        args.gpus = gpus

    args.world_size = args.gpus * args.nodes
    logging.info(args)
    
    if args.use_cuda:
        mp.spawn(train_eval, nprocs=args.gpus, args=(args,))
    else:
        train_eval(0, args)