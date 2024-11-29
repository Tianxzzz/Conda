import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models_for_diffusion.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models_for_diffusion.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from utils.metrics import get_link_prediction_metrics

from Diffusion_module.ddpm import ModelMeanType
from Diffusion_module.ddpm import GaussianDiffusion
from Diffusion_module.ddpm import DNN
from Diffusion_module.VAE import VAE
from Diffusion_module.VAE import compute_vae_loss


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    # get arguments
    args = get_link_prediction_args(is_evaluation=False)
    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)
    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    
    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
   
    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []


    for run in range(args.num_runs):
        set_random_seed(seed=run)
        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'
        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")
        logger.info(f'configuration is {args}')
        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
       
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        #graph model
        link_predictor = MergeLayer(input_dim=2*node_raw_features.shape[1], hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'graph_model -> {model}')
        logger.info(f'graph_model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)
        task_loss = nn.BCELoss()

        #diffusion model config
        if args.mean_type == 'x0':
            mean_type = ModelMeanType.START_X
        else:
            mean_type = ModelMeanType.EPSILON
        
        diffusion = GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max, args.steps, args.device)
        VAE= VAE(args.input_dim * (args.num_neighbors + 1), args.hidden_dim* (args.num_neighbors + 1), args.latent_dim* (args.num_neighbors + 1))

        in_dim = args.latent_dim * (args.diff_length)
        out_dim = args.latent_dim * (args.diff_length)
        reverse_model = DNN(in_dim, out_dim, args.emb_size, time_type="cat")
        
        diffusion = convert_to_gpu(diffusion, device=args.device)
        VAE = convert_to_gpu(VAE, device=args.device)
        reverse_model = convert_to_gpu(reverse_model, device=args.device)

        VAE_optimizer = create_optimizer(model=VAE, optimizer_name=args.vae_optimizer, learning_rate=args.vae_lr, weight_decay=args.vae_wd)
        diffusion_optimizer = create_optimizer(model=reverse_model, optimizer_name=args.diffusion_optimizer, learning_rate=args.diffusion_lr, weight_decay=args.diffusion_wd)
        
        #train graph_model first
        for epoch in range(args.pre_train_epochs):
            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                model[0].memory_bank.__init_memory_bank__()

            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                if args.model_name in ['TCL']:
                    batch_src_node_embeddings, batch_dst_node_embeddings, batch_src_neighbor_node_ids, batch_dst_neighbor_node_ids = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, batch_neg_src_neighbor_node_ids, batch_neg_dst_neighbor_node_ids = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                            dst_node_ids=batch_neg_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].encode(src_node_features = batch_src_node_embeddings,
                                            src_neighbor_node_ids = batch_src_neighbor_node_ids,
                                            dst_node_features = batch_dst_node_embeddings,
                                            dst_neighbor_node_ids = batch_dst_neighbor_node_ids)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].encode(src_node_features = batch_neg_src_node_embeddings,
                                            src_neighbor_node_ids = batch_neg_src_neighbor_node_ids,
                                            dst_node_features = batch_neg_dst_node_embeddings,
                                            dst_neighbor_node_ids = batch_neg_dst_neighbor_node_ids)                                              
              
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                pos_edge_embeddings = torch.cat([batch_src_node_embeddings, batch_dst_node_embeddings], dim = 1) 
                neg_edge_embeddings = torch.cat([batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings], dim = 1) 
                positive_probabilities = model[1](edge_embeddings = pos_edge_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](edge_embeddings=neg_edge_embeddings).squeeze(dim=-1).sigmoid()
                # positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                # negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
                loss = task_loss(input=predicts, target=labels)
                train_losses.append(loss.item())
                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    model[0].memory_bank.detach_memory_bank()

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=task_loss,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)
            
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)   

            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')

         
            for metric_name, metric_value in val_metrics.items():
                logger.info(f'validate {metric_name}, {metric_value:.4f}')

            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=task_loss,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for new testing nodes
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name, metric_value in test_metrics.items():
                    logger.info(f'test {metric_name}, {metric_value:.4f}')

            val_metric_indicator = []
            for metric_name, metric_value in val_metrics.items():
                val_metric_indicator.append((metric_name, metric_value, True))
            early_stop = early_stopping.step(val_metric_indicator, model)
            if early_stop:
                break

            

        #alternative training
        for iters in range(args.iter_nums):
            #diffusion model
            early_stopping.load_checkpoint(model)
            for epoch in range(args.diffusion_iter_epochs):
                
                model.eval()
                if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                    model[0].set_neighbor_sampler(train_neighbor_sampler)
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    model[0].memory_bank.__init_memory_bank__()         
                VAE.train()
                reverse_model.train()
                train_losses, train_metrics = [], []
                train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
                for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                    train_data_indices = train_data_indices.numpy()
                    batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                        train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                        train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]
                    _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                    batch_neg_src_node_ids = batch_src_node_ids
                    with torch.no_grad():
                        if args.model_name in ['TCL']:
                            batch_src_node_embeddings, batch_dst_node_embeddings, batch_src_neighbor_node_ids, batch_dst_neighbor_node_ids = \
                                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                                    dst_node_ids=batch_dst_node_ids,
                                                                                    node_interact_times=batch_node_interact_times,
                                                                                    num_neighbors=args.num_neighbors)
                            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, batch_neg_src_neighbor_node_ids, batch_neg_dst_neighbor_node_ids = \
                                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                                    dst_node_ids=batch_neg_dst_node_ids,
                                                                                    node_interact_times=batch_node_interact_times,
                                                                                    num_neighbors=args.num_neighbors)
                        else:
                            raise ValueError(f"Wrong value for model_name {args.model_name}!")
                        # pos_edge_embeddings = torch.cat([batch_src_node_embeddings, batch_dst_node_embeddings], dim = 1) 
                        # neg_edge_embeddings = torch.cat([batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings], dim = 1) 
                        # edge_emb = torch.cat([pos_edge_embeddings,neg_edge_embeddings],dim=0)
                    batch_size,num_neighbor,feature = batch_src_node_embeddings.shape
                    x = batch_src_node_embeddings.reshape(batch_size,(args.num_neighbors+1)*172)
                    z, mu, logvar = VAE.encode(x)
                    _,dim = z.shape
                    z = z.reshape(batch_size,(args.num_neighbors+1), int(dim/(args.num_neighbors+1)))
                    z_con = z[:, :(args.num_neighbors+1-args.diff_length), :]
                    z_diff = z[:, (args.num_neighbors+1-args.diff_length):, :]
               
                    terms = diffusion.training_losses(reverse_model, z_diff.view(batch_size,int(dim*(args.diff_length/(args.num_neighbors+1)))))
                    elbo = terms["loss"].mean()  
                    z_diffusion= terms["pred_xstart"].reshape([batch_size,args.diff_length,-1])
                    z_recon = torch.cat([z_con,z_diff],dim=1)
                    x_recon = VAE.decode(z_recon.reshape(batch_size,-1))
                    vae_loss = compute_vae_loss(x_recon, x, mu, logvar)
                    # print(vae_loss)
                    loss = elbo+vae_loss
                    # total_loss += loss
                    train_losses.append(loss.item())
                    VAE_optimizer.zero_grad()
                    diffusion_optimizer.zero_grad()
                    loss.backward()
                    VAE_optimizer.step()
                    diffusion_optimizer.step()
                # print(f"epoch:{epoch+1}, trian loss:{np.mean(train_losses)}")
                
               
            
            # early_stopping.load_checkpoint(model)
            early_stopping.reset()
            for epoch in range(args.graph_iter_epochs):
                
                model[1].train()
                model[0].eval()
                if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                    model[0].set_neighbor_sampler(train_neighbor_sampler)
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    model[0].memory_bank.__init_memory_bank__()
                VAE.eval()
                reverse_model.eval()

                train_losses, train_metrics = [], []
                train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
                for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                    train_data_indices = train_data_indices.numpy()
                    batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                        train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                        train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                    _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                    batch_neg_src_node_ids = batch_src_node_ids

                    if args.model_name in ['TCL']:
                        batch_src_node_embeddings, batch_dst_node_embeddings, batch_src_neighbor_node_ids, batch_dst_neighbor_node_ids = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                                dst_node_ids=batch_dst_node_ids,
                                                                                node_interact_times=batch_node_interact_times,
                                                                                num_neighbors=args.num_neighbors)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, batch_neg_src_neighbor_node_ids, batch_neg_dst_neighbor_node_ids = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                                dst_node_ids=batch_neg_dst_node_ids,
                                                                                node_interact_times=batch_node_interact_times,
                                                                                num_neighbors=args.num_neighbors)
                    else:
                        raise ValueError(f"Wrong value for model_name {args.model_name}!")
                    
                    with torch.no_grad():
                        x = batch_src_node_embeddings.reshape(args.batch_size,-1)
                       
                        z, mu, logvar = VAE.encode(x)
                        z = z.reshape(args.batch_size,args.num_neighbors,-1)
                        z_con = z[:, :(args.num_neighbors+1-args.diff_length), :]
                        z_diff = z[:, (args.num_neighbors+1-args.diff_length):, :]
                        z_diffsion = diffusion.p_sample(reverse_model, z_diff.reshape(args.batch_size,-1), args.sampling_steps, args.sampling_noise)
                        z_da = torch.cat([z_con,z_diff],dim=1)
                        x_da = VAE.decode(z_recon.reshape(args.batch_size,-1))
              
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].encode(src_node_features = x_da,
                                            src_neighbor_node_ids = batch_src_neighbor_node_ids,
                                            dst_node_features = batch_dst_node_embeddings,
                                            dst_neighbor_node_ids = batch_dst_neighbor_node_ids)
                    neg_batch_src_node_embeddings, neg_batch_dst_node_embeddings = \
                        model[0].encode(src_node_features = neg_batch_src_node_embeddings,
                                            src_neighbor_node_ids = neg_batch_src_neighbor_node_ids,
                                            dst_node_features = neg_batch_dst_node_embeddings,
                                            dst_neighbor_node_ids = neg_batch_dst_neighbor_node_ids)   
                    pos_edge_embeddings = torch.cat([batch_src_node_embeddings, batch_dst_node_embeddings], dim = 1) 
                    neg_edge_embeddings = torch.cat([batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings], dim = 1)  

                    positive_probabilities = model[1](edge_embeddings = pos_edge_embeddings).squeeze(dim=-1).sigmoid()
                    negative_probabilities = model[1](edge_embeddings= neg_edge_embeddings).squeeze(dim=-1).sigmoid()
                    
                    predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                    labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
                    loss = task_loss(input=predicts, target=labels)
                    train_losses.append(loss.item())
                    train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
                    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        model[0].memory_bank.detach_memory_bank()
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
                val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                        model=model,
                                                                        neighbor_sampler=full_neighbor_sampler,
                                                                        evaluate_idx_data_loader=val_idx_data_loader,
                                                                        evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                        evaluate_data=val_data,
                                                                        loss_func=task_loss,
                                                                        num_neighbors=args.num_neighbors,
                                                                        time_gap=args.time_gap)
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)        
                logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
                for metric_name in train_metrics[0].keys():
                    logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
                logger.info(f'validate loss: {np.mean(val_losses):.4f}')

                
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f'validate {metric_name}, {metric_value:.4f}')
                

                if (epoch + 1) % args.test_interval_epochs == 0:
                    test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                            model=model,
                                                                            neighbor_sampler=full_neighbor_sampler,
                                                                            evaluate_idx_data_loader=test_idx_data_loader,
                                                                            evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                            evaluate_data=test_data,
                                                                            loss_func=task_loss,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)
                    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        # reload validation memory bank for new testing nodes
                        model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                    logger.info(f'test loss: {np.mean(test_losses):.4f}')
                    for metric_name, metric_value in test_metrics.items():
                        logger.info(f'test {metric_name}, {metric_value:.4f}')

                val_metric_indicator = []
                for metric_name, metric_value in val_metrics.items():
                    val_metric_indicator.append((metric_name, metric_value, True))
                early_stop = early_stopping.step(val_metric_indicator, model)
                if early_stop:
                    break


        # load the best model
        early_stopping.load_checkpoint(model)
        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=task_loss,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)
        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # the memory in the best model has seen the validation edges, we need to backup the memory for new testing nodes
            val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=task_loss,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap)

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name,average_val_metric in val_metrics.items():
                
                logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                val_metric_dict[metric_name] = average_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name,average_test_metric in test_metrics.items():
           
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)  
        test_metric_all_runs.append(test_metric_dict)
        
        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            }   
        result_json = json.dumps(result_json, indent=4)
        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}/"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

       
    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    
    sys.exit()



