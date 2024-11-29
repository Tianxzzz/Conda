import random
import torch
import torch.nn as nn
import numpy as np

from utils.DataLoader import Data


def train_graph_model_train(model: nn.Module):   
    model.train()
    if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # training, only use training graph
        model[0].set_neighbor_sampler(train_neighbor_sampler)
    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
        # reinitialize memory of memory-based models at the start of each epoch
        model[0].memory_bank.__init_memory_bank__()

    # store train losses and metrics
    train_losses, train_metrics = [], []
    train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
    for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
        train_data_indices = train_data_indices.numpy()
        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
            train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
            train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

        _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
        batch_neg_src_node_ids = batch_src_node_ids


        if args.model_name in ['TGAT', 'CAWN', 'TCL']:
            # two Tensors, with shape (batch_size, node_feat_dim)
            batch_src_node_embeddings, batch_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                    dst_node_ids=batch_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times,
                                                                    num_neighbors=args.num_neighbors)
            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                    dst_node_ids=batch_neg_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times,
                                                                    num_neighbors=args.num_neighbors)
        elif args.model_name in ['JODIE', ' DyRep', 'TGN']:
            
            batch_src_node_embeddings, batch_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                    dst_node_ids=batch_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times,
                                                                    edge_ids=batch_edge_ids,
                                                                    edges_are_positive=True,
                                                                    num_neighbors=args.num_neighbors)
            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                    dst_node_ids=batch_neg_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times,
                                                                    edge_ids=None,
                                                                    edges_are_positive=False,
                                                                    num_neighbors=args.num_neighbors)
        elif args.model_name in ['GraphMixer']:

            batch_src_node_embeddings, batch_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                    dst_node_ids=batch_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times,
                                                                    num_neighbors=args.num_neighbors,
                                                                    time_gap=args.time_gap)
            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                    dst_node_ids=batch_neg_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times,
                                                                    num_neighbors=args.num_neighbors,
                                                                    time_gap=args.time_gap)
        elif args.model_name in ['DyGFormer']:
            
            batch_src_node_embeddings, batch_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                    dst_node_ids=batch_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times)
            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                    dst_node_ids=batch_neg_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        
        positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
        negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

        predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
        labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

        loss = loss_func(input=predicts, target=labels)
        train_losses.append(loss.item())

        train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
            model[0].memory_bank.detach_memory_bank()


    