import os, sys
BASE_DIR = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import json
import datetime
from collections import defaultdict

from data_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_euclidean(query_vecs, mus, clip_vec=False):
	
	queries_normalized = torch.square((query_vecs - mus))
	distances = torch.sum(queries_normalized, dim= -1)

	return distances

def compute_mahalanobis(query_vecs, mus, sigmas, activation_fn=None, clip_vec=False):

	if not activation_fn == None:
		sigmas = activation_fn(sigmas) + 1.0e-6

	if clip_vec:
		query_vecs = query_vecs.clamp(-100.0, 100.0)
	
	queries_normalized = torch.square(torch.mul((query_vecs - mus), sigmas))
	distances = torch.sum(queries_normalized, dim= -1)

	return distances

def order_embedding_distance(query_vec, source_vecs, device, activation_fn=None, projected=False):
	## Order embedding 
	# distance between query to source is max(0, query_vec - source_vec)
	if not activation_fn == None:
		query_vec = activation_fn(query_vec)
		if not projected: 	
			source_vecs = activation_fn(source_vecs) 	

	distances = torch.max(query_vec - source_vecs, torch.zeros(query_vec.shape, device=device))
	distances = torch.sum(torch.square(distances), dim=-1)

	return distances

def order_embedding_projection(query_vec, source_vecs, device, activation_fn=None):
	## Order embedding projection
	if not activation_fn == None:
		query_vec = activation_fn(query_vec) 	
		source_vecs = activation_fn(source_vecs)

	distances = torch.max(query_vec - source_vecs, torch.zeros(query_vec.shape, device=device))
	projection = query_vec - distances

	return projection


def margin_selection(fitting_sorted_idx, embedding_distance, K, num_negs=5):
	#select the positive to be the closest by fitting loss
	positive_idx = fitting_sorted_idx[0,:]

	#random selection of negatives that are "far away"
	perm = torch.randperm(fitting_sorted_idx.size(0)-1) + 1

	negative_idx = fitting_sorted_idx[perm[:num_negs], :]

	#gather corresponding distances
	positive_distances = torch.gather(embedding_distance, 0, positive_idx.unsqueeze(0))
	positive_distances = positive_distances.unsqueeze(0).repeat(num_negs,1,1)
	negative_distances = torch.gather(embedding_distance, 0, negative_idx)

	return positive_distances, negative_distances

def margin_loss_multi(positive_distances, negative_distances, margin, device):
	num_negs = negative_distances.shape[0]
	positive_distances, _ = torch.min(positive_distances, dim=0)
	positive_distances = positive_distances.unsqueeze(0).repeat(num_negs,1)

	l = positive_distances - negative_distances + margin
	l = torch.max(l, torch.zeros(l.shape, device=device))
	return l

def margin_loss(positive_distances, negative_distances, margin, device):
	l = positive_distances - negative_distances + margin
	l = torch.max(l, torch.zeros(l.shape, device=device))
	return l

def regression_loss(embedding_distance, actual_distance, obj_sigmas):
	obj_sigmas = torch.sigmoid(obj_sigmas)
	# obj_sigmas = 1.0
	
	embedding_distance = embedding_distance/100.0
	qij = F.softmax(-embedding_distance, dim= -1)

	#tranform to reasonable ranges
	actual_distance = actual_distance*100.0

	pij = torch.div(actual_distance, obj_sigmas)
	pij = F.softmax(-actual_distance, dim= -1)

	# loss = torch.sum(torch.square(pij-qij), dim=-1)
	loss = torch.sum(torch.abs(pij-qij), dim= -1)

	# print(actual_distance)
	# print(pij)
	# print()
	# print(embedding_distance)
	# print(qij)

	return loss

def get_symmetric(pc):

	reflected_pc = torch.cat([-pc[:,:,0].unsqueeze(-1), pc[:,:,1].unsqueeze(-1), pc[:,:,2].unsqueeze(-1)], axis=2)

	return reflected_pc

### Property 2
def inclusion_margin_loss():
	# Property 1: embedding_distance ~ fitting_loss


	return