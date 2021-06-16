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
from pytorch3d.loss import chamfer_distance

from dataset import *
from model import * 
from losses import *

import math
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_full_chairs.json", help="path to the json config file", type=str)

parser.add_argument("--logdir", default="log_chair_keypoint_icpl2_srcenc_symm/", help="path to the log directory", type=str)
parser.add_argument('--dump_dir', default= "dump_test/", type=str)

parser.add_argument("--data_split", default="test", type=str)
parser.add_argument('--use_bn', default= False, type=bool)
parser.add_argument('--category', default= "chair", type=str)

parser.add_argument('--share_src_latent', default= False, type=bool)
parser.add_argument('--shared_encoder', default= False, type=bool)
parser.add_argument('--distance_function', default= "mahalanobis", type=str)
parser.add_argument('--activation_fn', default= "sigmoid", type=str)
parser.add_argument('--normalize', default= False, type=bool)
parser.add_argument('--joint_model', default= True, type=bool)
parser.add_argument('--use_connectivity', default= True, type=bool)
parser.add_argument('--num_sources', default= 500, type=int)

parser.add_argument('--use_src_encoder_retrieval', default= True, type=bool)
parser.add_argument('--use_singleaxis', default= False, type=bool)
parser.add_argument('--use_keypoint', default= True, type=bool)

parser.add_argument('--complementme', default= False, type=int)
parser.add_argument('--all_models', default= False, type=bool)

parser.add_argument('--mesh_visu', default= False, type=bool)


FLAGS = parser.parse_args()

config = FLAGS.config
LOG_DIR = FLAGS.logdir

fname = os.path.join(LOG_DIR, "config.json")

args = json.load(open(fname))

DATA_DIR = args["data_dir"]
OBJ_CAT = FLAGS.category

DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
temp_fol = os.path.join(DUMP_DIR, "tmp")
if not os.path.exists(temp_fol): os.mkdir(temp_fol)

##For mesh visu
mesh_fol = os.path.join(DUMP_DIR, "mesh")
if not os.path.exists(mesh_fol): os.mkdir(mesh_fol)
temp_fol = os.path.join(mesh_fol, "tmp")
if not os.path.exists(temp_fol): os.mkdir(temp_fol)

LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate_rankings.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

ALPHA = args["alpha"]
USE_BN = FLAGS.use_bn

SOURCE_LATENT_DIM = args["source_latent_dim"]
TARGET_LATENT_DIM = args["target_latent_dim"]
PART_LATENT_DIM = args["part_latent_dim"]

# FILENAME = args["filename"]

DATA_SPLIT = FLAGS.data_split

SHARE_SRC_LATENT = FLAGS.share_src_latent
SHARED_ENCODER = FLAGS.shared_encoder
DIST_FUNC = FLAGS.distance_function
NORMALIZE = FLAGS.normalize
ACTIVATION_FN = FLAGS.activation_fn
JOINT_MODEL = FLAGS.joint_model
USE_CONNECTIVITY = FLAGS.use_connectivity

NUM_SOURCES = FLAGS.num_sources
print("Num sources: "+str(NUM_SOURCES))

MESH_VISU = FLAGS.mesh_visu
ALL_MODELS = FLAGS.all_models

USE_SRC_ENCODER_RETRIEVAL = FLAGS.use_src_encoder_retrieval
USE_SINGLEAXIS = FLAGS.use_singleaxis
USE_KEYPOINT = FLAGS.use_keypoint
COMPLEMENTME = FLAGS.complementme

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

if __name__ == "__main__":

	start_time = time.time()

	if not ALL_MODELS:
		if COMPLEMENTME:
			print("Using ComplementMe dataset")
			src_data_fol = os.path.join(BASE_DIR, "data_complementme_final", OBJ_CAT, "h5_new")	

		elif USE_SINGLEAXIS:
			print("Using single axis constraint")
			src_data_fol = os.path.join(BASE_DIR, "data_aabb_constraints_singleaxis", OBJ_CAT, "h5")
		elif USE_KEYPOINT:
			print("Using keypoint constraint")
			src_data_fol = os.path.join(BASE_DIR, "data_aabb_constraints_keypoint", OBJ_CAT, "h5")
		else:
			src_data_fol = os.path.join(BASE_DIR, "data_aabb_constraints", OBJ_CAT, "h5")

		if COMPLEMENTME:
			filename_pickle = os.path.join("generated_datasplits_complementme", OBJ_CAT+"_"+str(NUM_SOURCES)+".pickle")
		else:
			filename_pickle = os.path.join("generated_datasplits", OBJ_CAT+"_"+str(NUM_SOURCES)+".pickle")
		sources, _, _ = get_all_selected_models_pickle(filename_pickle)


		if COMPLEMENTME:
			filename = os.path.join("generated_datasplits_complementme", OBJ_CAT+"_"+str(NUM_SOURCES)+"_"+DATA_SPLIT+".h5")
		else:
			#### Get data for all target models
			filename = os.path.join("generated_datasplits", OBJ_CAT+"_"+str(NUM_SOURCES)+"_"+DATA_SPLIT+".h5")
	else:
		src_data_fol = os.path.join(BASE_DIR, "data_aabb_constraints_keypoint")
		filename_pickle = os.path.join("generated_datasplits", "all_classes_chairtablecabinet_smaller.pickle")
		sources, sources_cat, _, _ = get_all_selected_models_pickle(filename_pickle, all_models=True)		
		filename = os.path.join("generated_datasplits", "all_classes_test_smaller.h5")		


	DATA_SPLIT = "test"	
	batch_size = 1

	dataset = StructureNetDataset_h5(filename)	

	loader = torch.utils.data.DataLoader(
	    dataset,
	    batch_size=batch_size,
	    num_workers=args["num_workers"],
	    pin_memory=True,
	    shuffle=False,
	)

	#### Torch
	device = args["device"]

	print("Loading sources...")

	##Get the data of the sources
	## Get max number of params for the embedding size
	MAX_NUM_PARAMS = -1
	MAX_NUM_PARTS = -1
	SOURCE_MODEL_INFO = []

	for i in range(len(sources)):
		source_model = sources[i]
		src_filename = str(source_model) + "_leaves.h5"

		if not ALL_MODELS:
			if (USE_CONNECTIVITY):
				box_params, orig_ids, default_param, points, point_labels, points_mat, \
									vertices, vertices_mat, faces, face_labels, \
									constraint_mat,	constraint_proj_mat	= get_model(os.path.join(src_data_fol, src_filename), mesh=True, constraint=True)
			else:
				box_params, orig_ids, default_param, points, point_labels, points_mat, \
							 vertices, vertices_mat, faces, face_labels, = get_model(os.path.join(src_data_fol, src_filename), mesh=True)

		else:
			if (USE_CONNECTIVITY):
				box_params, orig_ids, default_param, points, point_labels, points_mat, \
									vertices, vertices_mat, faces, face_labels, \
									constraint_mat,	constraint_proj_mat	= get_model(os.path.join(src_data_fol, sources_cat[i], "h5", src_filename), mesh=True, constraint=True)
			else:
				box_params, orig_ids, default_param, points, point_labels, points_mat, \
							 vertices, vertices_mat, faces, face_labels, = get_model(os.path.join(src_data_fol, sources_cat[i], "h5", src_filename), mesh=True)


		curr_source_dict = {}
		curr_source_dict["default_param"] = default_param
		curr_source_dict["points"] = points
		curr_source_dict["point_labels"] = point_labels
		curr_source_dict["points_mat"] = points_mat
		curr_source_dict["vertices"] = vertices
		curr_source_dict["vertices_mat"] = vertices_mat
		curr_source_dict["faces"] = faces
		curr_source_dict["face_labels"] = face_labels
		curr_source_dict["model_id"] = source_model

		if (USE_CONNECTIVITY):
			curr_source_dict["constraint_mat"] = constraint_mat
			curr_source_dict["constraint_proj_mat"] = constraint_proj_mat

		# Get number of parts of the model
		num_parts = len(np.unique(point_labels))
		curr_source_dict["num_parts"] = num_parts

		curr_num_params = default_param.shape[0]
		if (MAX_NUM_PARAMS < curr_num_params):
			MAX_NUM_PARAMS = curr_num_params
			MAX_NUM_PARTS = int(MAX_NUM_PARAMS/6)

		SOURCE_MODEL_INFO.append(curr_source_dict)

	print("Done loading sources.")
	print(len(SOURCE_MODEL_INFO))
	# exit()

	print(MAX_NUM_PARAMS)
	print(MAX_NUM_PARTS)
	embedding_size = 6

	#### Load model
	target_encoder = TargetEncoder(
	    TARGET_LATENT_DIM,
	    args["input_channels"],
	)
	target_encoder.to(device, dtype=torch.float)

	decoder_input_dim = TARGET_LATENT_DIM + SOURCE_LATENT_DIM + PART_LATENT_DIM
	param_decoder = ParamDecoder2(decoder_input_dim, 256, embedding_size)
	param_decoder.to(device, dtype=torch.float)

	## For Retrieval
	retrieval_encoder = TargetEncoder(
	    TARGET_LATENT_DIM,
	    args["input_channels"],
	)
	retrieval_encoder.to(device, dtype=torch.float)	

	np.random.seed(0)
	fname = os.path.join(LOG_DIR, "model.pth")
	# fname = os.path.join(LOG_DIR, "checkpoint_0269.pth")
	target_encoder.load_state_dict(torch.load(fname)["target_encoder"])
	target_encoder.to(device)
	target_encoder.eval()

	param_decoder.load_state_dict(torch.load(fname)["param_decoder"])
	param_decoder.to(device)
	param_decoder.eval()

	SOURCE_LATENT_CODES = torch.load(fname)["source_latent_codes"]
	SOURCE_PART_LATENT_CODES = torch.load(fname)["part_latent_codes"]


	if JOINT_MODEL:
		if not SHARED_ENCODER:
			retrieval_encoder.load_state_dict(torch.load(fname)["retrieval_encoder"])
			retrieval_encoder.to(device)
			retrieval_encoder.eval()

		if (DIST_FUNC == "mahalanobis"):
			SOURCE_VARIANCES = torch.load(fname)["source_variances"]

		if not SHARE_SRC_LATENT:
			RETRIEVAL_SOURCE_LATENT_CODES = torch.load(fname)["retrieval_source_latent_codes"]
	########

	num_evaluated = 0
	total_cd_error = 0
	used_sources = []
	num_correct_retrieved = 0

	candidates = []

	ranking_x_axis = np.arange(len(SOURCE_MODEL_INFO))
	per_rank_total_cd_error_retrieved = np.zeros(len(SOURCE_MODEL_INFO))
	per_rank_total_cd_error = np.zeros(len(SOURCE_MODEL_INFO))

	cd_threshold_x_axis = np.arange(0.0085, 0.07, 0.0005)
	num_sources_above_threshold = np.zeros(cd_threshold_x_axis.shape[0])
	# print(cd_threshold_x_axis)
	# print(cd_threshold_x_axis.shape)
	# print(num_sources_above_threshold.shape)
	# exit()

	for i, batch in enumerate(loader):
		'''
		Per batch output:
			self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index], \
			self.corres_source_label[index]
		'''			
		# target_shapes, target_ids, target_labels, _, source_labels_gt = batch
		# print("a")
		target_shapes, target_ids, target_labels, _ = batch


		if COMPLEMENTME:
			target_shapes[:,:,2] = -target_shapes[:,:,2]

		source_label_shape = torch.zeros(target_shapes.shape[0])

		x = [x.to(device, dtype=torch.float) for x in target_shapes]
		x = torch.stack(x)

		##Target Encoder
		target_latent_codes = target_encoder(x)

		# print(target_latent_codes.shape)

		target_latent_codes = target_latent_codes.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1)
		source_labels = source_label_shape.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1)

		## Reshape to (K*batch_size, ...) to feed into the network
		## Source assignments have to be done accordingly
		target_latent_codes = target_latent_codes.view(-1, target_latent_codes.shape[-1])
		source_labels = source_labels.view(-1)

		#Get all labels
		source_labels = get_all_source_labels(source_labels, len(SOURCE_MODEL_INFO))


		##Also overwrite x for chamfer distance					
		x_repeated = x.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1,1)
		x_repeated = x_repeated.view(-1, x_repeated.shape[-2], x_repeated.shape[-1])

		# print("a.1")
		###Set up source A matrices and default params based on source_labels of the target
		# src_mats, src_default_params = get_source_info(source_labels, SOURCE_MODEL_INFO, MAX_NUM_PARAMS)
		src_mats, src_default_params, src_connectivity_mat = get_source_info(source_labels, SOURCE_MODEL_INFO, MAX_NUM_PARAMS, use_connectivity= USE_CONNECTIVITY)
		# print("a.2")

		###Set up source latent codes based on source_labels of the target
		src_latent_codes = get_source_latent_codes_fixed(source_labels, SOURCE_LATENT_CODES, device)

		mat = [mat.to(device, dtype=torch.float) for mat in src_mats]
		def_param = [def_param.to(device, dtype=torch.float) for def_param in src_default_params]

		mat = torch.stack(mat)
		def_param = torch.stack(def_param)

		## If using connectivity
		if (USE_CONNECTIVITY):
			conn_mat = [conn_mat.to(device, dtype=torch.float) for conn_mat in src_connectivity_mat]
			conn_mat = torch.stack(conn_mat)

		concat_latent_code = torch.cat((src_latent_codes, target_latent_codes), dim=1)

		# print(concat_latent_code.shape)

		##Param Decoder per part
		# Make the part latent codes of each source into a (K x PART_LATENT_DIM) tensor
		all_params = []
		for j in range(concat_latent_code.shape[0]):
			curr_num_parts = SOURCE_MODEL_INFO[source_labels[j]]["num_parts"]
			curr_code = concat_latent_code[j]
			curr_code_repeated = curr_code.view(1,curr_code.shape[0]).repeat(curr_num_parts, 1)
			
			part_latent_codes = SOURCE_PART_LATENT_CODES[source_labels[j]]

			full_latent_code = torch.cat((curr_code_repeated, part_latent_codes), dim=1)

			params = param_decoder(full_latent_code, use_bn=USE_BN)

			## Pad with extra zero rows to cater to max number of parameters
			if (curr_num_parts < MAX_NUM_PARTS):
				dummy_params = torch.zeros((MAX_NUM_PARTS-curr_num_parts, embedding_size), dtype=torch.float, device=device)
				params = torch.cat((params, dummy_params), dim=0)

			params = params.view(-1, 1)
			all_params.append(params)

		params = torch.stack(all_params)

		if (USE_CONNECTIVITY):
			output_pcs = get_shape(mat, params, def_param, ALPHA, connectivity_mat=conn_mat)
		else:
			output_pcs = get_shape(mat, params, def_param, ALPHA)

		cd_loss, _ = chamfer_distance(output_pcs, x_repeated, batch_reduction=None)
			
		output_pcs = output_pcs.view(len(SOURCE_MODEL_INFO), target_shapes.shape[0], target_shapes.shape[1], target_shapes.shape[2])
		cd_loss = cd_loss.view(len(SOURCE_MODEL_INFO), -1)

		# print("b")

		if JOINT_MODEL:
			## Retrieval
			if not SHARED_ENCODER:
				retrieval_latent_codes = retrieval_encoder(x)
				retrieval_latent_codes = retrieval_latent_codes.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1)
				retrieval_latent_codes = retrieval_latent_codes.view(-1, retrieval_latent_codes.shape[-1])

			else:
				retrieval_latent_codes = target_latent_codes

			if (NORMALIZE):
				retrieval_latent_codes = F.normalize(retrieval_latent_codes)	
						
			retrieval_latent_codes = retrieval_latent_codes.view(len(SOURCE_MODEL_INFO), -1, TARGET_LATENT_DIM)
			
			if USE_SRC_ENCODER_RETRIEVAL:
				with torch.no_grad():
					src_latent_codes = []
					num_sets = 10
					interval = int(len(source_labels)/num_sets)

					for j in range(num_sets):
						if (j==num_sets-1):
							curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:], SOURCE_MODEL_INFO, retrieval_encoder, device=device)
						else:
							curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:(j+1)*interval], SOURCE_MODEL_INFO, retrieval_encoder, device=device)
						src_latent_codes.append(curr_src_latent_codes)

					src_latent_codes = torch.cat(src_latent_codes).view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

			else:
				src_latent_codes = get_source_latent_codes_fixed(source_labels, RETRIEVAL_SOURCE_LATENT_CODES, device=device)
				src_latent_codes = src_latent_codes.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)
			
			if (DIST_FUNC == "mahalanobis"):
				src_variances = get_source_latent_codes_fixed(source_labels, SOURCE_VARIANCES, device)
				src_variances = src_variances.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)


			if (DIST_FUNC == "mahalanobis"):
				src_variances = get_source_latent_codes_fixed(source_labels, SOURCE_VARIANCES, device)
				src_variances = src_variances.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

				if (ACTIVATION_FN.lower() == "none"):
					distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances)
				elif (ACTIVATION_FN == "sigmoid"):
					distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, activation_fn=torch.sigmoid)
				elif (ACTIVATION_FN == "relu"):
					distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, activation_fn=torch.relu)		

			elif (DIST_FUNC == "order"):
				if (ACTIVATION_FN.lower() == "none"):
					distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device)
				elif (ACTIVATION_FN == "sigmoid"):
					distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device, activation_fn=torch.sigmoid)
				elif (ACTIVATION_FN == "relu"):
					distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device, activation_fn=torch.relu)	

			# distances = distances.to("cpu").detach().numpy()[:,0]
			# distances = distances/100.0
			# distances_normalized = np.exp(-(distances - np.max(distances)))
			# probs = distances / distances.sum()							

			# plt.clf()	
			# plt.figure()			
			# print(probs.shape)
			# plt.plot(ranking_x_axis, probs, 'bo-', linewidth=0.5, markersize=2)
			# plt.xlabel('Source Model Index')
			# plt.ylabel('Soft Retrieval Probability')

			# plt.savefig(os.path.join("0_paper_fig", 'sampling_distribution.png'))
			# exit()

			sorted_indices = torch.argsort(distances, dim=0)
			retrieved_idx = sorted_indices[0,:] 

		# sorted_by_cd = torch.argsort(cd_loss, dim=0)
		cd_loss_sorted, sorted_by_cd = torch.sort(cd_loss, dim=0)
		cd_retrieved = sorted_by_cd[0,:]

		if JOINT_MODEL:
			cd_loss_retrieved = torch.gather(cd_loss, 0, sorted_indices)

		else:
			# print("Random retrieval")
			# Random permutation for deformation function only
			# sorted_indices = []
			# for _ in range(sorted_by_cd.shape[1]):
			# 	sorted_indices.append(torch.randperm(len(SOURCE_MODEL_INFO)).to(device))
			# sorted_indices = torch.stack(sorted_indices).T

			#ground truth sorted
			sorted_indices = sorted_by_cd
			retrieved_idx = sorted_indices[0, :]

			cd_loss_retrieved = torch.gather(cd_loss, 0, sorted_indices)

			# print(cd_loss_sorted.shape)
			# print(sorted_indices.shape)
			# print(retrieved_idx.shape)
			# print(cd_loss_retrieved.shape)
			# exit()

		# print("c")

		'''
		For Mesh Rendering
		'''
		params = params.view(len(SOURCE_MODEL_INFO), target_shapes.shape[0], -1)
		# print(params.shape)
		# print(retrieved_idx.shape)
		# exit()
		retrieved_idx_repeated = retrieved_idx.unsqueeze(0).unsqueeze(-1).repeat(1,1,params.shape[-1])
		params_retrieved = torch.gather(params, 0, retrieved_idx_repeated)
		params_retrieved = params_retrieved.to("cpu")
		params_retrieved = params_retrieved.detach().numpy()[0]
		# print(params_retrieved.shape)

		####For mesh visualization
		src_vertices_mats, src_default_params, src_conn_mat = get_source_info_mesh(retrieved_idx, SOURCE_MODEL_INFO, MAX_NUM_PARAMS, use_connectivity=USE_CONNECTIVITY)
		# print(len(src_vertices_mats))
		# print(src_vertices_mats[0])
		# print(src_vertices_mats[0].shape)
		# print(src_vertices_mats[1].shape)
		# print(src_vertices_mats[2].shape)
		# exit()
		########################

		cd_loss = cd_loss.to("cpu")
		cd_loss = cd_loss.detach().numpy()
		sorted_by_cd = sorted_by_cd.to("cpu")
		sorted_by_cd = sorted_by_cd.detach().numpy()

		cd_loss_sorted = cd_loss_sorted.to("cpu")
		cd_loss_sorted = cd_loss_sorted.detach().numpy().T

		cd_retrieved = cd_retrieved.to("cpu")
		cd_retrieved = cd_retrieved.detach().numpy()
		
		retrieved_idx = retrieved_idx.to("cpu")
		retrieved_idx = retrieved_idx.detach().numpy()
		
		sorted_indices = sorted_indices.to("cpu")
		sorted_indices = sorted_indices.detach().numpy()	
		cd_loss_retrieved = cd_loss_retrieved.to("cpu").detach().numpy().T		
		
		target_shapes = target_shapes.to("cpu")
		target_shapes = target_shapes.detach().numpy()
		target_labels = target_labels.to("cpu")
		target_labels = target_labels.detach().numpy()
		target_ids = target_ids.to("cpu")
		target_ids = target_ids.detach().numpy()

		'''
		Get source points, ids and labels
		'''
		src_points, src_labels, src_ids, _, src_vertices, src_faces, src_face_labels = get_source_info_visualization(retrieved_idx, SOURCE_MODEL_INFO, mesh=True)
		correct_retrieved = np.equal(retrieved_idx, cd_retrieved)

		for j in range(cd_retrieved.shape[0]):
			num_evaluated += 1

			total_cd_error += cd_loss_retrieved[j][0]

			if not src_ids[j] in used_sources:
				used_sources.append(src_ids[j])

			num_correct_retrieved += correct_retrieved[j]

			curr_candidates = sorted_indices[:,j]
			candidates.append(curr_candidates)

			## For graph 1
			per_rank_total_cd_error += cd_loss_sorted[j]
			per_rank_total_cd_error_retrieved += cd_loss_retrieved[j]

			##For graph 2 
			cd_threshold_x_axis_tiled = np.tile(np.expand_dims(cd_threshold_x_axis, -1), (1, cd_loss_sorted[j].shape[0]))

			count_greater = np.greater(cd_threshold_x_axis_tiled, cd_loss_sorted[j])
			count_greater = np.sum(count_greater, axis=-1)
			num_sources_above_threshold += count_greater

			if (j<2 and MESH_VISU):
				## For mesh rendering
				curr_param = np.expand_dims(params_retrieved[j], -1)
				curr_mat = src_vertices_mats[j].detach().numpy()
				
				if (USE_CONNECTIVITY):
					curr_conn_mat = src_conn_mat[j].detach().numpy()
				else:
					curr_conn_mat = None


				curr_default_param = src_default_params[j].detach().numpy().T

				output_vertices = get_shape_numpy(curr_mat, curr_param, curr_default_param, ALPHA, connectivity_mat=curr_conn_mat)

				if (COMPLEMENTME):
					output_vertices[:, 2] = -output_vertices[:, 2]
					curr_src_vertices = src_vertices[j].copy()
					curr_src_vertices[:, 2] = -src_vertices[j][:, 2]
					target_shapes[j][:, 2] = -target_shapes[j][:, 2]
				else:
					curr_src_vertices = src_vertices[j]

				output_visualization_mesh(output_vertices, curr_src_vertices, src_faces[j], target_shapes[j], src_face_labels[j], target_labels[j], src_ids[j], target_ids[j], mesh_fol)

		if (i%20==0):
			print("Time elapsed: "+str(time.time()-start_time)+" sec for batch "+str(i)+ "/"+ str(len(loader))+".")

	### Save numerical results
	mean_cd_loss = total_cd_error/float(num_evaluated)
	log_string("Num evaluated= "+str(num_evaluated))
	log_string("")
	log_string("Number of unique selected sources: "+str(len(used_sources))+"/"+str(len(SOURCE_MODEL_INFO)))
	log_string("")
	log_string("Mean CD error= "+str(mean_cd_loss))

	if (JOINT_MODEL):
		accuracy = float(num_correct_retrieved)/num_evaluated
		log_string("Retrieval accuracy= "+str(accuracy))

	##Graph 1 rank vs mean cd_error
	print(per_rank_total_cd_error.shape)
	per_rank_total_cd_error = per_rank_total_cd_error/num_evaluated
	per_rank_total_cd_error_retrieved = per_rank_total_cd_error_retrieved/num_evaluated

	fname = DUMP_DIR[5:-1]
	plt.clf()	
	plt.figure()
	plt.scatter(ranking_x_axis, per_rank_total_cd_error, label='curr_deformation_function_ranking', c='b', s=1)
	plt.scatter(ranking_x_axis, per_rank_total_cd_error_retrieved, label='embedding_ranking', c='r', s=1)
	plt.legend()
	plt.xlabel('Retrieved Rank')
	plt.ylabel('Mean CD error')
	plt.title(fname +" CD error")
	plt.savefig(os.path.join(DUMP_DIR, fname+'_rankvscd.png'))
	print(per_rank_total_cd_error)
	print()
	print(per_rank_total_cd_error_retrieved)

	filename = os.path.join(DUMP_DIR, fname+'_rankvscd.pickle')

	log_string("Filename: "+filename)

	dict_value = {"curr_deformation_function_ranking": per_rank_total_cd_error,
				"embedding_ranking": per_rank_total_cd_error_retrieved,
				"ranking_x_axis": ranking_x_axis}

	with open(filename, 'wb') as handle:
	    pickle.dump(dict_value, handle, protocol=pickle.HIGHEST_PROTOCOL)

	log_string(" ")
	log_string("Ranking Retrieved Mean CD error:")
	### Output to text file
	for i in range(5):
		log_string("\tRank "+str(i+1)+": "+str(per_rank_total_cd_error_retrieved[i]))

	log_string(" ")

	log_string("Ranking Oracle Mean CD error:")
	### Output to text file
	for i in range(5):
		log_string("\tRank "+str(i+1)+": "+str(per_rank_total_cd_error[i]))

	log_string(" ")

	##Graph 2 cd_error threshold vs number of sources
	print(num_sources_above_threshold.shape)
	num_sources_above_threshold = num_sources_above_threshold/num_evaluated

	plt.clf()	
	plt.figure()
	plt.scatter(cd_threshold_x_axis, num_sources_above_threshold, c='b', s=1)
	plt.xlabel('CD Threshold')
	plt.ylabel('Num sources within the threshold')
	plt.title(fname +" Count")
	plt.savefig(os.path.join(DUMP_DIR, fname+'_cdthreshvsnum.png'))
	print(cd_threshold_x_axis)
	print(num_sources_above_threshold)

	filename = os.path.join(DUMP_DIR, fname+'_cdthreshvsnum.pickle')

	log_string("Filename: "+filename)

	dict_value = {"thresholds": cd_threshold_x_axis,
				"num_sources": num_sources_above_threshold}

	with open(filename, 'wb') as handle:
	    pickle.dump(dict_value, handle, protocol=pickle.HIGHEST_PROTOCOL)

	LOG_FOUT.close()	


