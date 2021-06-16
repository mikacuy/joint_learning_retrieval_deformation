import os, sys
BASE_DIR = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import json
import datetime
import time
from collections import defaultdict

from data_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

from dataset import *
from model import * 
from losses import *

import gc

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_chairs_final.json", help="path to the json config file", type=str)
parser.add_argument("--logdir", default="log_test2", help="path to the log directory", type=str)
parser.add_argument('--dump_dir', default= "dump_test2", type=str)
parser.add_argument('--category', default= "storagefurniture", type=str)

parser.add_argument('--to_train', default= False, type=bool)
parser.add_argument('--part_loss', default= False, type=bool)
parser.add_argument('--use_bn', default= False, type=bool)

parser.add_argument('--update_src_latent', default= False, type=bool)
parser.add_argument('--share_src_latent', default= False, type=bool)
parser.add_argument('--normalize', default= False, type=bool)
parser.add_argument('--init_deformation', default= False, type=bool)
parser.add_argument('--fixed_deformation', default= False, type=bool)
parser.add_argument('--shared_encoder', default= False, type=bool)
parser.add_argument('--clip_vec', default= False, type=bool)

parser.add_argument('--distance_function', default= "mahalanobis", type=str)
parser.add_argument('--loss_function', default= "regression", type=str)
parser.add_argument('--selection', default= "random", type=str)

parser.add_argument('--use_connectivity', default= False, type=bool)
parser.add_argument('--use_src_encoder', default= False, type=bool)
parser.add_argument('--use_src_encoder_retrieval', default= False, type=bool)
parser.add_argument('--use_symmetry', default= False, type=bool)
parser.add_argument('--use_singleaxis', default= False, type=bool)
parser.add_argument('--use_keypoint', default= False, type=bool)

parser.add_argument('--prop2', default= False, type=bool)
parser.add_argument('--prop2_more', default= False, type=bool)
parser.add_argument('--prop3', default= False, type=bool)

# parser.add_argument('--model_init', default= "log_chair500_conn/", type=str)
parser.add_argument('--model_init', default= "log_chair500_conn/", type=str)
parser.add_argument('--K', default= 10, type=int)
parser.add_argument('--margin', default= 10.0, type=float)
parser.add_argument('--activation_fn', default= "sigmoid", type=str)
parser.add_argument('--visualize', default= False, type=bool)
parser.add_argument('--eval_selection', default= "retrieval", type=str)

parser.add_argument('--num_sources', default= 500, type=int)
parser.add_argument('--complementme', default= False, type=int)
parser.add_argument('--all_models', default= False, type=bool)

parser.add_argument('--fine_tune', default= False, type=bool)
parser.add_argument("--ref_logdir", default="log_all_models_A_J3/", help="path to the log directory", type=str)

FLAGS = parser.parse_args()

config = FLAGS.config
LOG_DIR = FLAGS.logdir

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

TO_TRAIN = FLAGS.to_train
fname = os.path.join(LOG_DIR, "config.json")

if TO_TRAIN:
	args = json.load(open(config))
	with open(fname, "w") as fp:
	    json.dump(args, fp, indent=4)
else:
	args = json.load(open(fname))

curr_fname = sys.argv[0]
if TO_TRAIN:
	os.system('cp %s %s' % (curr_fname, LOG_DIR))

DATA_DIR = args["data_dir"]
# OBJ_CAT = args["category"]
OBJ_CAT = FLAGS.category

DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
temp_fol = os.path.join(DUMP_DIR, "tmp")
if not os.path.exists(temp_fol): os.mkdir(temp_fol)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

ALPHA = args["alpha"]

SOURCE_LATENT_DIM = args["source_latent_dim"]
TARGET_LATENT_DIM = args["target_latent_dim"]
PART_LATENT_DIM = args["part_latent_dim"]

PART_LOSS = FLAGS.part_loss
K = FLAGS.K
MARGIN = FLAGS.margin

USE_BN = FLAGS.use_bn
ACTIVATION_FN = FLAGS.activation_fn
UPDATE_SRC_LATENT = FLAGS.update_src_latent
SHARE_SRC_LATENT = FLAGS.share_src_latent
NORMALIZE = FLAGS.normalize
INIT_DEFORMATION = FLAGS.init_deformation
FIXED_DEFORMATION = FLAGS.fixed_deformation
SHARED_ENCODER = FLAGS.shared_encoder
MODEL_INIT = FLAGS.model_init
CLIP_VEC = FLAGS.clip_vec
EVAL_SELECTION = FLAGS.eval_selection
LOSS_FUNC = FLAGS.loss_function
DIST_FUNC = FLAGS.distance_function
SELECTION = FLAGS.selection

USE_CONNECTIVITY = FLAGS.use_connectivity
USE_SRC_ENCODER = FLAGS.use_src_encoder
USE_SRC_ENCODER_RETRIEVAL = FLAGS.use_src_encoder_retrieval
USE_SYMMETRY = FLAGS.use_symmetry
USE_SINGLEAXIS = FLAGS.use_singleaxis
USE_KEYPOINT = FLAGS.use_keypoint

PROP2 = FLAGS.prop2
PROP2_MORE = FLAGS.prop2_more
PROP3 = FLAGS.prop3

NUM_SOURCES = FLAGS.num_sources
print("Num sources: "+str(NUM_SOURCES))
COMPLEMENTME = FLAGS.complementme
ALL_MODELS = FLAGS.all_models

FINE_TUNE = FLAGS.fine_tune
REF_LOGDIR = FLAGS.ref_logdir

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

log_string("Normalize: " + str(NORMALIZE))
log_string("Use connectivity: " + str(USE_CONNECTIVITY))
log_string("Update source vec: " + str(UPDATE_SRC_LATENT))
log_string("Shared src_latent: " + str(SHARE_SRC_LATENT))
log_string("")

log_string("Fixed deformation: " + str(FIXED_DEFORMATION))
log_string("Init Deformation: " + str(INIT_DEFORMATION)+" model: "+MODEL_INIT)
log_string("Shared encoder: " + str(SHARED_ENCODER))
log_string("")

log_string("Loss Func: " + str(LOSS_FUNC))
log_string("Distance Func: " + str(DIST_FUNC))
log_string("Activation Func: " + str(ACTIVATION_FN))
log_string("Selection: " + str(SELECTION))
log_string("Use Property 2: " + str(PROP2))
log_string("Use Property 2 more: " + str(PROP2_MORE))
log_string("Use Property 3: " + str(PROP3))
log_string("")

LOG_FOUT.write(str(FLAGS)+'\n')

TO_VISU = FLAGS.visualize

global DEFORMATION_CANDIDATES
DEFORMATION_CANDIDATES = {}

global DEFORMATION_DISTANCES
DEFORMATION_DISTANCES = {}

if __name__ == "__main__":

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
		filename = os.path.join("generated_datasplits", "all_classes_train_smaller.h5")


	if (TO_TRAIN):
		DATA_SPLIT = "train"
		batch_size = args["batch_size"]			
	else:
		DATA_SPLIT = "test"	
		batch_size = 2

	dataset = StructureNetDataset_h5(filename)	

	to_shuffle = TO_TRAIN
	# print(to_shuffle)

	loader = torch.utils.data.DataLoader(
	    dataset,
	    batch_size=batch_size,
	    num_workers=args["num_workers"],
	    pin_memory=True,
	    shuffle=to_shuffle,
	)

	#### Torch
	device = args["device"]

	##Get the data of the sources

	## Get max number of params for the embedding size
	MAX_NUM_PARAMS = -1
	MAX_NUM_PARTS = -1
	SOURCE_MODEL_INFO = []
	SOURCE_SEMANTICS = []
	SOURCE_PART_LATENT_CODES = []

	print("Loading sources...")

	for i in range(len(sources)):
		source_model = sources[i]
		src_filename = str(source_model) + "_leaves.h5"

		if not ALL_MODELS:
			if (USE_CONNECTIVITY):
				box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic, \
									constraint_mat,	constraint_proj_mat	= get_model(os.path.join(src_data_fol, src_filename), semantic=True, constraint=True)
			else:
				box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic = get_model(os.path.join(src_data_fol, src_filename), semantic=True)

		else:
			if (USE_CONNECTIVITY):
				box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic, \
									constraint_mat,	constraint_proj_mat	= get_model(os.path.join(src_data_fol, sources_cat[i], "h5", src_filename), semantic=True, constraint=True)
			else:
				box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic = get_model(os.path.join(src_data_fol, sources_cat[i], "h5", src_filename), semantic=True)

		curr_source_dict = {}
		curr_source_dict["default_param"] = default_param
		curr_source_dict["points"] = points
		curr_source_dict["point_labels"] = point_labels
		curr_source_dict["points_mat"] = points_mat
		curr_source_dict["point_semantic"] = point_semantic
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

		# For source semantics also get a list of unique labels
		src_semantic = torch.from_numpy(point_semantic)
		src_semantic = src_semantic.to(device)
		unique_labels = torch.unique(src_semantic)
		SOURCE_SEMANTICS.append([src_semantic, unique_labels])

		# Create latent code per part for the autodecoder setup
		part_latent_codes = torch.autograd.Variable(torch.randn((num_parts,PART_LATENT_DIM), dtype=torch.float, device=device), requires_grad=True)
		SOURCE_PART_LATENT_CODES.append(part_latent_codes)

	## Create source latent
	SOURCE_LATENT_CODES = torch.autograd.Variable(torch.randn((len(sources),SOURCE_LATENT_DIM), dtype=torch.float, device=device), requires_grad=True)

	if not SHARE_SRC_LATENT:
		RETRIEVAL_SOURCE_LATENT_CODES = torch.autograd.Variable(torch.randn((len(sources),SOURCE_LATENT_DIM), dtype=torch.float, device=device), requires_grad=True)


	if (DIST_FUNC == "mahalanobis"):
		SOURCE_VARIANCES = torch.autograd.Variable(torch.randn((len(sources),SOURCE_LATENT_DIM), dtype=torch.float, device=device), requires_grad=True)


	if (LOSS_FUNC == "regression"):
		SOURCE_SIGMAS = torch.autograd.Variable(torch.randn((len(sources),1), dtype=torch.float, device=device), requires_grad=True)

	print("Done loading sources.")
	print(len(SOURCE_MODEL_INFO))
	embedding_size = 6

	## Define Networks

	## For Deformation
	target_encoder = TargetEncoder(
	    TARGET_LATENT_DIM,
	    args["input_channels"],
	)
	target_encoder.to(device, dtype=torch.float)

	## For Retrieval
	retrieval_encoder = TargetEncoder(
	    TARGET_LATENT_DIM,
	    args["input_channels"],
	)


	retrieval_encoder.to(device, dtype=torch.float)	

	decoder_input_dim = TARGET_LATENT_DIM + SOURCE_LATENT_DIM + PART_LATENT_DIM
	param_decoder = ParamDecoder2(decoder_input_dim, 256, embedding_size, use_bn=USE_BN)
	param_decoder.to(device, dtype=torch.float)

	## Define loss and optimizer
	learning_rate = args["learning_rate"]
	n_epochs = args["epochs"]

	## To fine tune the retrieval module
	if FINE_TUNE:
		fname = os.path.join(REF_LOGDIR, "model.pth")
		target_encoder.load_state_dict(torch.load(fname)["target_encoder"])
		target_encoder.to(device)
		target_encoder.eval()

		param_decoder.load_state_dict(torch.load(fname)["param_decoder"])
		param_decoder.to(device)
		param_decoder.eval()

		SOURCE_LATENT_CODES = torch.load(fname)["source_latent_codes"]
		SOURCE_PART_LATENT_CODES = torch.load(fname)["part_latent_codes"]

		for child in target_encoder.children():
			if type(child)==nn.BatchNorm1d:
			    child.track_running_stats = False
			elif type(child)==nn.Sequential:
				for ii in range(len(child)):
				    if type(child[ii])==nn.BatchNorm1d:
				        child[ii].track_running_stats = False				

		for child in param_decoder.children():
			if type(child)==nn.BatchNorm1d:
			    child.track_running_stats = False

		if not SHARED_ENCODER:
			retrieval_encoder.load_state_dict(torch.load(fname)["retrieval_encoder"])
			retrieval_encoder.to(device)
			retrieval_encoder.eval()

		if (DIST_FUNC == "mahalanobis"):
			SOURCE_VARIANCES = torch.load(fname)["source_variances"]

		if not SHARE_SRC_LATENT:
			RETRIEVAL_SOURCE_LATENT_CODES = torch.load(fname)["retrieval_source_latent_codes"]

		FIXED_DEFORMATION = True
		########		

	elif FIXED_DEFORMATION:
		###Load a model and keep it fixed
		fname = os.path.join(MODEL_INIT, "model.pth")
		target_encoder.load_state_dict(torch.load(fname)["target_encoder"])
		target_encoder.to(device)
		target_encoder.eval()

		param_decoder.load_state_dict(torch.load(fname)["param_decoder"])
		param_decoder.to(device)
		param_decoder.eval()

		SOURCE_LATENT_CODES = torch.load(fname)["source_latent_codes"]
		SOURCE_PART_LATENT_CODES = torch.load(fname)["part_latent_codes"]

		for child in target_encoder.children():
			if type(child)==nn.BatchNorm1d:
			    child.track_running_stats = False
			elif type(child)==nn.Sequential:
				for ii in range(len(child)):
				    if type(child[ii])==nn.BatchNorm1d:
				        child[ii].track_running_stats = False				

		for child in param_decoder.children():
			if type(child)==nn.BatchNorm1d:
			    child.track_running_stats = False

	else:
		if (INIT_DEFORMATION):
			#Load model
			fname = os.path.join(MODEL_INIT, "model.pth")
			target_encoder.load_state_dict(torch.load(fname)["target_encoder"])
			target_encoder.to(device)

			param_decoder.load_state_dict(torch.load(fname)["param_decoder"])
			param_decoder.to(device)

			SOURCE_LATENT_CODES = torch.load(fname)["source_latent_codes"]
			SOURCE_PART_LATENT_CODES = torch.load(fname)["part_latent_codes"]

		target_encoder_params = list(target_encoder.parameters())
		decoder_params = list(param_decoder.parameters())


		all_params = target_encoder_params + decoder_params

		## For the deformation
		optimizer_deformation = torch.optim.SGD(
		    all_params,
		    lr=args["learning_rate"],
		    momentum=args["momentum"],
		    weight_decay=args["weight_decay"],
		)

		optimizer_deformation.add_param_group({"params": SOURCE_LATENT_CODES})
		optimizer_deformation.add_param_group({"params": SOURCE_PART_LATENT_CODES})

	### Optimizer for retrieval 
	if not SHARED_ENCODER :
		params_embedding = list(retrieval_encoder.parameters())

		optimizer_embedding = torch.optim.SGD(
		    params_embedding,
		    lr=args["learning_rate"],
		    momentum=args["momentum"],
		    weight_decay=args["weight_decay"],
		)

		if not SHARE_SRC_LATENT:
			optimizer_embedding.add_param_group({"params": RETRIEVAL_SOURCE_LATENT_CODES})

		elif SHARE_SRC_LATENT and UPDATE_SRC_LATENT:
			optimizer_embedding.add_param_group({"params": SOURCE_LATENT_CODES})

		if (DIST_FUNC == "mahalanobis"):
			optimizer_embedding.add_param_group({"params": SOURCE_VARIANCES})

		if (LOSS_FUNC == "regression"):
			optimizer_embedding.add_param_group({"params": SOURCE_SIGMAS, "lr": 0.1})

	else:
		if not SHARE_SRC_LATENT:
			optimizer_embedding = torch.optim.SGD(
			    [RETRIEVAL_SOURCE_LATENT_CODES],
			    lr=args["learning_rate"],
			    momentum=args["momentum"],
			    weight_decay=args["weight_decay"],
			)

			if (DIST_FUNC == "mahalanobis"):
				optimizer_embedding.add_param_group({"params": SOURCE_VARIANCES})

			if (LOSS_FUNC == "regression"):
				optimizer_embedding.add_param_group({"params": SOURCE_SIGMAS, "lr": 0.1})


		elif SHARE_SRC_LATENT and UPDATE_SRC_LATENT:
			optimizer_embedding = torch.optim.SGD(
			    [SOURCE_LATENT_CODES],
			    lr=args["learning_rate"],
			    momentum=args["momentum"],
			    weight_decay=args["weight_decay"],
			)
			if (DIST_FUNC == "mahalanobis"):
				optimizer_embedding.add_param_group({"params": SOURCE_VARIANCES})

			if (LOSS_FUNC == "regression"):
				optimizer_embedding.add_param_group({"params": SOURCE_SIGMAS, "lr": 0.1})

		else:
			if (DIST_FUNC == "mahalanobis"):
				optimizer_embedding = torch.optim.SGD(
				    [SOURCE_VARIANCES],
				    lr=args["learning_rate"],
				    momentum=args["momentum"],
				    weight_decay=args["weight_decay"],
				)
			elif (LOSS_FUNC == "regression"):
				optimizer_embedding = torch.optim.SGD(
				    [SOURCE_SIGMAS],
				    lr=args["learning_rate"],
				    momentum=args["momentum"],
				    weight_decay=args["weight_decay"],
				)

	def construct_candidates_dict(loader, encoder):
		print("Creating candidates dict.")
		start_time = time.time()

		for i, batch in enumerate(loader):
			# target_shapes, target_ids, _, _, _ = batch
			target_shapes, target_ids, _, _ = batch

			x = [x.to(device, dtype=torch.float) for x in target_shapes]
			x = torch.stack(x)

			retrieval_latent_codes = encoder(x)
			retrieval_latent_codes = retrieval_latent_codes.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1)
			retrieval_latent_codes = retrieval_latent_codes.view(-1, retrieval_latent_codes.shape[-1])
			
			###Get dummy source labels
			source_label_shape = torch.zeros(target_shapes.shape[0])
			source_labels = source_label_shape.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1)
			source_labels = source_labels.view(-1)
			#Get all labels
			source_labels = get_all_source_labels(source_labels, len(SOURCE_MODEL_INFO))			

			if (NORMALIZE):
				retrieval_latent_codes = F.normalize(retrieval_latent_codes)	
						
			retrieval_latent_codes = retrieval_latent_codes.view(len(SOURCE_MODEL_INFO), -1, TARGET_LATENT_DIM)
			

			if USE_SRC_ENCODER_RETRIEVAL:
				with torch.no_grad():
					## Split to two batches for memory
					src_latent_codes = []
					num_sets = 20
					interval = int(len(source_labels)/num_sets)

					for j in range(num_sets):
						if (j==num_sets-1):
							curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:], SOURCE_MODEL_INFO, encoder, device=device)
						else:
							curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:(j+1)*interval], SOURCE_MODEL_INFO, encoder, device=device)
						src_latent_codes.append(curr_src_latent_codes)

					src_latent_codes = torch.cat(src_latent_codes).view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

			else:
				if (SHARE_SRC_LATENT):
					src_latent_codes = src_latent_codes.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)
				else:
					src_latent_codes = get_source_latent_codes_fixed(source_labels, RETRIEVAL_SOURCE_LATENT_CODES, device=device)
					src_latent_codes = src_latent_codes.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

			if (DIST_FUNC == "mahalanobis"):
				src_variances = get_source_latent_codes_fixed(source_labels, SOURCE_VARIANCES, device=device)
				src_variances = src_variances.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

				if (ACTIVATION_FN.lower() == "none"):
					distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, clip_vec=CLIP_VEC)
				elif (ACTIVATION_FN == "sigmoid"):
					distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, activation_fn=torch.sigmoid, clip_vec=CLIP_VEC)
				elif (ACTIVATION_FN == "relu"):
					distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, activation_fn=torch.relu, clip_vec=CLIP_VEC)		

			elif (DIST_FUNC == "order"):
				if (ACTIVATION_FN.lower() == "none"):
					distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device)
				elif (ACTIVATION_FN == "sigmoid"):
					distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device, activation_fn=torch.sigmoid)
				elif (ACTIVATION_FN == "relu"):
					distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device, activation_fn=torch.relu)	

			sorted_distances, sorted_indices = torch.sort(distances, dim=0)

			sorted_indices = sorted_indices.to("cpu")
			sorted_indices = sorted_indices.detach().numpy().T

			distances = distances.to("cpu")
			distances = distances.detach().numpy().T

			target_ids = target_ids.to("cpu")
			target_ids = target_ids.detach().numpy()

			# print(sorted_indices)
			for j in range(sorted_indices.shape[0]):
				DEFORMATION_CANDIDATES[target_ids[j]] = sorted_indices[j]
				DEFORMATION_DISTANCES[target_ids[j]] = distances[j]

			if (i%20==0):
				print("Time elapsed: "+str(time.time()-start_time)+" sec for batch "+str(i)+ "/"+ str(len(loader))+".")
		return

	def construct_candidates_dict_faster(loader, encoder):
		print("Creating candidates dict. Faster version.")
		start_time = time.time()

		source_labels = np.expand_dims(np.arange(len(SOURCE_MODEL_INFO)), axis=1)
		source_labels = np.reshape(source_labels, (-1))
		print(source_labels.shape)
		##Cache source latent codes
		## Source latent codes
		if USE_SRC_ENCODER_RETRIEVAL:
			with torch.no_grad():
				## Split to two batches for memory
				src_latent_codes_template = []
				num_sets = 20
				interval = int(len(SOURCE_MODEL_INFO)/num_sets)

				for j in range(num_sets):
					if (j==num_sets-1):
						curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:], SOURCE_MODEL_INFO, retrieval_encoder, device=device)
					else:
						curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:(j+1)*interval], SOURCE_MODEL_INFO, retrieval_encoder, device=device)
					src_latent_codes_template.append(curr_src_latent_codes)

				src_latent_codes_template = torch.cat(src_latent_codes_template).view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

		else:
			if (SHARE_SRC_LATENT):
				print("Error in creating dict with shared latent code.")
			else:
				src_latent_codes_template = get_source_latent_codes_fixed(source_labels, RETRIEVAL_SOURCE_LATENT_CODES, device=device)
				src_latent_codes_template = src_lsrc_latent_codes_templateatent_codes.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)		

		for i, batch in enumerate(loader):
			# target_shapes, target_ids, _, _, _ = batch
			target_shapes, target_ids, _, _ = batch

			if COMPLEMENTME:
				target_shapes[:,:,2] = -target_shapes[:,:,2]

			x = [x.to(device, dtype=torch.float) for x in target_shapes]
			x = torch.stack(x)

			retrieval_latent_codes = encoder(x)
			retrieval_latent_codes = retrieval_latent_codes.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1)
			retrieval_latent_codes = retrieval_latent_codes.view(-1, retrieval_latent_codes.shape[-1])
			
			###Get dummy source labels
			source_label_shape = torch.zeros(target_shapes.shape[0])
			source_labels = source_label_shape.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1)
			source_labels = source_labels.view(-1)
			#Get all labels
			source_labels = get_all_source_labels(source_labels, len(SOURCE_MODEL_INFO))			

			if (NORMALIZE):
				retrieval_latent_codes = F.normalize(retrieval_latent_codes)	
						
			retrieval_latent_codes = retrieval_latent_codes.view(len(SOURCE_MODEL_INFO), -1, TARGET_LATENT_DIM)
			
			##Repeat the src_latent vector tensor
			src_latent_codes = src_latent_codes_template.repeat(1, target_shapes.shape[0],1)

			if (DIST_FUNC == "mahalanobis"):
				src_variances = get_source_latent_codes_fixed(source_labels, SOURCE_VARIANCES, device=device)
				src_variances = src_variances.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

				if (ACTIVATION_FN.lower() == "none"):
					distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, clip_vec=CLIP_VEC)
				elif (ACTIVATION_FN == "sigmoid"):
					distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, activation_fn=torch.sigmoid, clip_vec=CLIP_VEC)
				elif (ACTIVATION_FN == "relu"):
					distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, activation_fn=torch.relu, clip_vec=CLIP_VEC)		

			elif (DIST_FUNC == "order"):
				if (ACTIVATION_FN.lower() == "none"):
					distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device)
				elif (ACTIVATION_FN == "sigmoid"):
					distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device, activation_fn=torch.sigmoid)
				elif (ACTIVATION_FN == "relu"):
					distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device, activation_fn=torch.relu)	

			sorted_distances, sorted_indices = torch.sort(distances, dim=0)

			sorted_indices = sorted_indices.to("cpu")
			sorted_indices = sorted_indices.detach().numpy().T

			distances = distances.to("cpu")
			distances = distances.detach().numpy().T

			target_ids = target_ids.to("cpu")
			target_ids = target_ids.detach().numpy()

			# print(sorted_indices)
			for j in range(sorted_indices.shape[0]):
				DEFORMATION_CANDIDATES[target_ids[j]] = sorted_indices[j]
				DEFORMATION_DISTANCES[target_ids[j]] = distances[j]

			if (i%20==0):
				print("Time elapsed: "+str(time.time()-start_time)+" sec for batch "+str(i)+ "/"+ str(len(loader))+".")
		return


	def construct_deformation_candidates_dict(loader, encoder):
		print("Creating deformation candidates dict.")

		for i, batch in enumerate(loader):
			# target_shapes, target_ids, _, _, _ = batch
			target_shapes, target_ids, _, _ = batch

			x = [x.to(device, dtype=torch.float) for x in target_shapes]
			x = torch.stack(x)

			target_latent_codes = encoder(x)

			target_latent_codes = target_latent_codes.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1)
			source_label_shape = torch.zeros(target_shapes.shape[0])
			source_labels = source_label_shape.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1)

			## Reshape to (K*batch_size, ...) to feed into the network
			## Source assignments have to be done accordingly
			target_latent_codes = target_latent_codes.view(-1, target_latent_codes.shape[-1])
			source_labels = source_labels.view(-1)

			#Get all labels
			source_labels = get_all_source_labels(source_labels, len(SOURCE_MODEL_INFO))

			##Also overwrite x for chamfer distance					
			x_repeated = x.unsqueeze(0).repeat(source_labels.shape[0],1,1,1)
			x_repeated = x_repeated.view(-1, x_repeated.shape[-2], x_repeated.shape[-1])

			num_sub_epochs = int(source_labels.shape[0]/batch_size) + 1 

			all_cd_loss = []
			for j in range(num_sub_epochs):

				curr_source_labels = source_labels[j*batch_size:(j+1)*batch_size]

				num_samples = curr_source_labels.shape[0]
				if (num_samples <=0):
					break

				curr_target_latent_codes = target_latent_codes[j*batch_size:(j+1)*batch_size, :]
				curr_x_repeated = x_repeated[j*batch_size:(j+1)*batch_size, :, :]		

				###Set up source A matrices and default params based on source_labels of the target
				src_mats, src_default_params, src_connectivity_mat = get_source_info(curr_source_labels, SOURCE_MODEL_INFO, MAX_NUM_PARAMS, use_connectivity= USE_CONNECTIVITY)

				###Set up source latent codes based on source_labels of the target
				src_latent_codes = get_source_latent_codes_fixed(curr_source_labels, SOURCE_LATENT_CODES, device=device)

				mat = [mat.to(device, dtype=torch.float) for mat in src_mats]
				def_param = [def_param.to(device, dtype=torch.float) for def_param in src_default_params]

				mat = torch.stack(mat)
				def_param = torch.stack(def_param)

				## If using connectivity
				if (USE_CONNECTIVITY):
					conn_mat = [conn_mat.to(device, dtype=torch.float) for conn_mat in src_connectivity_mat]
					conn_mat = torch.stack(conn_mat)
				
				concat_latent_code = torch.cat((src_latent_codes, curr_target_latent_codes), dim=1)

				##Param Decoder per part
				# Make the part latent codes of each source into a (K x PART_LATENT_DIM) tensor
				all_params = []
				for k in range(concat_latent_code.shape[0]):
					curr_num_parts = SOURCE_MODEL_INFO[curr_source_labels[k]]["num_parts"]
					curr_code = concat_latent_code[k]
					curr_code_repeated = curr_code.view(1,curr_code.shape[0]).repeat(curr_num_parts, 1)
					
					part_latent_codes = SOURCE_PART_LATENT_CODES[curr_source_labels[k]]

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

				cd_loss, _ = chamfer_distance(output_pcs, curr_x_repeated, batch_reduction=None)
				cd_loss = cd_loss.to("cpu").detach().numpy()

				all_cd_loss.append(cd_loss)

			all_cd_loss = np.array(all_cd_loss)
			all_cd_loss = np.reshape(all_cd_loss, (target_shapes.shape[0], -1))
			
			distances = all_cd_loss
			sorted_indices = np.argsort(distances, axis=1)

			target_ids = target_ids.to("cpu")
			target_ids = target_ids.detach().numpy()

			# print(sorted_indices)
			for j in range(sorted_indices.shape[0]):
				DEFORMATION_CANDIDATES[target_ids[j]] = sorted_indices[j]
				DEFORMATION_DISTANCES[target_ids[j]] = distances[j]

		return

	def get_dict_candidates(target_ids, K):
		target_ids = target_ids.to("cpu")
		target_ids = target_ids.detach().numpy()

		all_selected = []
		all_weights = []
		for target_id in target_ids:
			ranked_candidates = np.array(DEFORMATION_CANDIDATES[target_id])
			distances_candidates = np.array(DEFORMATION_DISTANCES[target_id])

			##softmax of distances
			# distances_candidates = distances_candidates/np.min(distances_candidates)
			if (SELECTION == "retrieval_candidates"):
				distances_candidates = distances_candidates/100.0

			distances_normalized = np.exp(-(distances_candidates - np.max(distances_candidates)))

			probs = distances_normalized / distances_normalized.sum()

			selected_candidates = np.random.choice(np.arange(len(SOURCE_MODEL_INFO)), size=K, replace=False, p= probs)
			all_selected.append(selected_candidates)

			weights = probs[selected_candidates]/np.sum(probs[selected_candidates])
			all_weights.append(weights)

		all_selected = np.array(all_selected).T
		all_selected = all_selected.flatten()

		all_weights = np.array(all_weights).T
		
		print(all_weights.T[0])
		print()

		all_weights = all_weights.flatten()

		return all_selected, all_weights


	if (TO_TRAIN):

		if not FIXED_DEFORMATION:
			target_encoder.train()
			param_decoder.train()
		else:
			target_encoder.eval()
			param_decoder.eval()			

		if not SHARED_ENCODER:
			retrieval_encoder.train()

		best_loss = np.Inf

		## Training loop
		for epoch in range(n_epochs):
			start = datetime.datetime.now()
			scalars = defaultdict(list)

			for i, batch in enumerate(loader):
				'''
				Per batch output:
					self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index], \
					self.corres_source_label[index]
				'''

				# target_shapes, target_ids, _, semantics, source_label_shape = batch
				target_shapes, target_ids, _, semantics = batch
				source_label_shape = torch.zeros(target_shapes.shape[0])

				if COMPLEMENTME:
					target_shapes[:,:,2] = -target_shapes[:,:,2]

				x = [x.to(device, dtype=torch.float) for x in target_shapes]
				x = torch.stack(x)
				##Target Encoder
				target_latent_codes = target_encoder(x)

				## K is the number of sources selected for each target
				## Repeat each K times
				target_latent_codes = target_latent_codes.unsqueeze(0).repeat(K,1,1)
				semantics = semantics.unsqueeze(0).repeat(K,1,1)
				source_labels = source_label_shape.unsqueeze(0).repeat(K,1)

				## Reshape to (K*batch_size, ...) to feed into the network
				## Source assignments have to be done accordingly
				target_latent_codes = target_latent_codes.view(-1, target_latent_codes.shape[-1])
				semantics = semantics.view(-1, semantics.shape[-1])
				source_labels = source_labels.view(-1)

				## Get the source assignments
				## For now, we get all sources 1,2,...,K
				if (SELECTION == "exhaustive"):
					source_labels = get_all_source_labels(source_labels, len(SOURCE_MODEL_INFO))

				elif (SELECTION == "random"):
					source_labels = get_random_labels(source_labels, len(SOURCE_MODEL_INFO))

				elif (SELECTION == "retrieval_candidates"):
					#after epochs where retrieval module is trained
					if (epoch>30):
					# if (1):
						to_select_from_dict=1
					else:
						to_select_from_dict=0

					if (to_select_from_dict):

						##Dict has not been constructed
						if len(DEFORMATION_CANDIDATES.keys())==0:
							if SHARED_ENCODER:
								target_encoder.eval()
								construct_candidates_dict_faster(loader, target_encoder)
								target_encoder.train()
							else:
								retrieval_encoder.eval()
								construct_candidates_dict_faster(loader, retrieval_encoder)
								retrieval_encoder.train()

						source_labels, weights = get_dict_candidates(target_ids, K)

					#get random
					else:
						source_labels = get_random_labels(source_labels, len(SOURCE_MODEL_INFO))

				elif (SELECTION == "deformation_candidates"):
					to_select_from_dict=1

					if len(DEFORMATION_CANDIDATES.keys())==0:
						target_encoder.eval()
						construct_deformation_candidates_dict(loader, target_encoder)
						target_encoder.train()

					source_labels, weights = get_dict_candidates(target_ids, K)

				##Also overwrite x for chamfer distance					
				x_repeat = x.unsqueeze(0).repeat(K,1,1,1)
				x_repeat = x_repeat.view(-1, x_repeat.shape[-2], x_repeat.shape[-1])

				###Set up source A matrices and default params based on source_labels of the target
				src_mats, src_default_params, src_connectivity_mat = get_source_info(source_labels, SOURCE_MODEL_INFO, MAX_NUM_PARAMS, use_connectivity= USE_CONNECTIVITY)
				# src_mats, src_default_params = get_source_info(source_labels, SOURCE_MODEL_INFO, MAX_NUM_PARAMS)
				
				if USE_SRC_ENCODER:
					##Use the encoder to get the source latent code
					src_latent_codes = get_source_latent_codes_encoder(source_labels, SOURCE_MODEL_INFO, target_encoder, device=device)

				else:
					## Autodecoded: Set up source latent codes based on source_labels of the target
					src_latent_codes = get_source_latent_codes_fixed(source_labels, SOURCE_LATENT_CODES, device=device)


				mat = [mat.to(device, dtype=torch.float) for mat in src_mats]
				def_param = [def_param.to(device, dtype=torch.float) for def_param in src_default_params]

				mat = torch.stack(mat)
				def_param = torch.stack(def_param)

				## If using connectivity
				if (USE_CONNECTIVITY):
					conn_mat = [conn_mat.to(device, dtype=torch.float) for conn_mat in src_connectivity_mat]
					conn_mat = torch.stack(conn_mat)

				concat_latent_code = torch.cat((src_latent_codes, target_latent_codes), dim=1)

				##Param Decoder per part

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
				# print(SOURCE_PART_LATENT_CODES[1])

				if (USE_CONNECTIVITY):
					output_pc = get_shape(mat, params, def_param, ALPHA, connectivity_mat=conn_mat)
				else:
					output_pc = get_shape(mat, params, def_param, ALPHA)

				if (PART_LOSS):
					# Target semantics
					seman = [seman.to(device) for seman in semantics]
					seman = torch.stack(seman)

					part_cd_loss = 0

					for j in range(output_pc.shape[0]):
						source_label = source_labels[j]
						src_semantic = SOURCE_SEMANTICS[source_label][0]
						unique_parts = SOURCE_SEMANTICS[source_label][1]

						for k in range(len(unique_parts)):
							curr_label = unique_parts[k]
							output_part = output_pc[j][src_semantic==curr_label]
							target_part = x_repeat[j][seman[j]==curr_label]
							
							output_part = output_part.view(1, output_part.shape[0], output_part.shape[1])
							target_part = target_part.view(1, target_part.shape[0], target_part.shape[1])

							#If part does not exist in the target
							if (target_part.shape[1] <= 0):
								# print("Part not found. Skipping...")
								continue
							
							curr_l, _ = chamfer_distance(output_part, target_part, batch_reduction=None)
							part_cd_loss += curr_l

					total_cd_loss, _ = chamfer_distance(output_pc, x_repeat, batch_reduction=None)
					loss = total_cd_loss + part_cd_loss					

				else:
					cd_loss, _ = chamfer_distance(output_pc, x_repeat, batch_reduction=None)
					loss = cd_loss

				## Get the min loss from the different sources
				loss = loss.view(K, -1)

				# For the fitting loss
				# loss_min, selected_cluster = torch.min(loss, dim=0)

				##if only updating embedding and not deformation
				if (LOSS_FUNC == "triplet" or PROP2 or PROP3 or PROP2_MORE):					
					sorted_loss, sorted_indices = torch.sort(loss, dim=0)
					positive_indices = sorted_indices[0,:]


				if (not FIXED_DEFORMATION) and ((INIT_DEFORMATION and i%2==0) or ((not INIT_DEFORMATION) and (epoch<20 or i%2==0))):
					### To not double compute
					if not (LOSS_FUNC == "triplet" or PROP2 or PROP3 or PROP2_MORE):
						sorted_loss, sorted_indices = torch.sort(loss, dim=0)
						# sorted_indices = torch.argsort(loss, dim=0)
						positive_indices = sorted_indices[0,:]

					### Weighted by embedding distance
					if (SELECTION == "retrieval_candidates" or SELECTION == "deformation_candidates"):
						
						# to_select_from_dict=0

						if (to_select_from_dict):
							#Weighted by chamfer distance for all
							weights = torch.from_numpy(weights).view(K, -1).to(device)
							weighted_loss = torch.mul(weights, loss)
							weighted_loss = torch.sum(weighted_loss, dim=0)
							fitting_loss = torch.mean(weighted_loss)

						else: 
							# ### Minimum fitting loss
							# loss_min = torch.gather(loss, 0, positive_indices.unsqueeze(0))
							# fitting_loss = torch.mean(loss_min)

							# ### Backprop all within a threshold of (1.1 * minimum_fitting_loss)
							# top1_loss = sorted_loss[0,:]

							# thresh_loss = 1.1*top1_loss
							# thresh_loss = thresh_loss.unsqueeze(0).repeat(K, 1)

							# loss_mask = torch.le(loss, thresh_loss)
							# selected_loss = loss[loss_mask]
							# fitting_loss = torch.mean(selected_loss)

							### Backprop all
							fitting_loss = torch.mean(loss)

					else:
						### Minimum fitting loss
						# loss_min = torch.gather(loss, 0, positive_indices.unsqueeze(0))
						# fitting_loss = torch.mean(loss_min)

						# ### Backprop all within a threshold of (1.1 * minimum_fitting_loss)
						# top1_loss = sorted_loss[0,:]

						# thresh_loss = 1.1*top1_loss
						# thresh_loss = thresh_loss.unsqueeze(0).repeat(K, 1)

						# loss_mask = torch.le(loss, thresh_loss)
						# selected_loss = loss[loss_mask]
						# fitting_loss = torch.mean(selected_loss)

						## Backprop all
						fitting_loss = torch.mean(loss)

					if USE_SYMMETRY:
						reflected_pc = get_symmetric(output_pc)
						symmetric_loss, _ = chamfer_distance(output_pc, reflected_pc)
						fitting_loss += symmetric_loss

					optimizer_deformation.zero_grad()
					fitting_loss.backward()
					optimizer_deformation.step()

					scalars["fitting_loss"].append(fitting_loss)

					now = datetime.datetime.now()
					log = "{} | Batch [{:04d}/{:04d}] | fitting loss: {:.4f} |"
					log = log.format(now.strftime("%c"), i, len(loader), fitting_loss.item())
					print(log)

				# elif FIXED_DEFORMATION or (INIT_DEFORMATION and i%2==1) or (epoch >=20 and i%2==1):
				else:
					### Embedding loss
					if not SHARED_ENCODER:
						retrieval_latent_codes_single = retrieval_encoder(x)
						retrieval_latent_codes = retrieval_latent_codes_single.unsqueeze(0).repeat(K,1,1)
						retrieval_latent_codes = retrieval_latent_codes.view(-1, retrieval_latent_codes.shape[-1])

					else:
						retrieval_latent_codes = target_latent_codes

					if (NORMALIZE):
						retrieval_latent_codes = F.normalize(retrieval_latent_codes)

					retrieval_latent_codes = retrieval_latent_codes.view(K, -1, TARGET_LATENT_DIM)

					if USE_SRC_ENCODER_RETRIEVAL:
						##Use the encoder to get the source latent code
						src_latent_codes = get_source_latent_codes_encoder(source_labels, SOURCE_MODEL_INFO, retrieval_encoder, device=device)
						src_latent_codes = src_latent_codes.view(K, -1, SOURCE_LATENT_DIM)

					else:
						if (SHARE_SRC_LATENT):
							src_latent_codes = src_latent_codes.view(K, -1, SOURCE_LATENT_DIM)
						else:
							src_latent_codes = get_source_latent_codes_fixed(source_labels, RETRIEVAL_SOURCE_LATENT_CODES, device=device)
							src_latent_codes = src_latent_codes.view(K, -1, SOURCE_LATENT_DIM)

					if (DIST_FUNC == "mahalanobis"):
						src_variances = get_source_latent_codes_fixed(source_labels, SOURCE_VARIANCES, device=device)
						src_variances = src_variances.view(K, -1, SOURCE_LATENT_DIM)

						if (ACTIVATION_FN.lower() == "none"):
							distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, clip_vec=CLIP_VEC)
						elif (ACTIVATION_FN == "sigmoid"):
							distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, activation_fn=torch.sigmoid, clip_vec=CLIP_VEC)
						elif (ACTIVATION_FN == "relu"):
							distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, activation_fn=torch.relu, clip_vec=CLIP_VEC)		

					elif (DIST_FUNC == "order"):
						if (ACTIVATION_FN.lower() == "none"):
							distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device)
						elif (ACTIVATION_FN == "sigmoid"):
							distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device, activation_fn=torch.sigmoid)
						elif (ACTIVATION_FN == "relu"):
							distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device, activation_fn=torch.relu)	

					if (LOSS_FUNC == "triplet"):
						positive_distances, negative_distances = margin_selection(sorted_indices, distances, K, num_negs=5)
						embedding_loss = margin_loss(positive_distances, negative_distances, MARGIN, device=device)

					elif (LOSS_FUNC == "regression"):
						obj_sigmas =  get_source_latent_codes_fixed(source_labels, SOURCE_SIGMAS, device=device)
						obj_sigmas = obj_sigmas.view(K, -1)
						embedding_loss = regression_loss(distances, loss, obj_sigmas)
					
					embedding_loss = torch.mean(embedding_loss)

					## Re-encode the deformed source shape and enforce it is close to the region
					# Deformed source shape is output_pc
					# Closest model is indexed positive_indices

					if (PROP2):
						# Get the nearest model and re-encode that deformed source shape
						output_pcs = output_pc.view(K, target_shapes.shape[0], target_shapes.shape[1], target_shapes.shape[2])
						positive_indices_repeated = positive_indices.unsqueeze(0).unsqueeze(-1).repeat(1,1,output_pcs.shape[-2]).unsqueeze(-1).repeat(1,1,1,output_pcs.shape[-1])
						output_pcs = torch.gather(output_pcs, 0, positive_indices_repeated)
						output_pcs = output_pcs.view(target_shapes.shape[0], target_shapes.shape[1], target_shapes.shape[2])

						# Get the corresponding source label
						source_labels = source_labels.reshape(K,-1)

						source_labels = torch.from_numpy(source_labels)
						source_labels = source_labels.to(device)
						selected_label = torch.gather(source_labels, 0, positive_indices.unsqueeze(0)).squeeze()

						selected_label = selected_label.to("cpu")
						selected_label = selected_label.detach().numpy()

						if not SHARED_ENCODER:
							deformed_latent_codes = retrieval_encoder(output_pcs)
						else:
							deformed_latent_codes = target_encoder(output_pcs)

						if (NORMALIZE):
							deformed_latent_codes = F.normalize(deformed_latent_codes)

						if (SHARE_SRC_LATENT):
							src_latent_codes = get_source_latent_codes_fixed(selected_label, SOURCE_LATENT_CODES, device=device)
						else:
							src_latent_codes = get_source_latent_codes_fixed(selected_label, RETRIEVAL_SOURCE_LATENT_CODES, device=device)
						

						if (DIST_FUNC == "mahalanobis"):
							src_variances = get_source_latent_codes_fixed(selected_label, SOURCE_VARIANCES, device=device)

							if (ACTIVATION_FN.lower() == "none"):
								pos_distances = compute_mahalanobis(deformed_latent_codes, src_latent_codes, src_variances, clip_vec=CLIP_VEC)
							elif (ACTIVATION_FN == "sigmoid"):
								pos_distances = compute_mahalanobis(deformed_latent_codes, src_latent_codes, src_variances, activation_fn=torch.sigmoid, clip_vec=CLIP_VEC)
							elif (ACTIVATION_FN == "relu"):
								pos_distances = compute_mahalanobis(deformed_latent_codes, src_latent_codes, src_variances, activation_fn=torch.relu, clip_vec=CLIP_VEC)		

						elif (DIST_FUNC == "order"):
							if (ACTIVATION_FN.lower() == "none"):
								pos_distances = order_embedding_distance(deformed_latent_codes, src_latent_codes, device=device)
							elif (ACTIVATION_FN == "sigmoid"):
								pos_distances = order_embedding_distance(deformed_latent_codes, src_latent_codes, device=device, activation_fn=torch.sigmoid)
							elif (ACTIVATION_FN == "relu"):
								pos_distances = order_embedding_distance(deformed_latent_codes, src_latent_codes, device=device, activation_fn=torch.relu)	


						# ### Negative is the target (deformed source should be closer to the region than the target)
						# # target latent codes : retrieval_latent_codes_single

						if (DIST_FUNC == "mahalanobis"):
							if (ACTIVATION_FN.lower() == "none"):
								neg_distances = compute_mahalanobis(retrieval_latent_codes_single, src_latent_codes, src_variances, clip_vec=CLIP_VEC)
							elif (ACTIVATION_FN == "sigmoid"):
								neg_distances = compute_mahalanobis(retrieval_latent_codes_single, src_latent_codes, src_variances, activation_fn=torch.sigmoid, clip_vec=CLIP_VEC)
							elif (ACTIVATION_FN == "relu"):
								neg_distances = compute_mahalanobis(retrieval_latent_codes_single, src_latent_codes, src_variances, activation_fn=torch.relu, clip_vec=CLIP_VEC)		


						elif (DIST_FUNC == "order"):
							if (ACTIVATION_FN.lower() == "none"):
								neg_distances = order_embedding_distance(retrieval_latent_codes_single, src_latent_codes, device=device)
							elif (ACTIVATION_FN == "sigmoid"):
								neg_distances = order_embedding_distance(retrieval_latent_codes_single, src_latent_codes, device=device, activation_fn=torch.sigmoid)
							elif (ACTIVATION_FN == "relu"):
								neg_distances = order_embedding_distance(retrieval_latent_codes_single, src_latent_codes, device=device, activation_fn=torch.relu)	
				

						# ### Negatives are distance of deformed source and other sources
						# ############## Select random negative index
						# num_negs = 5
						# neg_idx = np.tile(np.arange(len(SOURCE_MODEL_INFO)), (src_latent_codes.shape[0],1))
						# mask = np.ones((src_latent_codes.shape[0],len(SOURCE_MODEL_INFO)), dtype=bool)
						# mask[range(src_latent_codes.shape[0]), selected_label] = False
						# neg_idx_ = neg_idx[mask].reshape(src_latent_codes.shape[0], len(SOURCE_MODEL_INFO)-1)
						# # print(neg_idx)
						# # print(neg_idx_)

						# negative_candidates = torch.from_numpy(neg_idx_)
						# negative_candidates = negative_candidates.to(device)

						# random_idx = torch.randint(0, len(SOURCE_MODEL_INFO)-1, (deformed_latent_codes.shape[0],num_negs))				
						# # print(random_idx)

						# random_idx = random_idx.view(-1)

						# if (SHARE_SRC_LATENT):
						# 	src_latent_codes = get_source_latent_codes_fixed(random_idx, SOURCE_LATENT_CODES, device=device)
						# else:
						# 	src_latent_codes = get_source_latent_codes_fixed(random_idx, RETRIEVAL_SOURCE_LATENT_CODES, device=device)

						# src_latent_codes = src_latent_codes.view(num_negs, -1, SOURCE_LATENT_DIM)

						# if (DIST_FUNC == "mahalanobis"):
						# 	src_variances = get_source_latent_codes_fixed(random_idx, SOURCE_VARIANCES, device=device)
						# 	src_variances = src_variances.view(num_negs, -1, SOURCE_LATENT_DIM)

						# 	if (ACTIVATION_FN.lower() == "none"):
						# 		neg_distances = compute_mahalanobis(deformed_latent_codes, src_latent_codes, src_variances, clip_vec=CLIP_VEC)
						# 	elif (ACTIVATION_FN == "sigmoid"):
						# 		neg_distances = compute_mahalanobis(deformed_latent_codes, src_latent_codes, src_variances, activation_fn=torch.sigmoid, clip_vec=CLIP_VEC)
						# 	elif (ACTIVATION_FN == "relu"):
						# 		neg_distances = compute_mahalanobis(deformed_latent_codes, src_latent_codes, src_variances, activation_fn=torch.relu, clip_vec=CLIP_VEC)		

						# elif (DIST_FUNC == "order"):
						# 	if (ACTIVATION_FN.lower() == "none"):
						# 		neg_distances = order_embedding_distance(deformed_latent_codes, src_latent_codes, device=device)
						# 	elif (ACTIVATION_FN == "sigmoid"):
						# 		neg_distances = order_embedding_distance(deformed_latent_codes, src_latent_codes, activation_fn=torch.sigmoid, device=device)
						# 	elif (ACTIVATION_FN == "relu"):
						# 		neg_distances = order_embedding_distance(deformed_latent_codes, src_latent_codes, activation_fn=torch.relu, device=device)	

						# # print(neg_distances.shape)
						# pos_distances = pos_distances.unsqueeze(0).repeat(num_negs, 1)
						# # print(pos_distances.shape)
						# ##############################################


						###Compute for margin loss						
						prop2_loss = margin_loss(pos_distances, neg_distances, 0.0, device=device)
						prop2_loss = torch.mean(prop2_loss)

						print("Property 2 loss")
						print(prop2_loss)

						embedding_loss += 0.5*prop2_loss
					################

					if (PROP2_MORE):
						# Get the nearest model and re-encode that deformed source shape
						num_to_select = 5
						positive_indices = sorted_indices[:num_to_select,:]

						output_pcs = output_pc.view(K, target_shapes.shape[0], target_shapes.shape[1], target_shapes.shape[2])
						positive_indices_repeated = positive_indices.unsqueeze(-1).repeat(1,1,output_pcs.shape[-2]).unsqueeze(-1).repeat(1,1,1,output_pcs.shape[-1])
						output_pcs = torch.gather(output_pcs, 0, positive_indices_repeated)
						output_pcs = output_pcs.view(-1, target_shapes.shape[1], target_shapes.shape[2])


						# Get the corresponding source label
						source_labels = source_labels.reshape(K,-1)

						source_labels = torch.from_numpy(source_labels)
						source_labels = source_labels.to(device)
						selected_label = torch.gather(source_labels, 0, positive_indices).squeeze()

						selected_label = selected_label.view(-1)

						selected_label = selected_label.to("cpu")
						selected_label = selected_label.detach().numpy()

						if not SHARED_ENCODER:
							deformed_latent_codes = retrieval_encoder(output_pcs)
						else:
							deformed_latent_codes = target_encoder(output_pcs)


						if (NORMALIZE):
							deformed_latent_codes = F.normalize(deformed_latent_codes)

						if (SHARE_SRC_LATENT):
							src_latent_codes = get_source_latent_codes_fixed(selected_label, SOURCE_LATENT_CODES, device=device)
						else:
							src_latent_codes = get_source_latent_codes_fixed(selected_label, RETRIEVAL_SOURCE_LATENT_CODES, device=device)
						

						if (DIST_FUNC == "mahalanobis"):
							src_variances = get_source_latent_codes_fixed(selected_label, SOURCE_VARIANCES, device=device)

							if (ACTIVATION_FN.lower() == "none"):
								pos_distances = compute_mahalanobis(deformed_latent_codes, src_latent_codes, src_variances, clip_vec=CLIP_VEC)
							elif (ACTIVATION_FN == "sigmoid"):
								pos_distances = compute_mahalanobis(deformed_latent_codes, src_latent_codes, src_variances, activation_fn=torch.sigmoid, clip_vec=CLIP_VEC)
							elif (ACTIVATION_FN == "relu"):
								pos_distances = compute_mahalanobis(deformed_latent_codes, src_latent_codes, src_variances, activation_fn=torch.relu, clip_vec=CLIP_VEC)		

						elif (DIST_FUNC == "order"):
							if (ACTIVATION_FN.lower() == "none"):
								pos_distances = order_embedding_distance(deformed_latent_codes, src_latent_codes, device=device)
							elif (ACTIVATION_FN == "sigmoid"):
								pos_distances = order_embedding_distance(deformed_latent_codes, src_latent_codes, device=device, activation_fn=torch.sigmoid)
							elif (ACTIVATION_FN == "relu"):
								pos_distances = order_embedding_distance(deformed_latent_codes, src_latent_codes, device=device, activation_fn=torch.relu)	


						# ### Negative is the target (deformed source should be closer to the region than the target)
						# # target latent codes : retrieval_latent_codes_single

						target_latent_codes = retrieval_latent_codes_single.unsqueeze(0).repeat(num_to_select,1,1)
						target_latent_codes = target_latent_codes.view(-1, target_latent_codes.shape[-1])

						if (DIST_FUNC == "mahalanobis"):
							if (ACTIVATION_FN.lower() == "none"):
								neg_distances = compute_mahalanobis(target_latent_codes, src_latent_codes, src_variances, clip_vec=CLIP_VEC)
							elif (ACTIVATION_FN == "sigmoid"):
								neg_distances = compute_mahalanobis(target_latent_codes, src_latent_codes, src_variances, activation_fn=torch.sigmoid, clip_vec=CLIP_VEC)
							elif (ACTIVATION_FN == "relu"):
								neg_distances = compute_mahalanobis(target_latent_codes, src_latent_codes, src_variances, activation_fn=torch.relu, clip_vec=CLIP_VEC)		


						elif (DIST_FUNC == "order"):
							if (ACTIVATION_FN.lower() == "none"):
								neg_distances = order_embedding_distance(target_latent_codes, src_latent_codes, device=device)
							elif (ACTIVATION_FN == "sigmoid"):
								neg_distances = order_embedding_distance(target_latent_codes, src_latent_codes, device=device, activation_fn=torch.sigmoid)
							elif (ACTIVATION_FN == "relu"):
								neg_distances = order_embedding_distance(target_latent_codes, src_latent_codes, device=device, activation_fn=torch.relu)	


						###Compute for margin loss						
						prop2_loss = margin_loss(pos_distances, neg_distances, 0.0, device=device)
						prop2_loss = torch.mean(prop2_loss)

						print("Property 2 loss")
						print(prop2_loss)

						embedding_loss += 0.5*prop2_loss
					################


					### Property 3: projection of the target onto the source region ~ re-encoded deformed source
					if (PROP3):
						if (DIST_FUNC == "mahalanobis"):
							print("ERROR property 3 cannot be be enforced using the mahalanobis distance.")

						# Deformed source shape is output_pc
						# Closest model is indexed positive_indices

						# Get the nearest model and re-encode that deformed source shape
						output_pcs = output_pc.view(K, target_shapes.shape[0], target_shapes.shape[1], target_shapes.shape[2])
						positive_indices_repeated = positive_indices.unsqueeze(0).unsqueeze(-1).repeat(1,1,output_pcs.shape[-2]).unsqueeze(-1).repeat(1,1,1,output_pcs.shape[-1])
						output_pcs = torch.gather(output_pcs, 0, positive_indices_repeated)
						output_pcs = output_pcs.view(target_shapes.shape[0], target_shapes.shape[1], target_shapes.shape[2])

						# Get the corresponding source label
						source_labels = source_labels.reshape(K,-1)

						source_labels = torch.from_numpy(source_labels)
						source_labels = source_labels.to(device)
						selected_label = torch.gather(source_labels, 0, positive_indices.unsqueeze(0)).squeeze()
						selected_label = selected_label.to("cpu")
						selected_label = selected_label.detach().numpy()

						if not SHARED_ENCODER:
							deformed_latent_codes = retrieval_encoder(output_pcs)
						else:
							deformed_latent_codes = target_encoder(output_pcs)

						if (NORMALIZE):
							deformed_latent_codes = F.normalize(deformed_latent_codes)

						# deformed_latent_codes = deformed_latent_codes.view(K, -1, TARGET_LATENT_DIM)

						if (SHARE_SRC_LATENT):
							src_latent_codes = get_source_latent_codes_fixed(selected_label, SOURCE_LATENT_CODES, device=device)
						else:
							src_latent_codes = get_source_latent_codes_fixed(selected_label, RETRIEVAL_SOURCE_LATENT_CODES, device=device)
						

						if (DIST_FUNC == "order"):
							if (ACTIVATION_FN.lower() == "none"):
								retrieval_latent_projected = order_embedding_projection(retrieval_latent_codes_single, src_latent_codes, device=device)
							elif (ACTIVATION_FN == "sigmoid"):
								retrieval_latent_projected = order_embedding_projection(retrieval_latent_codes_single, src_latent_codes, device=device, activation_fn=torch.sigmoid)
							elif (ACTIVATION_FN == "relu"):
								retrieval_latent_projected = order_embedding_projection(retrieval_latent_codes_single, src_latent_codes, device=device, activation_fn=torch.relu)


						if (ACTIVATION_FN.lower() == "none"):
							diff = order_embedding_distance(deformed_latent_codes, retrieval_latent_projected, device=device, projected=1)
						elif (ACTIVATION_FN == "sigmoid"):
							diff = order_embedding_distance(deformed_latent_codes, retrieval_latent_projected, device=device, activation_fn=torch.sigmoid, projected=1)
						elif (ACTIVATION_FN == "relu"):
							diff = order_embedding_distance(deformed_latent_codes, retrieval_latent_projected, device=device, activation_fn=torch.relu, projected=1)	

						# diff = torch.sum(torch.square(retrieval_latent_codes - deformed_latent_codes), dim=-1)
						prop3_loss = torch.mean(diff)						

						print("Property 3 loss")
						print(prop3_loss)

						embedding_loss += 0.5*prop3_loss

					optimizer_embedding.zero_grad()
					embedding_loss.backward()
					optimizer_embedding.step()

					scalars["embedding_loss"].append(embedding_loss)

					now = datetime.datetime.now()
					log = "Epoch: {} | Batch [{:04d}/{:04d}] | embedding loss: {:.4f} |"
					log = log.format(str(epoch)+'/'+str(n_epochs), i, len(loader), embedding_loss.item())
					print(log)

				gc.collect()

			if ((epoch+1) %10 == 0):
			# if (1):
				# Summary after each epoch
				summary = {}
				now = datetime.datetime.now()
				duration = (now - start).total_seconds()
				log = "> {} | Epoch [{:04d}/{:04d}] | duration: {:.1f}s |"
				log = log.format(now.strftime("%c"), epoch, args["epochs"], duration)
				for m, v in scalars.items():
				    summary[m] = torch.stack(v).mean()
				    log += " {}: {:.4f} |".format(m, summary[m].item())

				fname = os.path.join(LOG_DIR, "checkpoint_{:04d}.pth".format(epoch))
				print("> Saving model to {}...".format(fname))
				model = {"target_encoder": target_encoder.state_dict(),
						"param_decoder":param_decoder.state_dict(),
						"source_latent_codes":SOURCE_LATENT_CODES,
						"part_latent_codes": SOURCE_PART_LATENT_CODES}

				if (DIST_FUNC == "mahalanobis"):
					model["source_variances"] = SOURCE_VARIANCES

				if not SHARE_SRC_LATENT:
					model["retrieval_source_latent_codes"] = RETRIEVAL_SOURCE_LATENT_CODES

				if not SHARED_ENCODER:
					model["retrieval_encoder"] =retrieval_encoder.state_dict()

				torch.save(model, fname)

				if epoch >20 and summary["embedding_loss"] < best_loss:
					best_loss = summary["embedding_loss"]
					fname = os.path.join(LOG_DIR, "model.pth")
					print("> Saving model to {}...".format(fname))
					model = {"target_encoder": target_encoder.state_dict(),
							"param_decoder":param_decoder.state_dict(),
							"source_latent_codes":SOURCE_LATENT_CODES,
							"part_latent_codes": SOURCE_PART_LATENT_CODES}

					if (DIST_FUNC == "mahalanobis"):
						model["source_variances"] = SOURCE_VARIANCES

					if not SHARE_SRC_LATENT:
						model["retrieval_source_latent_codes"] = RETRIEVAL_SOURCE_LATENT_CODES

					if not SHARED_ENCODER:
						model["retrieval_encoder"] =retrieval_encoder.state_dict()	

					torch.save(model, fname)
				log += " best: {:.4f} |".format(best_loss)

				fname = os.path.join(LOG_DIR, "train.log")
				with open(fname, "a") as fp:
				    fp.write(log + "\n")

				print(log)
				print("--------------------------------------------------------------------------")

			### For candidate selection
			if (SELECTION == "retrieval_candidates"):
				if (epoch>=30 and epoch%5==0):
					to_update_dict=1
				else:
					to_update_dict=0

				if (to_update_dict):
					if SHARED_ENCODER:
						target_encoder.eval()
						construct_candidates_dict_faster(loader, target_encoder)
						target_encoder.train()
					else:
						retrieval_encoder.eval()
						construct_candidates_dict_faster(loader, retrieval_encoder)
						retrieval_encoder.train()


			if (SELECTION == "deformation_candidates"):
				if (epoch%10==0):
					to_update_dict=1
				else:
					to_update_dict=0

				if (to_update_dict):
					target_encoder.eval()
					construct_deformation_candidates_dict(loader, target_encoder)
					target_encoder.train()



	else:		
		fname = os.path.join(LOG_DIR, "model.pth")
		target_encoder.load_state_dict(torch.load(fname)["target_encoder"])
		target_encoder.to(device)
		target_encoder.eval()

		param_decoder.load_state_dict(torch.load(fname)["param_decoder"])
		param_decoder.to(device)
		param_decoder.eval()

		if not SHARED_ENCODER:
			retrieval_encoder.load_state_dict(torch.load(fname)["retrieval_encoder"])
			retrieval_encoder.to(device)
			retrieval_encoder.eval()

		SOURCE_LATENT_CODES = torch.load(fname)["source_latent_codes"]
		SOURCE_PART_LATENT_CODES = torch.load(fname)["part_latent_codes"]

		if (DIST_FUNC == "mahalanobis"):
			SOURCE_VARIANCES = torch.load(fname)["source_variances"]

		if not SHARE_SRC_LATENT:
			RETRIEVAL_SOURCE_LATENT_CODES = torch.load(fname)["retrieval_source_latent_codes"]

		####After training get visualization results
		num_evaluated = 0
		total_cd_error = 0
		num_correct_retrieved= 0
		used_sources = []

		for i, batch in enumerate(loader):
			'''
			Per batch output:
				self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index], \
				self.corres_source_label[index]
			'''			
			# target_shapes, target_ids, target_labels, _, source_label_shape = batch
			target_shapes, target_ids, target_labels, _ = batch
			
			source_label_shape = torch.zeros(target_shapes.shape[0])

			x = [x.to(device, dtype=torch.float) for x in target_shapes]
			x = torch.stack(x)

			##Target Encoder
			target_latent_codes = target_encoder(x)

			if (EVAL_SELECTION == "random"):
				source_labels = get_random_labels(source_label_shape, len(SOURCE_MODEL_INFO))

			elif (EVAL_SELECTION == "minimum" or EVAL_SELECTION == "retrieval"):
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

			###Set up source A matrices and default params based on source_labels of the target
			src_mats, src_default_params, src_connectivity_mat = get_source_info(source_labels, SOURCE_MODEL_INFO, MAX_NUM_PARAMS, use_connectivity= USE_CONNECTIVITY)
			# src_mats, src_default_params = get_source_info(source_labels, SOURCE_MODEL_INFO, MAX_NUM_PARAMS)

			if USE_SRC_ENCODER:
				##Use the encoder to get the source latent code
				src_latent_codes = get_source_latent_codes_encoder(source_labels, SOURCE_MODEL_INFO, target_encoder, device=device)

			else:
				## Autodecoded: Set up source latent codes based on source_labels of the target
				src_latent_codes = get_source_latent_codes_fixed(source_labels, SOURCE_LATENT_CODES, device=device)

			mat = [mat.to(device, dtype=torch.float) for mat in src_mats]
			def_param = [def_param.to(device, dtype=torch.float) for def_param in src_default_params]

			mat = torch.stack(mat)
			def_param = torch.stack(def_param)

			## If using connectivity
			if (USE_CONNECTIVITY):
				conn_mat = [conn_mat.to(device, dtype=torch.float) for conn_mat in src_connectivity_mat]
				conn_mat = torch.stack(conn_mat)
				
			concat_latent_code = torch.cat((src_latent_codes, target_latent_codes), dim=1)

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

			if (EVAL_SELECTION == "random"):
				retrieved_idx = source_labels

			elif (EVAL_SELECTION == "minimum" or EVAL_SELECTION == "retrieval"):			
				output_pcs = output_pcs.view(len(SOURCE_MODEL_INFO), target_shapes.shape[0], target_shapes.shape[1], target_shapes.shape[2])
				cd_loss = cd_loss.view(len(SOURCE_MODEL_INFO), -1)

				if (EVAL_SELECTION == "minimum"):
					##Selection
					sorted_indices = torch.argsort(cd_loss, dim=0)
					retrieved_idx = sorted_indices[0,:] 

				elif (EVAL_SELECTION == "retrieval"):
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
						##Use the encoder to get the source latent code
						src_latent_codes = get_source_latent_codes_encoder(source_labels, SOURCE_MODEL_INFO, retrieval_encoder, device=device)
						src_latent_codes = src_latent_codes.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

					else:
						if (SHARE_SRC_LATENT):
							src_latent_codes = src_latent_codes.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)
						else:
							src_latent_codes = get_source_latent_codes_fixed(source_labels, RETRIEVAL_SOURCE_LATENT_CODES, device=device)
							src_latent_codes = src_latent_codes.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

					if (DIST_FUNC == "mahalanobis"):
						src_variances = get_source_latent_codes_fixed(source_labels, SOURCE_VARIANCES, device=device)
						src_variances = src_variances.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

						if (ACTIVATION_FN.lower() == "none"):
							distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, clip_vec=CLIP_VEC)
						elif (ACTIVATION_FN == "sigmoid"):
							distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, activation_fn=torch.sigmoid, clip_vec=CLIP_VEC)
						elif (ACTIVATION_FN == "relu"):
							distances = compute_mahalanobis(retrieval_latent_codes, src_latent_codes, src_variances, activation_fn=torch.relu, clip_vec=CLIP_VEC)		

					elif (DIST_FUNC == "order"):
						if (ACTIVATION_FN.lower() == "none"):
							distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device)
						elif (ACTIVATION_FN == "sigmoid"):
							distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device, activation_fn=torch.sigmoid)
						elif (ACTIVATION_FN == "relu"):
							distances = order_embedding_distance(retrieval_latent_codes, src_latent_codes, device=device, activation_fn=torch.relu)	

					sorted_indices = torch.argsort(distances, dim=0)
					retrieved_idx = sorted_indices[0,:] 

					#To compute by some "accuracy"
					sorted_by_cd = torch.argsort(cd_loss, dim=0)
					cd_retrieved = sorted_by_cd[0,:]
					cd_retrieved = cd_retrieved.to("cpu")
					cd_retrieved = cd_retrieved.detach().numpy()	

				retrieved_idx_repeated = retrieved_idx.unsqueeze(0).unsqueeze(-1).repeat(1,1,output_pcs.shape[-2]).unsqueeze(-1).repeat(1,1,1,output_pcs.shape[-1])

				output_pcs = torch.gather(output_pcs, 0, retrieved_idx_repeated)
				output_pcs = output_pcs.view(target_shapes.shape[0], target_shapes.shape[1], target_shapes.shape[2])

				cd_loss = torch.gather(cd_loss, 0, retrieved_idx.unsqueeze(0))
				cd_loss = cd_loss.view(-1)
				
				retrieved_idx = retrieved_idx.to("cpu")
				retrieved_idx = retrieved_idx.detach().numpy()

			## For visualization			
			output_pcs = output_pcs.to("cpu")
			output_pcs = output_pcs.detach().numpy()

			target_shapes = target_shapes.to("cpu")
			target_shapes = target_shapes.detach().numpy()
			target_labels = target_labels.to("cpu")
			target_labels = target_labels.detach().numpy()
			target_ids = target_ids.to("cpu")
			target_ids = target_ids.detach().numpy()


			cd_loss = cd_loss.to("cpu")
			cd_loss = cd_loss.detach().numpy()
						
			'''
			Get source points, ids and labels
			'''
			src_points, src_labels, src_ids = get_source_info_visualization(retrieved_idx, SOURCE_MODEL_INFO)

			if (EVAL_SELECTION == "retrieval"):
				correct_retrieved = np.equal(retrieved_idx, cd_retrieved)

			for j in range(output_pcs.shape[0]):
				num_evaluated += 1
				total_cd_error += cd_loss[j]

				if not src_ids[j] in used_sources:
					used_sources.append(src_ids[j])

				if (EVAL_SELECTION == "retrieval"):
					num_correct_retrieved += correct_retrieved[j]

				if (TO_VISU):
					target_points = target_shapes[j]
					output_pc = output_pcs[j]			
					output_visualization(output_pc, src_points[j], target_points, src_labels[j], target_labels[j], src_ids[j], target_ids[j], DUMP_DIR)  

		### Save numerical results
		mean_cd_loss = total_cd_error/float(num_evaluated)
		log_string("Num evaluated= "+str(num_evaluated))
		log_string("")
		log_string("Number of unique selected sources: "+str(len(used_sources))+"/"+str(len(SOURCE_MODEL_INFO)))
		log_string("")
		log_string("Mean CD error= "+str(mean_cd_loss))

		if (EVAL_SELECTION == "retrieval"):
			accuracy = float(num_correct_retrieved)/num_evaluated
			log_string("Retrieval accuracy= "+str(accuracy))


