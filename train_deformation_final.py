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


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_chairs_final.json", help="path to the json config file", type=str)
parser.add_argument("--logdir", default="log_test/", help="path to the log directory", type=str)
parser.add_argument('--dump_dir', default= "dump_test/", type=str)
parser.add_argument('--category', default= "chair", type=str)

parser.add_argument('--to_train', default= False, type=bool)

parser.add_argument('--part_loss', default= False, type=bool)
parser.add_argument('--use_bn', default= False, type=bool)
parser.add_argument('--selection', default= "random", type=str) # can also be candidates

parser.add_argument('--use_connectivity', default= True, type=bool)
parser.add_argument('--use_symmetry', default= False, type=bool)
parser.add_argument('--use_singleaxis', default= False, type=bool)
parser.add_argument('--use_keypoint', default= False, type=bool)

parser.add_argument('--init_deformation', default= False, type=bool)
parser.add_argument('--model_init', default= "log_region_init_mh_reg/", type=str)

parser.add_argument('--visualize', default= False, type=bool)
parser.add_argument('--eval_selection', default= "minimum", type=str)

parser.add_argument('--num_sources', default= 200, type=int)
parser.add_argument('--complementme', default= False, type=bool)

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

TARGET_DATA_DIR = args["data_dir"]
#OBJ_CAT = args["category"]
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
SELECTION = FLAGS.selection
USE_BN = FLAGS.use_bn

INIT_DEFORMATION = FLAGS.init_deformation
MODEL_INIT = FLAGS.model_init

TO_VISU = FLAGS.visualize
EVAL_SELECTION = FLAGS.eval_selection

USE_CONNECTIVITY = FLAGS.use_connectivity
USE_SYMMETRY = FLAGS.use_symmetry
USE_SINGLEAXIS = FLAGS.use_singleaxis
USE_KEYPOINT = FLAGS.use_keypoint

NUM_SOURCES = FLAGS.num_sources
print("Num sources: "+str(NUM_SOURCES))

COMPLEMENTME = FLAGS.complementme

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


if __name__ == "__main__":

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

	if (TO_TRAIN):
		DATA_SPLIT = "train"
		batch_size = args["batch_size"]
	else:
		DATA_SPLIT = "test"	
		batch_size = 2

	if COMPLEMENTME:
		filename = os.path.join("generated_datasplits_complementme", OBJ_CAT+"_"+str(NUM_SOURCES)+"_"+DATA_SPLIT+".h5")

	else:
		#### Get data for all target models
		filename = os.path.join("generated_datasplits", OBJ_CAT+"_"+str(NUM_SOURCES)+"_"+DATA_SPLIT+".h5")

	dataset = StructureNetDataset_h5(filename)
				
	to_shuffle = TO_TRAIN
	print(to_shuffle)

	loader = torch.utils.data.DataLoader(
	    dataset,
	    batch_size=batch_size,
	    num_workers=args["num_workers"],
	    pin_memory=True,
	    shuffle=to_shuffle,
	)

	#### Torch
	device = args["device"]


	## Get max number of params for the embedding size
	MAX_NUM_PARAMS = -1
	MAX_NUM_PARTS = -1
	SOURCE_MODEL_INFO = []
	SOURCE_SEMANTICS = []
	SOURCE_PART_LATENT_CODES = []

	print("Loading sources...")
	for source_model in sources:
		src_filename = str(source_model) + "_leaves.h5"
		# box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic = get_model(os.path.join(data_fol, src_filename), semantic=True)
		if (USE_CONNECTIVITY):
			box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic, \
								constraint_mat,	constraint_proj_mat	= get_model(os.path.join(src_data_fol, src_filename), semantic=True, constraint=True)
		else:
			box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic = get_model(os.path.join(src_data_fol, src_filename), semantic=True)

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

		part_latent_codes = torch.autograd.Variable(torch.randn((num_parts,PART_LATENT_DIM), dtype=torch.float, device=device), requires_grad=True)
		SOURCE_PART_LATENT_CODES.append(part_latent_codes)

	print("Done loading sources.")
	print(len(SOURCE_MODEL_INFO))
	print(MAX_NUM_PARAMS)
	print(MAX_NUM_PARTS)
	embedding_size = 6

	## Create source latent
	SOURCE_LATENT_CODES = torch.autograd.Variable(torch.randn((len(sources),SOURCE_LATENT_DIM), dtype=torch.float, device=device), requires_grad=True)

	## Define Networks
	target_encoder = TargetEncoder(
	    TARGET_LATENT_DIM,
	    args["input_channels"],
	)
	target_encoder.to(device, dtype=torch.float)

	decoder_input_dim = TARGET_LATENT_DIM + SOURCE_LATENT_DIM + PART_LATENT_DIM
	param_decoder = ParamDecoder2(decoder_input_dim, 256, embedding_size)
	param_decoder.to(device, dtype=torch.float)

	if (INIT_DEFORMATION):
		#Load model
		fname = os.path.join(MODEL_INIT, "model.pth")
		target_encoder.load_state_dict(torch.load(fname)["target_encoder"])
		target_encoder.to(device)

		param_decoder.load_state_dict(torch.load(fname)["param_decoder"])
		param_decoder.to(device)

		SOURCE_LATENT_CODES = torch.load(fname)["source_latent_codes"]
		SOURCE_PART_LATENT_CODES = torch.load(fname)["part_latent_codes"]
		print("Model Loaded.")

	## Define loss and optimizer
	learning_rate = args["learning_rate"]
	n_epochs = args["epochs"]

	target_encoder_params = list(target_encoder.parameters())
	decoder_params = list(param_decoder.parameters())

	all_params = target_encoder_params + decoder_params
	print(len(all_params))


	optimizer = torch.optim.SGD(
	    all_params,
	    lr=args["learning_rate"],
	    momentum=args["momentum"],
	    weight_decay=args["weight_decay"],
	)

	optimizer.add_param_group({"params": SOURCE_LATENT_CODES, "lr": args["lr_autodecoder"]})
	optimizer.add_param_group({"params": SOURCE_PART_LATENT_CODES, "lr": args["lr_autodecoder"]})


	if (TO_TRAIN):

		target_encoder.train()
		param_decoder.train()
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
				target_shapes, _, target_labels, semantics = batch
				source_label_shape = torch.zeros(target_shapes.shape[0])

				if COMPLEMENTME:
					target_shapes[:,:,2] = -target_shapes[:,:,2]

				if (SELECTION == "random"):
					source_labels = get_random_labels(source_label_shape, len(SOURCE_MODEL_INFO))

				else:
					print("Error in selection type. To implement.")
					exit()
				# elif (SELECTION == "candidates"):
				# 	source_labels = source_labels.to("cpu")
				# 	source_labels = source_labels.detach().numpy()

				###Set up source A matrices and default params based on source_labels of the target
				src_mats, src_default_params, src_connectivity_mat = get_source_info(source_labels, SOURCE_MODEL_INFO, MAX_NUM_PARAMS, use_connectivity= USE_CONNECTIVITY)
				###Set up source latent codes based on source_labels of the target
				src_latent_codes = get_source_latent_codes_fixed(source_labels, SOURCE_LATENT_CODES, device=device)
				# print(SOURCE_LATENT_CODES)
				# print(src_latent_codes.shape)

				x = [x.to(device, dtype=torch.float) for x in target_shapes]
				mat = [mat.to(device, dtype=torch.float) for mat in src_mats]
				def_param = [def_param.to(device, dtype=torch.float) for def_param in src_default_params]

				x = torch.stack(x)
				mat = torch.stack(mat)
				def_param = torch.stack(def_param)

				## If using connectivity
				if (USE_CONNECTIVITY):
					conn_mat = [conn_mat.to(device, dtype=torch.float) for conn_mat in src_connectivity_mat]
					conn_mat = torch.stack(conn_mat)

				##Target Encoder
				target_latent_codes = target_encoder(x)

				concat_latent_code = torch.cat((src_latent_codes, target_latent_codes), dim=1)

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

				# print(SOURCE_PART_LATENT_CODES[0])

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

						for i in range(len(unique_parts)):
							curr_label = unique_parts[i]
							output_part = output_pc[j][src_semantic==curr_label]
							target_part = x[j][seman[j]==curr_label]
							
							output_part = output_part.view(1, output_part.shape[0], output_part.shape[1])
							target_part = target_part.view(1, target_part.shape[0], target_part.shape[1])

							#If part does not exist in the target
							if (target_part.shape[1] <= 0):
								print("Part not found. Skipping...")
								continue
							
							curr_l, _ = chamfer_distance(output_part, target_part)
							part_cd_loss += curr_l

							# print(curr_l)

					total_cd_loss, _ = chamfer_distance(output_pc, x)
					loss = total_cd_loss + part_cd_loss					

				else:
					cd_loss, _ = chamfer_distance(output_pc, x)
					loss = cd_loss

				# norm_loss = torch.sum(torch.norm(params, dim=1))
				# loss = cd_loss + norm_loss

				## Symmetry loss
				if USE_SYMMETRY:
					reflected_pc = get_symmetric(output_pc)
					symmetric_loss, _ = chamfer_distance(output_pc, reflected_pc)
					loss += symmetric_loss

					# ### Debug by visualizing
					# output_pc = output_pc.to("cpu")
					# output_pc = output_pc.detach().numpy()					
					# reflected_pc = reflected_pc.to("cpu")
					# reflected_pc = reflected_pc.detach().numpy()
					# target_labels = target_labels.to("cpu")
					# target_labels = target_labels.detach().numpy()

					# for i in range(output_pc.shape[0]):
					# 	curr_shape = output_pc[i]
					# 	curr_reflected = reflected_pc[i]
					# 	curr_labels = target_labels[i]

					# 	# Save point cloud.
					# 	out_point_cloud_file = os.path.join(LOG_DIR, 'output_points.xyz')
					# 	np.savetxt(out_point_cloud_file, curr_shape, delimiter=' ', fmt='%f')
					# 	print("Saved '{}'.".format(out_point_cloud_file))

					# 	# Save point cloud.
					# 	reflected_point_cloud_file = os.path.join(LOG_DIR, 'reflected_points.xyz')
					# 	np.savetxt(reflected_point_cloud_file, curr_reflected, delimiter=' ', fmt='%f')
					# 	print("Saved '{}'.".format(reflected_point_cloud_file))

					# 	# Save point ids.
					# 	out_point_ids_file = os.path.join(LOG_DIR, 'output_point_ids.txt')
					# 	np.savetxt(out_point_ids_file, curr_labels, fmt='%d')
					# 	print("Saved '{}'.".format(out_point_ids_file))

					# 	# Render point_cloud.
					# 	out_points_snapshot_file = os.path.join(LOG_DIR, 'output'+str(i)+'_points')
					# 	render_point_cloud(out_point_cloud_file, out_point_ids_file,
					# 	        out_points_snapshot_file)

					# 	out_points_snapshot_file = os.path.join(LOG_DIR, 'reflected'+str(i)+'_points')
					# 	render_point_cloud(reflected_point_cloud_file, out_point_ids_file,
					# 	        out_points_snapshot_file)
					# exit()

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# scalars["cd_loss"].append(cd_loss)
				# scalars["norm_loss"].append(norm_loss)
				scalars["loss"].append(loss)

				now = datetime.datetime.now()
				log = "{} | Batch [{:04d}/{:04d}] | loss: {:.4f} |"
				log = log.format(now.strftime("%c"), i, len(loader), loss.item())
				print(log)

			if ((epoch+1) %10 == 0):
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
				torch.save(model, fname)

				if summary["loss"] < best_loss:
					best_loss = summary["loss"]
					fname = os.path.join(LOG_DIR, "model.pth")
					print("> Saving model to {}...".format(fname))
					model = {"target_encoder": target_encoder.state_dict(),
							"param_decoder":param_decoder.state_dict(),
							"source_latent_codes":SOURCE_LATENT_CODES,
							"part_latent_codes": SOURCE_PART_LATENT_CODES}
					torch.save(model, fname)
				log += " best: {:.4f} |".format(best_loss)

				fname = os.path.join(LOG_DIR, "train.log")
				with open(fname, "a") as fp:
				    fp.write(log + "\n")

				print(log)
				print("--------------------------------------------------------------------------")

	else:
		np.random.seed(0)
		fname = os.path.join(LOG_DIR, "model.pth")
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

		####After training get visualization results
		num_evaluated = 0
		total_cd_error = 0
		used_sources = []

		for i, batch in enumerate(loader):
			'''
			Per batch output:
				self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index], \
				self.corres_source_label[index]
			'''			
			target_shapes, target_ids, target_labels, _ = batch


			x = [x.to(device, dtype=torch.float) for x in target_shapes]
			x = torch.stack(x)

			##Target Encoder
			target_latent_codes = target_encoder(x)

			source_label_shape = torch.zeros(target_shapes.shape[0])


			if (EVAL_SELECTION == "random"):
				source_labels = get_random_labels(source_label_shape, len(SOURCE_MODEL_INFO))

			elif (EVAL_SELECTION == "minimum"):
				target_latent_codes = target_latent_codes.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1)
				source_labels = source_label_shape.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1)

				## Reshape to (K*batch_size, ...) to feed into the network
				## Source assignments have to be done accordingly
				target_latent_codes = target_latent_codes.view(-1, target_latent_codes.shape[-1])
				source_labels = source_labels.view(-1)

				#Get all labels
				source_labels = get_all_source_labels(source_labels, len(SOURCE_MODEL_INFO))

				##Also overwrite x for chamfer distance					
				x = x.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1,1)
				x = x.view(-1, x.shape[-2], x.shape[-1])


			###Set up source A matrices and default params based on source_labels of the target
			src_mats, src_default_params, src_connectivity_mat = get_source_info(source_labels, SOURCE_MODEL_INFO, MAX_NUM_PARAMS, use_connectivity= USE_CONNECTIVITY)
			
			###Set up source latent codes based on source_labels of the target
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

			cd_loss, _ = chamfer_distance(output_pcs, x, batch_reduction=None)

			if (EVAL_SELECTION == "random"):
				retrieved_idx = source_labels

			elif (EVAL_SELECTION == "minimum"):
				output_pcs = output_pcs.view(len(SOURCE_MODEL_INFO), target_shapes.shape[0], target_shapes.shape[1], target_shapes.shape[2])
				cd_loss = cd_loss.view(len(SOURCE_MODEL_INFO), -1)

				##Selection
				sorted_indices = torch.argsort(cd_loss, dim=0)
				retrieved_idx = sorted_indices[0,:] 
				retrieved_idx_repeated = retrieved_idx.unsqueeze(0).unsqueeze(-1).repeat(1,1,output_pcs.shape[-2]).unsqueeze(-1).repeat(1,1,1,output_pcs.shape[-1])

				output_pcs = torch.gather(output_pcs, 0, retrieved_idx_repeated)
				output_pcs = output_pcs.view(target_shapes.shape[0], target_shapes.shape[1], target_shapes.shape[2])

				cd_loss = torch.gather(cd_loss, 0, retrieved_idx.unsqueeze(0))
				cd_loss = cd_loss.view(-1)				

				retrieved_idx = retrieved_idx.to("cpu")
				retrieved_idx = retrieved_idx.detach().numpy()

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


			for j in range(output_pcs.shape[0]):
				num_evaluated += 1
				total_cd_error += cd_loss[j]

				if not src_ids[j] in used_sources:
					used_sources.append(src_ids[j])

				if (TO_VISU) and j==0:
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



