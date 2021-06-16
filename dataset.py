import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import sys
import argparse
import os
BASE_DIR = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__))))
from data_utils import *
from PIL import Image

def normalize_data(pcs):
	for pc in pcs:
		#get furthest point distance then normalize
		d = max(np.sum(np.abs(pc)**2,axis=-1)**(1./2))
		pc /= d

		# pc[:,0]/=max(abs(pc[:,0]))
		# pc[:,1]/=max(abs(pc[:,1]))
		# pc[:,2]/=max(abs(pc[:,2]))

	return pcs

def center_data(pcs):
	for pc in pcs:
		centroid = np.mean(pc, axis=0)
		pc[:,0]-=centroid[0]
		pc[:,1]-=centroid[1]
		pc[:,2]-=centroid[2]
	return pcs

class Scan2CAD_h5(Dataset):

	def __init__(self, filename):
		'''
		Does not store source data in the data loader
		just indicates per target which source it came from

		datapath : folder with the data
		source_models : list of source models
		target_models : list of list
		'''

		f = h5py.File(filename, "r")
		all_target_points = f['data'][:]
		all_target_points = center_data(all_target_points)
		all_target_points = normalize_data(all_target_points)

		num_samples = all_target_points.shape[0]

		self.target_points = all_target_points
		self.target_labels = np.zeros((all_target_points.shape[0], all_target_points.shape[1]))
		self.target_semantics = np.zeros((all_target_points.shape[0], all_target_points.shape[1]))
		self.target_ids = np.arange(all_target_points.shape[0])	

		self.n_samples = num_samples

		print("Number of targets: "+str(self.n_samples))


	def __getitem__(self, index):

		return self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index]

	def __len__(self):
	    return self.n_samples

class Classification_h5(Dataset):

	def __init__(self, filename):
		'''
		Does not store source data in the data loader
		just indicates per target which source it came from

		datapath : folder with the data
		source_models : list of source models
		target_models : list of list
		'''
		all_target_points, all_target_labels, all_target_semantics, all_target_model_id, all_class_labels = load_h5_classification(filename)
		self.target_points = all_target_points
		self.target_labels = all_target_labels
		self.target_semantics = all_target_semantics
		self.target_ids = all_target_model_id	
		self.class_labels = all_class_labels	

		self.n_samples = all_target_points.shape[0]

		print("Number of targets: "+str(self.n_samples))


	def __getitem__(self, index):

		return self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index], self.class_labels[index]

	def __len__(self):
	    return self.n_samples


class StructureNetDataset_h5(Dataset):

	def __init__(self, filename):
		'''
		Does not store source data in the data loader
		just indicates per target which source it came from

		datapath : folder with the data
		source_models : list of source models
		target_models : list of list
		'''
		all_target_points, all_target_labels, all_target_semantics, all_target_model_id = load_h5(filename)
		self.target_points = all_target_points
		self.target_labels = all_target_labels
		self.target_semantics = all_target_semantics
		self.target_ids = all_target_model_id	

		self.n_samples = all_target_points.shape[0]

		print("Number of targets: "+str(self.n_samples))


	def __getitem__(self, index):

		return self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index]

	def __len__(self):
	    return self.n_samples

class StructureNetDataset_h5_images(Dataset):

	def __init__(self, filename, image_folder, is_train=True):
		'''
		Does not store source data in the data loader
		just indicates per target which source it came from

		datapath : folder with the data
		source_models : list of source models
		target_models : list of list
		'''
		all_target_points, all_target_labels, all_target_semantics, all_target_model_id = load_h5(filename)
		self.target_points = all_target_points
		self.target_labels = all_target_labels
		self.target_semantics = all_target_semantics
		self.target_ids = all_target_model_id	

		self.n_samples = all_target_points.shape[0]

		## For loading images
		self.image_folder = image_folder
		self.img_size = 224

		print("Number of targets: "+str(self.n_samples))

		self.is_train = is_train


	def __getitem__(self, index):

		target = self.target_ids[index]

		if (self.is_train):
			random_view = torch.randint(24, size=(1,)).to("cpu").detach().numpy()
			img_filename = os.path.join(self.image_folder, str(int(target)), "view-"+str(int(random_view[0])).zfill(2), "shape-rgb.png")
		else:
			random_view = np.array([17])
			img_filename = os.path.join(self.image_folder, str(int(target)), "view-"+str(int(random_view[0])).zfill(2), "shape-rgb.png")
		# img = Image.open(img_filename)
		# img = np.asarray(img)		

		with Image.open(img_filename) as fimg:
		    out = np.array(fimg, dtype=np.float32) / 255.0
		white_img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32)
		mask = np.tile(out[:, :, 3:4], [1, 1, 3])

		out = out[:, :, :3] * mask + white_img * (1 - mask)
		out = torch.from_numpy(out).permute(2, 0, 1)

		if (self.is_train):
			return out, self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index]
		
		# To return the random view selected
		else:
			return out, self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index], random_view

	def __len__(self):
	    return self.n_samples

class StructureNetDataset_Triplet_images(Dataset):

	def __init__(self, filename, indices_dict, num_pos, num_neg, image_folder):
		'''
		Does not store source data in the data loader
		just indicates per target which source it came from

		datapath : folder with the data
		source_models : list of source models
		target_models : list of list
		'''
		all_target_points, all_target_labels, all_target_semantics, all_target_model_id = load_h5(filename)
		positive_idx = indices_dict["positives"]
		negative_idx = indices_dict["negatives"]

		### Remove entry if no positive index##
		##TODO
		#######

		self.target_points = all_target_points
		self.target_labels = all_target_labels
		self.target_semantics = all_target_semantics
		self.target_ids = all_target_model_id	

		self.n_samples = all_target_points.shape[0]

		print("Number of targets: "+str(self.n_samples))

		self.num_pos = num_pos
		self.num_neg = num_neg

		self.positives_idx = positive_idx
		self.negatives_idx = negative_idx

		## For loading images
		self.image_folder = image_folder
		self.img_size = 224

	def __getitem__(self, index):

		pos_candidates = self.positives_idx[index]
		neg_candidates = self.negatives_idx[index]

		pos_candidates = torch.from_numpy(np.array(pos_candidates))
		neg_candidates = torch.from_numpy(np.array(neg_candidates))

		positive_idx_selected = pos_candidates[torch.randint(len(pos_candidates), (self.num_pos,))]
		negative_idx_selected = neg_candidates[torch.randint(len(neg_candidates), (self.num_neg,))]    

		## For images
		target = self.target_ids[index]
		random_view = torch.randint(24, size=(1,)).to("cpu").detach().numpy()
		img_filename = os.path.join(self.image_folder, str(int(target)), "view-"+str(int(random_view[0])).zfill(2), "shape-rgb.png")	

		with Image.open(img_filename) as fimg:
		    out = np.array(fimg, dtype=np.float32) / 255.0
		white_img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32)
		mask = np.tile(out[:, :, 3:4], [1, 1, 3])

		out = out[:, :, :3] * mask + white_img * (1 - mask)
		out = torch.from_numpy(out).permute(2, 0, 1)

		return out, self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index], \
				positive_idx_selected, negative_idx_selected

	def __len__(self):
	    return self.n_samples

class MDS_images(Dataset):

	def __init__(self, model_ids, mds_latent_vecs, image_folder, is_train=True):

		self.model_ids = model_ids	
		self.mds_latent_vecs = mds_latent_vecs	

		self.n_samples = model_ids.shape[0]

		## For loading images
		self.image_folder = image_folder
		self.img_size = 224

		print("Number of targets: "+str(self.n_samples))

		self.is_train = is_train


	def __getitem__(self, index):

		model_id = self.model_ids[index]
		random_view = torch.randint(24, size=(1,)).to("cpu").detach().numpy()

		img_filename = os.path.join(self.image_folder, str(int(model_id)), "view-"+str(int(random_view[0])).zfill(2), "shape-rgb.png")
		# img = Image.open(img_filename)
		# img = np.asarray(img)		

		with Image.open(img_filename) as fimg:
		    out = np.array(fimg, dtype=np.float32) / 255.0
		white_img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32)
		mask = np.tile(out[:, :, 3:4], [1, 1, 3])

		out = out[:, :, :3] * mask + white_img * (1 - mask)
		out = torch.from_numpy(out).permute(2, 0, 1)


		if (self.is_train):
			return out, self.mds_latent_vecs[index]
		
		# To return the random view selected
		else:
			return out, self.mds_latent_vecs[index], random_view

	def __len__(self):
	    return self.n_samples

class StructureNetDataset_DeformCandidates(Dataset):

	def __init__(self, filename, candidates_dict):
		'''
		Does not store source data in the data loader
		just indicates per target which source it came from

		datapath : folder with the data
		source_models : list of source models
		target_models : list of list
		'''
		all_target_points, all_target_labels, all_target_semantics, all_target_model_id = load_h5(filename)
		self.target_points = all_target_points
		self.target_labels = all_target_labels
		self.target_semantics = all_target_semantics
		self.target_ids = all_target_model_id	

		self.n_samples = all_target_points.shape[0]

		print("Number of targets: "+str(self.n_samples))

		self.candidates_idx = candidates_dict["candidates"]

	def __getitem__(self, index):

		candidates_idx = self.candidates_idx[index]

		candidates_idx = torch.from_numpy(candidates_idx)
		perm = torch.randperm(candidates_idx.size(0))
		idx = perm[0]
		candidates_idx_selected = candidates_idx[idx]

		# idx = np.arange(len(candidates_idx))
		# np.random.shuffle(idx)
		# candidates_idx_selected = candidates_idx[idx[0]]
		# candidates_idx_selected = np.squeeze(np.array(candidates_idx_selected))	

		return self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index], \
				candidates_idx_selected

	def __len__(self):
	    return self.n_samples

class StructureNetDataset_Triplet_with_Cost(Dataset):

	def __init__(self, filename, indices_dict, num_pos, num_neg):
		'''
		Does not store source data in the data loader
		just indicates per target which source it came from

		datapath : folder with the data
		source_models : list of source models
		target_models : list of list
		'''
		all_target_points, all_target_labels, all_target_semantics, all_target_model_id = load_h5(filename)
		positive_idx = indices_dict["positives"]
		negative_idx = indices_dict["negatives"]

		positive_costs = indices_dict["positive_costs"]
		negative_costs = indices_dict["negative_costs"]

		### Remove entry if no positive index##
		##TODO
		#######

		self.target_points = all_target_points
		self.target_labels = all_target_labels
		self.target_semantics = all_target_semantics
		self.target_ids = all_target_model_id	

		self.n_samples = all_target_points.shape[0]

		print("Number of targets: "+str(self.n_samples))

		self.num_pos = num_pos
		self.num_neg = num_neg

		self.positives_idx = positive_idx
		self.negatives_idx = negative_idx
		self.positive_costs = positive_costs
		self.negative_costs = negative_costs

	def __getitem__(self, index):

		pos_candidates = self.positives_idx[index]
		neg_candidates = self.negatives_idx[index]
		pos_candidates = torch.from_numpy(np.array(pos_candidates))
		neg_candidates = torch.from_numpy(np.array(neg_candidates))

		pos_candidates_costs = self.positive_costs[index]
		neg_candidates_costs = self.negative_costs[index]
		pos_candidates_costs = torch.from_numpy(np.array(pos_candidates_costs))
		neg_candidates_costs = torch.from_numpy(np.array(neg_candidates_costs))

		pos_rnd_idx = torch.randint(len(pos_candidates), (self.num_pos,))
		neg_rnd_idx = torch.randint(len(neg_candidates), (self.num_neg,))

		positive_idx_selected = pos_candidates[pos_rnd_idx]
		negative_idx_selected = neg_candidates[neg_rnd_idx]	    

		positive_costs_selected = pos_candidates_costs[pos_rnd_idx]
		negative_costs_selected = neg_candidates_costs[neg_rnd_idx]	

		return self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index], \
				positive_idx_selected, negative_idx_selected, positive_costs_selected, negative_costs_selected

	def __len__(self):
	    return self.n_samples

class StructureNetDataset_Triplet(Dataset):

	def __init__(self, filename, indices_dict, num_pos, num_neg):
		'''
		Does not store source data in the data loader
		just indicates per target which source it came from

		datapath : folder with the data
		source_models : list of source models
		target_models : list of list
		'''
		all_target_points, all_target_labels, all_target_semantics, all_target_model_id = load_h5(filename)
		positive_idx = indices_dict["positives"]
		negative_idx = indices_dict["negatives"]

		### Remove entry if no positive index##
		##TODO
		#######

		self.target_points = all_target_points
		self.target_labels = all_target_labels
		self.target_semantics = all_target_semantics
		self.target_ids = all_target_model_id	

		self.n_samples = all_target_points.shape[0]

		print("Number of targets: "+str(self.n_samples))

		self.num_pos = num_pos
		self.num_neg = num_neg

		self.positives_idx = positive_idx
		self.negatives_idx = negative_idx

	def __getitem__(self, index):

		pos_candidates = self.positives_idx[index]
		neg_candidates = self.negatives_idx[index]

		# print(pos_candidates)
		# print(neg_candidates)

		pos_candidates = torch.from_numpy(np.array(pos_candidates))
		neg_candidates = torch.from_numpy(np.array(neg_candidates))

		positive_idx_selected = pos_candidates[torch.randint(len(pos_candidates), (self.num_pos,))]
		negative_idx_selected = neg_candidates[torch.randint(len(neg_candidates), (self.num_neg,))]

		# positive_idx_selected = np.random.choice(pos_candidates, self.num_pos, replace=True)
		# negative_idx_selected = np.random.choice(neg_candidates, self.num_neg, replace=True)

		# positive_idx_selected = np.array(positive_idx_selected)	    
		# negative_idx_selected = np.array(negative_idx_selected)	    

		return self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index], \
				positive_idx_selected, negative_idx_selected

	def __len__(self):
	    return self.n_samples


class StructureNetDataset_Multi_MemEff(Dataset):

	def __init__(self, datapath, source_models, target_models):
		'''
		Does not store source data in the data loader
		just indicates per target which source it came from

		datapath : folder with the data
		source_models : list of source models
		target_models : list of list
		'''
		total_num_models = 0
		for i in range(len(source_models)):
			total_num_models += len(target_models[i])
		# print(total_num_models)

		# Process Targets

		target_points = []
		target_labels = []
		target_semantics = []
		selected_target_model_id = []
		corres_source_label = []

		counter = 0
		##To check for invalid model
		all_files = os.listdir(datapath)

		for i in range(len(source_models)):
			curr_target_models = target_models[i]

			curr_target_points = []
			curr_target_labels = []
			curr_target_semantics = []
			curr_selected_target_model_id = []
			curr_corres_source_label = []

			for j in range(len(curr_target_models)):
				model = curr_target_models[j]
				h5_file = str(model)+"_leaves.h5"

				##Check for invalid model id
				if h5_file not in all_files:
					print(h5_file + " does not exist.")
					continue
				box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic = get_model(os.path.join(datapath, h5_file), semantic=True)
				curr_target_points.append(points)
				curr_target_labels.append(point_labels)
				curr_target_semantics.append(point_semantic)
				curr_selected_target_model_id.append(model)
				curr_corres_source_label.append(i)

				counter += 1
				if (counter % 50 ==0):
					print("Processed "+str(counter)+"/"+str(total_num_models)+" files.")

			curr_target_points = np.array(curr_target_points)
			curr_target_labels = np.array(curr_target_labels)
			curr_target_semantics = np.array(curr_target_semantics)
			curr_selected_target_model_id = np.array(curr_selected_target_model_id)
			curr_corres_source_label = np.array(curr_corres_source_label)

			# print(curr_target_points.shape)
			# print(curr_target_labels.shape)
			# print(curr_target_semantics.shape)
			# print(curr_selected_target_model_id.shape)
			# print(curr_corres_source_label.shape)

			target_points.append(curr_target_points)
			target_labels.append(curr_target_labels)
			target_semantics.append(curr_target_semantics)
			selected_target_model_id.append(curr_selected_target_model_id)
			corres_source_label.append(curr_corres_source_label)

		#### Concatenate target arrays
		all_target_points = target_points[0]
		all_target_labels = target_labels[0]
		all_target_semantics = target_semantics[0]
		all_target_model_id = selected_target_model_id[0]
		all_corres_source_label = corres_source_label[0]

		for i in range(1, len(source_models)):
			all_target_points = np.concatenate((all_target_points, target_points[i]), axis=0)
			all_target_labels = np.concatenate((all_target_labels, target_labels[i]), axis=0)
			all_target_semantics = np.concatenate((all_target_semantics, target_semantics[i]), axis=0)
			all_target_model_id = np.concatenate((all_target_model_id, selected_target_model_id[i]), axis=0)
			all_corres_source_label = np.concatenate((all_corres_source_label, corres_source_label[i]), axis=0)

		# print(all_target_points.shape)
		# print(all_target_labels.shape)
		# print(all_target_semantics.shape)
		# print(all_target_model_id.shape)
		# print(all_corres_source_label.shape)

		self.target_points = all_target_points
		self.target_labels = all_target_labels
		self.target_semantics = all_target_semantics
		self.target_ids = all_target_model_id
		self.corres_source_label = all_corres_source_label		

		self.n_samples = all_target_points.shape[0]

		print("Number of targets: "+str(self.n_samples))

	def __getitem__(self, index):
		return self.target_points[index], self.target_ids[index], self.target_labels[index], self.target_semantics[index], \
				self.corres_source_label[index]

	def __len__(self):
	    return self.n_samples


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--category', default= "chair", type=str)
	parser.add_argument('--data_dir', default= "data_aabb_labels", type=str)

	FLAGS = parser.parse_args()
	DATA_DIR = FLAGS.data_dir
	OBJ_CAT = FLAGS.category

	# data_fol = os.path.join(BASE_DIR, DATA_DIR, OBJ_CAT, "h5")
	# dataset = StructureNetDataset(data_fol)
	# # dataset = StructureNetDataset(data_fol, models=[1301, 2369, 2198, 1441])
	# print(len(dataset))
	# print(dataset[0])

	### Handling multiple sources
	sources, targets = get_all_selected_models("chairs_2cluster.csv")
	data_fol = os.path.join(BASE_DIR, DATA_DIR, OBJ_CAT, "h5")
	dataset = StructureNetDataset_Multi(data_fol, sources, targets)
	print(len(dataset))
	print(dataset[0])




