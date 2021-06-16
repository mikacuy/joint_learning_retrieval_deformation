import os, sys
BASE_DIR = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__))))
import argparse
import h5py
import numpy as np
import csv
import pickle

np.random.seed(0)

def get_model(h5_file, semantic=False, mesh=False, constraint=False):
    with h5py.File(h5_file, 'r') as f:

        box_params = f["box_params"][:]
        orig_ids = f["orig_ids"][:]
        default_param = f["default_param"][:]

        ##Point cloud
        points = f["points"][:]
        point_labels = f["point_labels"][:]
        points_mat = f["points_mat"][:]

        if (semantic):
        	point_semantic = f["point_semantic"][:]

        if (mesh) :
        	vertices = f["vertices"][:]
        	vertices_mat = f["vertices_mat"][:]
        	faces = f["faces"][:]
        	face_labels = f["face_labels"][:]

        if (constraint) :
        	constraint_mat = f["constraint_mat"][:]
        	constraint_proj_mat = f["constraint_proj_mat"][:]

    if constraint and semantic:
    	return box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic, constraint_mat, constraint_proj_mat

    if constraint and mesh:
    	return box_params, orig_ids, default_param, points, point_labels, points_mat, vertices, vertices_mat, faces, face_labels, constraint_mat, constraint_proj_mat

    if (semantic):
    	return box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic

    if (mesh):
    	return box_params, orig_ids, default_param, points, point_labels, points_mat, vertices, vertices_mat, faces, face_labels

    else:
    	return box_params, orig_ids, default_param, points, point_labels, points_mat

def get_all_selected_models_pickle(pickle_file):
	with open(pickle_file, 'rb') as handle:
		data_dict = pickle.load(handle)
		print("Pickle Loaded.")
		return data_dict["sources"], data_dict["train"], data_dict["test"]	

##### For h5 files ######
def save_dataset(fname, pcs, labels, semantics, model_ids):
    cloud = np.stack([pc for pc in pcs])
    cloud_label = np.stack([label for label in labels])
    cloud_semantics = np.stack([semantic for semantic in semantics])
    cloud_id = np.stack([model_id for model_id in model_ids])

    fout = h5py.File(fname)
    fout.create_dataset('data', data=cloud, compression='gzip', dtype='float32')
    fout.create_dataset('label', data=cloud_label, compression='gzip', dtype='int')
    fout.create_dataset('semantic', data=cloud_semantics, compression='gzip', dtype='int')
    fout.create_dataset('model_id', data=cloud_id, compression='gzip', dtype='float32')
    fout.close()

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    semantic = f['semantic'][:]
    model_id = f['model_id'][:]

    return data, label, semantic, model_id
#########################

def get_targets_h5(target_models, datapath, filename):
	total_num_models = len(target_models)

	# Process Targets
	target_points = []
	target_labels = []
	target_semantics = []
	selected_target_model_id = []

	counter = 0
	##To check for invalid model
	all_files = os.listdir(datapath)

	for i in range(len(target_models)):
		model = target_models[i]
		h5_file = str(model)+"_leaves.h5"

		##Check for invalid model id
		if h5_file not in all_files:
			print(h5_file + " does not exist.")
			continue
		box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic = get_model(os.path.join(datapath, h5_file), semantic=True)
		
		target_points.append(points)
		target_labels.append(point_labels)
		target_semantics.append(point_semantic)
		selected_target_model_id.append(model)

		counter += 1
		if (counter % 50 ==0):
			print("Processed "+str(counter)+"/"+str(total_num_models)+" files.")


	target_points = np.array(target_points)
	target_labels = np.array(target_labels)
	target_semantics = np.array(target_semantics)
	selected_target_model_id = np.array(selected_target_model_id)


	save_dataset(filename, target_points, target_labels, target_semantics, selected_target_model_id)
	data, label, semantic, model_id = load_h5(filename)
	print(data.shape)
	print(label.shape)
	print(semantic.shape)
	print(model_id.shape)
	return

def output_to_pickle(output, filename):
	with open(filename, 'wb') as handle:
	    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("Done ", filename)

def collect_sources_and_target_splits(source_data_fol, target_data_fol, num_sources, output_filename, source_txt_file=None):
	
	## If you have a list of pre-selected sources
	if source_txt_file is not None:
		with open(source_txt_file, "r") as f:
			sources = f.readlines()
		source_models = [int(x.strip()) for x in sources]
	else:
		source_models = []

	##Get all targets
	all_target_files = os.listdir(target_data_fol)
	all_target_model_ids = [int(x.strip()[:-10]) for x in all_target_files]

	### Use 10% of targets 
	num_sources = int(0.1 * len(all_target_model_ids))

	for sid in source_models:
		if sid in all_target_model_ids:
			all_target_model_ids.remove(sid)

	####Get more random source models
	all_source_files = os.listdir(source_data_fol)
	all_source_model_ids = [int(x.strip()[:-10]) for x in all_source_files]

	for sid in source_models:
		if sid in all_source_model_ids:
			all_source_model_ids.remove(sid)

	idx = np.arange(len(all_source_model_ids))
	np.random.shuffle(idx)

	for i in range(len(source_models), num_sources):
		source_models.append(all_source_model_ids[idx[i]])

	# Remove from list of targets
	for sid in source_models:
		if sid in all_target_model_ids:
			all_target_model_ids.remove(sid)
	#####

	#### Get train/test split for targets
	all_target_model_ids = np.array(all_target_model_ids)

	##Train and test split
	split_ratio = 0.8
	idx = np.arange(len(all_target_model_ids))
	np.random.shuffle(idx)

	train_split = all_target_model_ids[idx[:int(split_ratio*len(all_target_model_ids))]]
	test_split = all_target_model_ids[idx[int(split_ratio*len(all_target_model_ids)):]]

	##Number of train/test samples
	print("Num training: "+str(len(train_split)))	
	print("Num test: "+str(len(test_split)))
	print("Num sources: "+ str(len(source_models)))

	data_dict = {}
	data_dict["sources"] = source_models
	data_dict["train"] = train_split
	data_dict["test"] = test_split

	output_to_pickle(data_dict, output_filename)

parser = argparse.ArgumentParser()
parser.add_argument('--category', default= "vase", type=str)
parser.add_argument('--num_sources', default= 500, type=int)
parser.add_argument('--dump_dir', default= "generated_datasplits", type=str)
parser.add_argument('--nc', default= False, type=bool)
FLAGS = parser.parse_args()

OBJ_CAT = FLAGS.category
NUM_SOURCES = FLAGS.num_sources
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
NC = FLAGS.nc

###Generate pickle file with sources and train/test split for targets
source_data_fol = os.path.join("data_aabb_constraints_keypoint", OBJ_CAT, "h5")

target_data_fol = os.path.join("data_aabb_all_models", OBJ_CAT, "h5")

if not NC:
	output_filename_pickle = os.path.join(DUMP_DIR, OBJ_CAT+"_"+str(NUM_SOURCES)+".pickle")

	# target_data_fol = os.path.join("data_aabb_all_models_dense", "chair", "h5")
	# output_filename = os.path.join(DUMP_DIR, OBJ_CAT+"_"+str(NUM_SOURCES)+"_dense.pickle")
	collect_sources_and_target_splits(source_data_fol, target_data_fol, NUM_SOURCES, output_filename_pickle, source_txt_file=None)

##For neural cages different pickle protocol
else:
	output_filename_pickle = os.path.join(DUMP_DIR, OBJ_CAT+"_"+str(NUM_SOURCES)+"_nc.pickle")

	# target_data_fol = os.path.join("data_aabb_all_models_dense", "chair", "h5")
	# output_filename = os.path.join(DUMP_DIR, OBJ_CAT+"_"+str(NUM_SOURCES)+"_dense.pickle")
	collect_sources_and_target_splits(source_data_fol, target_data_fol, NUM_SOURCES, output_filename_pickle, source_txt_file=None)

### Get h5
sources, train_targets, test_targets = get_all_selected_models_pickle(output_filename_pickle)

filename = os.path.join(DUMP_DIR, OBJ_CAT+"_"+str(NUM_SOURCES)+"_train.h5")
get_targets_h5(train_targets, target_data_fol, filename)
filename = os.path.join(DUMP_DIR, OBJ_CAT+"_"+str(NUM_SOURCES)+"_test.h5")
get_targets_h5(test_targets, target_data_fol, filename)

# filename = os.path.join(DUMP_DIR, OBJ_CAT+"_"+str(NUM_SOURCES)+"_train_dense.h5")
# get_targets_h5(train_targets, target_data_fol, filename)
# filename = os.path.join(DUMP_DIR, OBJ_CAT+"_"+str(NUM_SOURCES)+"_test_dense.h5")
# get_targets_h5(test_targets, target_data_fol, filename)


