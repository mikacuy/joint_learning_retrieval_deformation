import os, sys
BASE_DIR = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__))))
import argparse
import h5py
import numpy as np
import csv
import pickle

import torch

from PIL import Image
import trimesh

np.random.seed(0)

# IMAGE_BASE_DIR = "/orion/downloads/partnet_dataset/partnet_rgb_masks_chair/"
# IMAGE_BASE_DIR = "/orion/downloads/partnet_dataset/partnet_rgb_masks_table/"
global IMAGE_BASE_DIR 
IMAGE_BASE_DIR = "/orion/downloads/partnet_dataset/partnet_rgb_masks_storagefurniture/"

def set_img_basedir(obj_cat):
	global IMAGE_BASE_DIR
	IMAGE_BASE_DIR = "/orion/downloads/partnet_dataset/partnet_rgb_masks_"+obj_cat+"/"

def get_model(h5_file, semantic=False, mesh=False, constraint=False):
    with h5py.File(h5_file, 'r') as f:
        # print(f.keys())

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

def get_all_selected_models_pickle(pickle_file, all_models=False):
	with open(pickle_file, 'rb') as handle:
		data_dict = pickle.load(handle)
		print("Pickle Loaded.")
		if not all_models:
			return data_dict["sources"], data_dict["train"], data_dict["test"]	
		else:
			return data_dict["sources"], data_dict["source_cat"], data_dict["train"], data_dict["test"]	

##### For h5 files ######
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    semantic = f['semantic'][:]
    model_id = f['model_id'][:]

    return data, label, semantic, model_id

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

def load_h5_classification(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    semantic = f['semantic'][:]
    model_id = f['model_id'][:]
    class_labels = f['class_labels'][:]

    return data, label, semantic, model_id, class_labels

def save_dataset_classification(fname, pcs, labels, semantics, model_ids, class_labels):
    cloud = np.stack([pc for pc in pcs])
    cloud_label = np.stack([label for label in labels])
    cloud_semantics = np.stack([semantic for semantic in semantics])
    cloud_id = np.stack([model_id for model_id in model_ids])
    c_class_labels = np.stack([label for label in class_labels])

    fout = h5py.File(fname)
    fout.create_dataset('data', data=cloud, compression='gzip', dtype='float32')
    fout.create_dataset('label', data=cloud_label, compression='gzip', dtype='int')
    fout.create_dataset('semantic', data=cloud_semantics, compression='gzip', dtype='int')
    fout.create_dataset('model_id', data=cloud_id, compression='gzip', dtype='float32')
    fout.create_dataset('class_labels', data=c_class_labels, compression='gzip', dtype='int')
    fout.close()

###############################


def render_point_cloud(point_cloud_file, point_labels_file, snapshot_file):
	g_renderer = '/orion/u/mhsung/app/primitive-fitting/build/OSMesaRenderer'
	g_azimuth_deg = -70
	g_elevation_deg = 20
	g_theta_deg = 0

	cmd = g_renderer + ' \\\n'
	cmd += ' --point_cloud=' + point_cloud_file + ' \\\n'
	cmd += ' --point_labels=' + point_labels_file + ' \\\n'
	cmd += ' --snapshot=' + snapshot_file + ' \\\n'
	cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
	cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
	cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
	cmd += ' >/dev/null 2>&1'
	os.system(cmd)
	# print(cmd)
	snapshot_file += '.png'
	print("Saved '{}'.".format(snapshot_file))

def render_mesh(mesh_file, face_labels_file, snapshot_file):
	g_renderer = '/orion/u/mhsung/app/primitive-fitting/build/OSMesaRenderer'
	g_azimuth_deg = -70
	g_elevation_deg = 20
	g_theta_deg = 0

	cmd = g_renderer + ' \\\n'
	cmd += ' --mesh=' + mesh_file + ' \\\n'
	cmd += ' --face_labels=' + face_labels_file + ' \\\n'
	cmd += ' --snapshot=' + snapshot_file + ' \\\n'
	cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
	cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
	cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
	cmd += ' >/dev/null 2>&1'
	os.system(cmd)
	# print(cmd)
	
	snapshot_file += '.png'
	print("Saved '{}'.".format(snapshot_file))

def output_visualization(output_pc, src_points, target_points, src_point_labels, target_point_labels, src_id, target_id, output_fol, chamfer_cost = None, method = None):
	temp_fol = os.path.join(output_fol, "tmp")

	### Source
	# Save point cloud.
	out_point_cloud_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) + '_points.xyz')
	np.savetxt(out_point_cloud_file, src_points, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(out_point_cloud_file))

	# Save point ids.
	out_point_ids_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) + '_point_ids.txt')
	np.savetxt(out_point_ids_file, src_point_labels, fmt='%d')
	print("Saved '{}'.".format(out_point_ids_file))

	# Render point_cloud.
	src_points_snapshot_file = os.path.join(temp_fol, str(src_id)+'_points')
	render_point_cloud(out_point_cloud_file, out_point_ids_file,
	        src_points_snapshot_file)

	### Target
	# Save point cloud.
	out_point_cloud_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) + '_points.xyz')
	np.savetxt(out_point_cloud_file, target_points, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(out_point_cloud_file))

	# Save point ids.
	out_point_ids_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) + '_point_ids.txt')
	np.savetxt(out_point_ids_file, target_point_labels, fmt='%d')
	print("Saved '{}'.".format(out_point_ids_file))

	# Render point_cloud.
	target_points_snapshot_file = os.path.join(temp_fol, str(target_id)+'_points')
	render_point_cloud(out_point_cloud_file, out_point_ids_file,
	        target_points_snapshot_file)

	### Output
	# Save point cloud.
	out_point_cloud_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) + '_points.xyz')
	np.savetxt(out_point_cloud_file, output_pc, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(out_point_cloud_file))

	# Save point ids.
	out_point_ids_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) + '_point_ids.txt')
	np.savetxt(out_point_ids_file, src_point_labels, fmt='%d')
	print("Saved '{}'.".format(out_point_ids_file))

	# Render point_cloud.
	out_points_snapshot_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) +'_points')
	render_point_cloud(out_point_cloud_file, out_point_ids_file,
	        out_points_snapshot_file)


	#Output to a single image
	height = 1080
	width = 1920
	new_im = Image.new('RGBA', (width*3, height))
	im1 = Image.open(src_points_snapshot_file+".png")
	im2 = Image.open(target_points_snapshot_file+".png")
	im3 = Image.open(out_points_snapshot_file+".png")
	images = [im1, im2, im3]
	x_offset = 0
	for im in images:
	    new_im.paste(im, (x_offset,0))
	    x_offset += width

	if (chamfer_cost == None):
		output_image_filename = os.path.join(output_fol, str(src_id) + "_" + str(target_id) +'_deform.png')
	else:
		output_image_filename = os.path.join(output_fol, str(src_id) + "_" + str(target_id) +'_' + method + '_' + str(float(chamfer_cost)) +".png")


	new_im.save(output_image_filename)    
	print("Saved '{}'.".format(output_image_filename))

	#### Clean up files
	# os.system("rm " + out_point_cloud_file)
	# os.system("rm " + out_point_ids_file)
	# os.system("rm " + src_points_snapshot_file+".png")
	# os.system("rm " + target_points_snapshot_file+".png")
	# os.system("rm " + out_points_snapshot_file+".png")

def output_visualization_mesh(output_vertices, src_vertices, src_faces, target_points, src_face_labels, target_point_labels, src_id, target_id, output_fol, chamfer_cost = None, method = None):
	temp_fol = os.path.join(output_fol, "tmp")

	### Source
	# Save mesh.
	out_mesh_file = os.path.join(temp_fol, str(src_id) +'_mesh.obj')
	mesh = trimesh.Trimesh(vertices=src_vertices, faces=src_faces)     
	mesh.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
	print("Saved '{}'.".format(out_mesh_file))

	# Save vertex ids.
	out_vertex_ids_file = os.path.join(temp_fol, str(src_id) + '_vertex_ids.txt')
	np.savetxt(out_vertex_ids_file, src_vertices, fmt='%d')
	print("Saved '{}'.".format(out_vertex_ids_file))

	# Save face ids.
	out_face_ids_file = os.path.join(temp_fol, str(src_id) + '_face_ids.txt')
	np.savetxt(out_face_ids_file, src_face_labels, fmt='%d')
	print("Saved '{}'.".format(out_face_ids_file))

	# Render mesh.
	src_mesh_snapshot_file = os.path.join(temp_fol, str(src_id)+'_mesh')
	render_mesh(out_mesh_file, out_face_ids_file, src_mesh_snapshot_file)


	### Target
	# Save point cloud.
	out_point_cloud_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) + '_points.xyz')
	np.savetxt(out_point_cloud_file, target_points, delimiter=' ', fmt='%f')
	print("Saved '{}'.".format(out_point_cloud_file))

	# Save point ids.
	out_point_ids_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) + '_point_ids.txt')
	np.savetxt(out_point_ids_file, target_point_labels, fmt='%d')
	print("Saved '{}'.".format(out_point_ids_file))

	# Render point_cloud.
	target_points_snapshot_file = os.path.join(temp_fol, str(target_id)+'_points')
	render_point_cloud(out_point_cloud_file, out_point_ids_file,
	        target_points_snapshot_file)

	### Output
	# Save mesh.
	out_mesh_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) +'_mesh.obj')
	mesh = trimesh.Trimesh(vertices=output_vertices, faces=src_faces)     
	mesh.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
	print("Saved '{}'.".format(out_mesh_file))

	# Save vertex ids.
	out_vertex_ids_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) +'_vertex_ids.txt')
	np.savetxt(out_vertex_ids_file, output_vertices, fmt='%d')
	print("Saved '{}'.".format(out_vertex_ids_file))

	# Render mesh.
	output_mesh_snapshot_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id)+'_mesh')
	render_mesh(out_mesh_file, out_face_ids_file, output_mesh_snapshot_file)

	#Output to a single image
	height = 1080
	width = 1920
	new_im = Image.new('RGBA', (width*3, height))
	im1 = Image.open(src_mesh_snapshot_file+".png")
	im2 = Image.open(target_points_snapshot_file+".png")
	im3 = Image.open(output_mesh_snapshot_file+".png")
	images = [im1, im2, im3]
	x_offset = 0
	for im in images:
	    new_im.paste(im, (x_offset,0))
	    x_offset += width

	if (chamfer_cost == None):
		output_image_filename = os.path.join(output_fol, str(src_id) + "_" + str(target_id) +'_deform.png')
	else:
		output_image_filename = os.path.join(output_fol, str(src_id) + "_" + str(target_id) +'_' + method + '_' + str(float(chamfer_cost)) +".png")

	new_im.save(output_image_filename)    
	print("Saved '{}'.".format(output_image_filename))

def output_visualization_mesh_images(output_vertices, src_vertices, src_faces, target_points, src_face_labels, target_point_labels, src_id, target_id, output_fol, random_view, obj_cat, chamfer_cost = None, method = None):
	temp_fol = os.path.join(output_fol, "tmp")

	### Source
	# Save mesh.
	out_mesh_file = os.path.join(temp_fol, str(src_id) +'_mesh.obj')
	mesh = trimesh.Trimesh(vertices=src_vertices, faces=src_faces)     
	mesh.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
	print("Saved '{}'.".format(out_mesh_file))

	# Save vertex ids.
	out_vertex_ids_file = os.path.join(temp_fol, str(src_id) + '_vertex_ids.txt')
	np.savetxt(out_vertex_ids_file, src_vertices, fmt='%d')
	print("Saved '{}'.".format(out_vertex_ids_file))

	# Save face ids.
	out_face_ids_file = os.path.join(temp_fol, str(src_id) + '_face_ids.txt')
	np.savetxt(out_face_ids_file, src_face_labels, fmt='%d')
	print("Saved '{}'.".format(out_face_ids_file))

	# Render mesh.
	src_mesh_snapshot_file = os.path.join(temp_fol, str(src_id)+'_mesh')
	render_mesh(out_mesh_file, out_face_ids_file, src_mesh_snapshot_file)


	### Target
	# Load the image view
	height = 1080
	width = 1920
	IMAGE_BASE_DIR = "/orion/downloads/partnet_dataset/partnet_rgb_masks_"+obj_cat+"/"	
	img_filename = os.path.join(IMAGE_BASE_DIR, str(int(target_id)), "view-"+str(int(random_view[0])).zfill(2), "shape-rgb.png")

	img = Image.open(img_filename)
	old_size = img.size
	ratio = float(height)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])
	img_resized = img.resize(new_size, Image.ANTIALIAS)
	padded_img = Image.new("RGBA", (width, height))
	padded_img.paste(img_resized, ((width-new_size[0])//2, (height-new_size[1])//2))	

	### Output
	# Save mesh.
	out_mesh_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) +'_mesh.obj')
	mesh = trimesh.Trimesh(vertices=output_vertices, faces=src_faces)     
	mesh.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
	print("Saved '{}'.".format(out_mesh_file))

	# Save vertex ids.
	out_vertex_ids_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id) +'_vertex_ids.txt')
	np.savetxt(out_vertex_ids_file, output_vertices, fmt='%d')
	print("Saved '{}'.".format(out_vertex_ids_file))

	# Render mesh.
	output_mesh_snapshot_file = os.path.join(temp_fol, str(src_id) + "_" + str(target_id)+'_mesh')
	render_mesh(out_mesh_file, out_face_ids_file, output_mesh_snapshot_file)

	#Output to a single image
	new_im = Image.new('RGBA', (width*3, height))
	im1 = Image.open(src_mesh_snapshot_file+".png")
	im2 = padded_img
	im3 = Image.open(output_mesh_snapshot_file+".png")
	images = [im1, im2, im3]
	x_offset = 0
	for im in images:
	    new_im.paste(im, (x_offset,0))
	    x_offset += width

	if (chamfer_cost == None):
		output_image_filename = os.path.join(output_fol, str(src_id) + "_" + str(target_id) +'_deform.png')
	else:
		output_image_filename = os.path.join(output_fol, str(src_id) + "_" + str(target_id) +'_' + method + '_' + str(float(chamfer_cost)) +".png")

	new_im.save(output_image_filename)    
	print("Saved '{}'.".format(output_image_filename))

def output_visualization_mesh_naturalimages(output_vertices, src_vertices, src_faces, src_face_labels, src_id, counter, output_fol):
	temp_fol = os.path.join(output_fol, "tmp")

	### Source
	# Save mesh.
	out_mesh_file = os.path.join(temp_fol, str(src_id) +'_mesh.obj')
	mesh = trimesh.Trimesh(vertices=src_vertices, faces=src_faces)     
	mesh.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
	print("Saved '{}'.".format(out_mesh_file))

	# Save vertex ids.
	out_vertex_ids_file = os.path.join(temp_fol, str(src_id) + '_vertex_ids.txt')
	np.savetxt(out_vertex_ids_file, src_vertices, fmt='%d')
	print("Saved '{}'.".format(out_vertex_ids_file))

	# Save face ids.
	out_face_ids_file = os.path.join(temp_fol, str(src_id) + '_face_ids.txt')
	np.savetxt(out_face_ids_file, src_face_labels, fmt='%d')
	print("Saved '{}'.".format(out_face_ids_file))

	# Render mesh.
	src_mesh_snapshot_file = os.path.join(temp_fol, str(src_id)+'_mesh')
	render_mesh(out_mesh_file, out_face_ids_file, src_mesh_snapshot_file)


	### Target
	# Load the image view
	height = 1080
	width = 1920
	IMAGE_BASE_DIR = "dump_natural_images/process_input_images_cabinet/"	
	img_filename = os.path.join(IMAGE_BASE_DIR, str(counter)+".png")

	img = Image.open(img_filename)
	old_size = img.size
	ratio = float(height)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])
	img_resized = img.resize(new_size, Image.ANTIALIAS)
	padded_img = Image.new("RGBA", (width, height))
	padded_img.paste(img_resized, ((width-new_size[0])//2, (height-new_size[1])//2))	

	### Output
	# Save mesh.
	out_mesh_file = os.path.join(temp_fol, str(src_id) + "_" + str(counter) +'_mesh.obj')
	mesh = trimesh.Trimesh(vertices=output_vertices, faces=src_faces)     
	mesh.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
	print("Saved '{}'.".format(out_mesh_file))

	# Save vertex ids.
	out_vertex_ids_file = os.path.join(temp_fol, str(src_id) + "_" + str(counter) +'_vertex_ids.txt')
	np.savetxt(out_vertex_ids_file, output_vertices, fmt='%d')
	print("Saved '{}'.".format(out_vertex_ids_file))

	# Render mesh.
	output_mesh_snapshot_file = os.path.join(temp_fol, str(src_id) + "_" + str(counter)+'_mesh')
	render_mesh(out_mesh_file, out_face_ids_file, output_mesh_snapshot_file)

	#Output to a single image
	new_im = Image.new('RGBA', (width*3, height))
	im1 = Image.open(src_mesh_snapshot_file+".png")
	im2 = padded_img
	im3 = Image.open(output_mesh_snapshot_file+".png")
	images = [im1, im2, im3]
	x_offset = 0
	for im in images:
	    new_im.paste(im, (x_offset,0))
	    x_offset += width

	output_image_filename = os.path.join(output_fol, str(src_id) + "_" + str(counter) +'_deform.png')

	new_im.save(output_image_filename)    
	print("Saved '{}'.".format(output_image_filename))

def output_visualization_mesh_images_mds(src_vertices, src_faces, target_points, src_face_labels, target_point_labels, src_id, target_id, output_fol, random_view, obj_cat, chamfer_cost = None, method = None):
	temp_fol = os.path.join(output_fol, "tmp")

	### Source
	# Save mesh.
	out_mesh_file = os.path.join(temp_fol, str(src_id) +'_mesh.obj')
	mesh = trimesh.Trimesh(vertices=src_vertices, faces=src_faces)     
	mesh.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
	print("Saved '{}'.".format(out_mesh_file))

	# Save vertex ids.
	out_vertex_ids_file = os.path.join(temp_fol, str(src_id) + '_vertex_ids.txt')
	np.savetxt(out_vertex_ids_file, src_vertices, fmt='%d')
	print("Saved '{}'.".format(out_vertex_ids_file))

	# Save face ids.
	out_face_ids_file = os.path.join(temp_fol, str(src_id) + '_face_ids.txt')
	np.savetxt(out_face_ids_file, src_face_labels, fmt='%d')
	print("Saved '{}'.".format(out_face_ids_file))

	# Render mesh.
	src_mesh_snapshot_file = os.path.join(temp_fol, str(src_id)+'_mesh')
	render_mesh(out_mesh_file, out_face_ids_file, src_mesh_snapshot_file)


	### Target
	# Load the image view
	height = 1080
	width = 1920
	IMAGE_BASE_DIR = "/orion/downloads/partnet_dataset/partnet_rgb_masks_"+obj_cat+"/"	
	img_filename = os.path.join(IMAGE_BASE_DIR, str(int(target_id)), "view-"+str(int(random_view[0])).zfill(2), "shape-rgb.png")

	img = Image.open(img_filename)
	old_size = img.size
	ratio = float(height)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])
	img_resized = img.resize(new_size, Image.ANTIALIAS)
	padded_img = Image.new("RGBA", (width, height))
	padded_img.paste(img_resized, ((width-new_size[0])//2, (height-new_size[1])//2))	


	#Output to a single image
	new_im = Image.new('RGBA', (width*2, height))
	im1 = Image.open(src_mesh_snapshot_file+".png")
	im2 = padded_img

	images = [im1, im2]
	x_offset = 0
	for im in images:
	    new_im.paste(im, (x_offset,0))
	    x_offset += width

	if (chamfer_cost == None):
		output_image_filename = os.path.join(output_fol, str(src_id) + "_" + str(target_id) +'_deform.png')
	else:
		output_image_filename = os.path.join(output_fol, str(src_id) + "_" + str(target_id) +'_' + method + '_' + str(float(chamfer_cost)) +".png")


def get_shape_numpy(A, param, src_default_param=None, weight=1.0, connectivity_mat = None):
	### A is the parametric model of the shape
	### assumes that the shape of A and param agree
	param = np.multiply(param, weight)

	if (src_default_param is None):
		param = param
	else:
		param = param + src_default_param

	# For connectivity
	if connectivity_mat is None:
		param = param

	else:
		# print("Using connectivity constraint for mesh generation.")
		param = np.matmul(connectivity_mat, param)

	pc = np.reshape(np.matmul(A, param), (-1, 3), order='C')

	return pc	

def get_shape_numpy_direct(A, param, connectivity_mat = None):
	### A is the parametric model of the shape
	### assumes that the shape of A and param agree

	# For connectivity
	if connectivity_mat is None:
		param = param

	else:
		# print("Using connectivity constraint for mesh generation.")
		param = np.matmul(connectivity_mat, param)

	pc = np.reshape(np.matmul(A, param), (-1, 3), order='C')

	return pc	

# def get_shape(A, param, src_default_param=None, weight=1.0, param_init=None):
# 	#batch matrix multiplication
# 	batch_size = param.shape[0]
# 	param_dim = param.shape[1]
# 	param = param.view(batch_size, param_dim, 1)
# 	src_default_param = src_default_param.view(batch_size, param_dim, 1)

# 	if (param_init==None):
# 		param = weight * param
# 	else:
# 		param_init = param_init.repeat(batch_size, 1)
# 		param_init = param_init.view(batch_size, param_dim, 1)
# 		param = weight * (param - param_init)

# 	if src_default_param == None:
# 		shape = torch.bmm(A, param)
# 	else:

# 		shape = torch.bmm(A, param+src_default_param)

# 	shape = shape.view(batch_size, -1, 3)

# 	return shape

# def get_source_info(source_labels, source_model_info, max_num_params):
# 	'''
# 	source_labels: contain the labels on which sources are assigned to each target
# 	source_model_info: dictionary containing the info of the source models such as matrix A and default params
# 	'''
# 	source_mats = []
# 	source_default_params = []
# 	for source_label in source_labels:
# 		points_mat = source_model_info[source_label]["points_mat"]
# 		padded_mat = np.zeros((points_mat.shape[0], max_num_params))
# 		padded_mat[0:points_mat.shape[0], 0:points_mat.shape[1]] = points_mat
# 		# padded_mat = np.expand_dims(padded_mat, axis=0)

# 		default_param = source_model_info[source_label]["default_param"]
# 		padded_default_param = np.zeros(max_num_params)
# 		padded_default_param[:default_param.shape[0]] = default_param
# 		padded_default_param = np.expand_dims(padded_default_param, axis=0)

# 		padded_mat = torch.from_numpy(padded_mat).float()
# 		padded_default_param = torch.from_numpy(padded_default_param).float()

# 		source_mats.append(padded_mat)
# 		source_default_params.append(padded_default_param)

# 	return source_mats, source_default_params

def get_shape(A, param, src_default_param=None, weight=1.0, param_init=None, connectivity_mat=None):
	#batch matrix multiplication
	batch_size = param.shape[0]
	param_dim = param.shape[1]
	param = param.view(batch_size, param_dim, 1)
	src_default_param = src_default_param.view(batch_size, param_dim, 1)

	if (param_init==None):
		param = weight * param
	else:
		param_init = param_init.repeat(batch_size, 1)
		param_init = param_init.view(batch_size, param_dim, 1)
		param = weight * (param - param_init)

	if src_default_param == None:
		if connectivity_mat is None:
			shape = torch.bmm(A, param)
		else:
			param = torch.bmm(connectivity_mat, param)
			shape = torch.bmm(A, param)

	else:
		if connectivity_mat is None:
			shape = torch.bmm(A, param+src_default_param)
		else:
			# print("Using connectivity constraint.")
			param = torch.bmm(connectivity_mat, param+src_default_param)
			shape = torch.bmm(A, param)

	shape = shape.view(batch_size, -1, 3)

	return shape

# Get initialized param for ICP fitting as post-process
def get_param_init(param, src_default_param=None, weight=1.0, param_init=None):
	#batch matrix multiplication
	batch_size = param.shape[0]
	param_dim = param.shape[1]
	param = param.view(batch_size, param_dim, 1)
	src_default_param = src_default_param.view(batch_size, param_dim, 1)

	if (param_init==None):
		param = weight * param
	else:
		param_init = param_init.repeat(batch_size, 1)
		param_init = param_init.view(batch_size, param_dim, 1)
		param = weight * (param - param_init)

	if src_default_param == None:
		return param

	else:
		return param+src_default_param

# To get back desired network output
def uninit_param(param, src_default_param=None, weight=1.0, param_init=None):
	#batch matrix multiplication
	batch_size = param.shape[0]
	param_dim = param.shape[1]
	param = param.view(batch_size, param_dim, 1)
	src_default_param = src_default_param.view(batch_size, param_dim, 1)

	if src_default_param == None:
		param = param

	else:
		param = param - src_default_param

	if (param_init==None):
		param = param / float(weight)
	else:
		param_init = param_init.repeat(batch_size, 1)
		param_init = param_init.view(batch_size, param_dim, 1)
		param = (param + param_init) / float(weight)

	return param

def icp_forward(A, param, connectivity_mat=None):
	#batch matrix multiplication
	batch_size = param.shape[0]
	param_dim = param.shape[2]
	param = param.view(batch_size, param_dim, 1)

	if connectivity_mat is None:
		shape = torch.bmm(A, param)
	else:
		param_cons = torch.bmm(connectivity_mat, param)
		shape = torch.bmm(A, param_cons)

	shape = shape.view(batch_size, -1, 3)
	return shape


def get_source_info(source_labels, source_model_info, max_num_params, use_connectivity=False):
	'''
	source_labels: contain the labels on which sources are assigned to each target
	source_model_info: dictionary containing the info of the source models such as matrix A and default params
	'''
	source_mats = []
	source_default_params = []
	source_connectivity_mat = []

	for source_label in source_labels:
		points_mat = source_model_info[source_label]["points_mat"]
		padded_mat = np.zeros((points_mat.shape[0], max_num_params))
		padded_mat[0:points_mat.shape[0], 0:points_mat.shape[1]] = points_mat
		# padded_mat = np.expand_dims(padded_mat, axis=0)

		default_param = source_model_info[source_label]["default_param"]
		padded_default_param = np.zeros(max_num_params)
		padded_default_param[:default_param.shape[0]] = default_param
		padded_default_param = np.expand_dims(padded_default_param, axis=0)

		padded_mat = torch.from_numpy(padded_mat).float()
		padded_default_param = torch.from_numpy(padded_default_param).float()

		source_mats.append(padded_mat)
		source_default_params.append(padded_default_param)
		# For connectivity constraint
		if (use_connectivity):
			constraint_proj_mat = source_model_info[source_label]["constraint_proj_mat"]
			constraint_padded_mat = np.zeros((max_num_params, max_num_params))
			constraint_padded_mat[0:constraint_proj_mat.shape[0], 0:constraint_proj_mat.shape[1]] = constraint_proj_mat	

			constraint_padded_mat = torch.from_numpy(constraint_padded_mat).float()
			source_connectivity_mat.append(constraint_padded_mat)

	if (use_connectivity):
		return source_mats, source_default_params, source_connectivity_mat
	else:
		return source_mats, source_default_params, None


def get_source_info_mesh(source_labels, source_model_info, max_num_params, use_connectivity=False):
	'''
	source_labels: contain the labels on which sources are assigned to each target
	source_model_info: dictionary containing the info of the source models such as matrix A and default params
	'''
	source_mats = []
	source_default_params = []
	source_connectivity_mat = []

	for source_label in source_labels:
		points_mat = source_model_info[source_label]["vertices_mat"]
		padded_mat = np.zeros((points_mat.shape[0], max_num_params))
		padded_mat[0:points_mat.shape[0], 0:points_mat.shape[1]] = points_mat
		# padded_mat = np.expand_dims(padded_mat, axis=0)

		default_param = source_model_info[source_label]["default_param"]
		padded_default_param = np.zeros(max_num_params)
		padded_default_param[:default_param.shape[0]] = default_param
		padded_default_param = np.expand_dims(padded_default_param, axis=0)

		padded_mat = torch.from_numpy(padded_mat).float()
		padded_default_param = torch.from_numpy(padded_default_param).float()

		source_mats.append(padded_mat)
		source_default_params.append(padded_default_param)

		if (use_connectivity):
			constraint_proj_mat = source_model_info[source_label]["constraint_proj_mat"]
			constraint_padded_mat = np.zeros((max_num_params, max_num_params))
			constraint_padded_mat[0:constraint_proj_mat.shape[0], 0:constraint_proj_mat.shape[1]] = constraint_proj_mat	

			constraint_padded_mat = torch.from_numpy(constraint_padded_mat).float()
			source_connectivity_mat.append(constraint_padded_mat)

	if not use_connectivity:
		source_connectivity_mat = None

	return source_mats, source_default_params, source_connectivity_mat

def get_source_info_visualization(source_labels, source_model_info, mesh=False):

	source_points = []
	source_point_labels = []
	source_model_ids = []
	source_vertices = []
	source_faces = []
	source_face_labels = []
	source_default_params = []

	for source_label in source_labels:

		src_points = source_model_info[source_label]["points"]
		src_point_labels = source_model_info[source_label]["point_labels"]
		default_param = source_model_info[source_label]["default_param"]
		src_model_id = source_model_info[source_label]["model_id"]		

		source_points.append(src_points)
		source_point_labels.append(src_point_labels)
		source_model_ids.append(src_model_id)
		source_default_params.append(default_param)

		if mesh:
			vertices = source_model_info[source_label]["vertices"]
			faces = source_model_info[source_label]["faces"]
			face_labels = source_model_info[source_label]["face_labels"]
			source_vertices.append(vertices)
			source_faces.append(faces)
			source_face_labels.append(face_labels)


	source_points = np.array(source_points)
	source_point_labels = np.array(source_point_labels)
	source_model_ids = np.array(source_model_ids)

	if mesh:
		source_vertices = np.array(source_vertices)
		source_faces = np.array(source_faces)
		source_face_labels = np.array(source_face_labels)		
		return source_points, source_point_labels, source_model_ids, source_default_params, source_vertices, source_faces, source_face_labels

	else:
		return source_points, source_point_labels, source_model_ids

def get_source_latent_codes(source_labels, latent_code_list):

	source_latent_codes = []

	for source_label in source_labels:
		src_latent_code = latent_code_list[source_label]
		source_latent_codes.append(src_latent_code)

	source_latent_codes = torch.stack(source_latent_codes)

	return source_latent_codes

def get_source_latent_codes_fixed(source_labels, latent_codes, device):
	source_labels = torch.from_numpy(source_labels)
	source_labels = source_labels.to(device)
	src_latent_codes = torch.gather(latent_codes, 0, source_labels.unsqueeze(-1).repeat(1,latent_codes.shape[-1]))

	return src_latent_codes

def get_source_latent_codes_encoder(source_labels, source_model_info, encoder, device):
	# print("Using encoder to get source latent codes.")
	source_points = []

	for source_label in source_labels:

		src_points = source_model_info[source_label]["points"]	
		source_points.append(src_points)

	source_points = np.array(source_points)
	source_points = torch.from_numpy(source_points)
	x = [x.to(device, dtype=torch.float) for x in source_points]
	x = torch.stack(x)

	src_latent_codes = encoder(x)

	# print(x.shape)
	# print(src_latent_codes.shape)
	# exit()

	return src_latent_codes

def get_source_latent_codes_encoder_images(source_labels, source_model_info, encoder, device, obj_cat, random=False):
	# print("Using encoder to get source latent codes.")
	img_size = 224
	source_images = []

	for source_label in source_labels:

		if random:
			src_id = source_model_info[source_label]["model_id"]

			## View either random or fixed (fixed to view-num 17)
			view = torch.randint(24, size=(1,)).to("cpu").detach().numpy()
			# view = np.array([17])

			IMAGE_BASE_DIR = "/orion/downloads/partnet_dataset/partnet_rgb_masks_"+obj_cat+"/"
			img_filename = os.path.join(IMAGE_BASE_DIR, str(int(src_id)), "view-"+str(int(view[0])).zfill(2), "shape-rgb.png")
			# img = Image.open(img_filename)
			# img = np.asarray(img)		

			with Image.open(img_filename) as fimg:
			    out = np.array(fimg, dtype=np.float32) / 255.0
			white_img = np.ones((img_size, img_size, 3), dtype=np.float32)
			mask = np.tile(out[:, :, 3:4], [1, 1, 3])

			out = out[:, :, :3] * mask + white_img * (1 - mask)
			out = torch.from_numpy(out).permute(2, 0, 1)

		else:
			out = source_model_info[source_label]["image"]

		source_images.append(out)

	x = [x.to(device, dtype=torch.float) for x in source_images]
	x = torch.stack(x)

	src_latent_codes = encoder(x)

	# print(x.shape)
	# print(src_latent_codes.shape)
	# exit()

	return src_latent_codes

def get_source_points(source_labels, source_model_info, device):
	# print("Using encoder to get source latent codes.")
	source_points = []

	for source_label in source_labels:

		src_points = source_model_info[source_label]["points"]	
		source_points.append(src_points)

	source_points = np.array(source_points)
	source_points = torch.from_numpy(source_points)
	x = [x.to(device, dtype=torch.float) for x in source_points]
	x = torch.stack(x)

	return x

def get_random_labels(source_labels, num_source_models):
	random_source_labels = np.random.randint(0, num_source_models, size=source_labels.shape)

	return random_source_labels

def get_all_source_labels(source_labels, num_source_models):
	# Get the source model label assignments
	# Now it is just selecting all models
	# Later it will based on retrieval

	batch_size = int(source_labels.shape[0]/num_source_models)

	source_labels = np.expand_dims(np.arange(num_source_models), axis=1)
	source_labels = np.tile(source_labels, (1, batch_size))

	source_labels = np.reshape(source_labels, (-1))
	# print(source_labels)
	return source_labels
