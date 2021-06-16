#!/usr/bin/env python

# Built on top of code from 
# Minhyuk Sung (mhsung@cs.stanford.edu)

# Modified and appended by 
# Mikaela Uy (mikacuy@stanford.edu)

import os, sys
BASE_DIR = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__))))

from global_variables import *
import argparse
import glob
import h5py
import json
import numpy as np
import scipy.linalg
import trimesh
from scipy.spatial import distance_matrix

# Parallel processing.
from joblib import Parallel, delayed
import multiprocessing

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--category', nargs='?', type=str, default='chair')
args = parser.parse_args()

### For chair
SEMANTIC_DICT = {   "chair_head":0,
                    "chair_back":1,
                    "chair_arm":2,
                    "chair_base":3,
                    "chair_seat":4,
                    "footrest":5}

SEMANTIC_DICT_INV = {   0: "chair_head",
                    1: "chair_back",
                    2: "chair_arm",
                    3: "chair_base",
                    4: "chair_seat",
                    5: "footrest"}

KEYPOINT_LABEL = {}
for i in range(6):
    KEYPOINT_LABEL[i] = "face center"
for i in range(6, 18):
    KEYPOINT_LABEL[i] = "edge midpoint"
for i in range(18, 26):
    KEYPOINT_LABEL[i] = "corner point"

def collect_leaf_nodes(structurenet_json):
    with open(structurenet_json, 'r') as f:
        data = json.load(f)

    data["level"] = 0
    queue = [data]
    leaves = []

    while queue:
        node = queue.pop()

        if 'children' not in node:
            leaves.append(node)
        else:

            for child in node['children']:
                child["level"] = node["level"] + 1
                child["label"] = node["label"] + "/" + child["label"]
                queue.append(child)

    return leaves


def find_corresponding_meshes(partnet_json, nodes):
    with open(partnet_json, 'r') as f:
        data = json.load(f)

    queue = data

    while queue:
        elem = queue.pop()

        for node in nodes:
            if node['id'] == elem['id']:
                node['objs'] = elem['objs']

        if 'children' in elem:
            for child in elem['children']:
                queue.append(child)

    return nodes


def get_default_param(box):
    # Center, scale, X axis, Y axis.
    c, double_s, ax, ay = np.array(box[0:3]), np.array(box[3:6]), \
            np.array(box[6:9]), np.array(box[9:12])

    # Scale: Half of the box scale.
    s = 0.5 * double_s

    # Translation: The box center.
    t = c

    return np.concatenate((t, s))


def compute_param_linear_equation(P, box):
    assert(np.shape(P)[1] == 3)
    assert(np.size(box) == 12)
    N = np.shape(P)[0]

    # Center, scale, X axis, Y axis.
    c, double_s, ax, ay = np.array(box[0:3]), np.array(box[3:6]), \
            np.array(box[6:9]), np.array(box[9:12])
    # Z axis.
    az = np.cross(ax, ay)

    # Rotation: X Y Z axes are columns.
    R = np.transpose(np.vstack((ax, ay, az)))
    inv_R = np.transpose(R)

    # Scale: Half of the box scale.
    s = 0.5 * double_s
    S = np.diag(s)
    inv_S = np.diag(1. / s)

    # Translation: The box center.
    t = c

    # q = S^-1 * R^-1 * (p - t)
    #q = np.matmul(inv_S, np.matmul(inv_R, (p - t)))
    Pt = np.transpose(P)
    Qt = np.matmul(inv_S, np.matmul(inv_R, (Pt - np.expand_dims(t, 1))))
    # (N, 3)
    Q = np.transpose(Qt)

    # p = R * S * q + t
    #   = R * diag(s) * q + t
    #   = R * diag(q) * s + Id * t
    #   = A_s * s + A_t * t, where A_s = R * diag(q) and A_t = Id.
    #   = A * [t^T, s^T]^T, where A = [A_t | A_s]
    #A_s = np.matmul(R, np.diag(q))
    #A_t = np.eye(3)
    #A = np.hstack((A_t, A_s))
    # (N, 3, 3)
    A_s = np.expand_dims(Q, 1) * np.expand_dims(R, 0)
    # (N, 3, 3)
    A_t = np.tile(np.eye(3), (N, 1, 1))
    # (N, 3, 6)
    A = np.concatenate((A_t, A_s), axis=2)
    # (3*N, 6)
    A = np.reshape(A, (3*N, 6))

    return A


def compute_param_models(nodes, target_only=False):
    for node in nodes:
        node['default_param'] = get_default_param(node['box'])
        node['vertices_mat'] = compute_param_linear_equation(
                node['vertices'], node['box'])

    return nodes


def merge_meshes(V_list, F_list, labels=None):
    V = np.empty((0, 3), dtype=int)
    F = np.empty((0, 3), dtype=int)
    n_meshes = len(V_list)
    assert(len(F_list) == n_meshes)

    if labels is not None:
        assert(len(labels) == n_meshes)
        VL = np.empty(0, dtype=int)
        FL = np.empty(0, dtype=int)

    for i in range(n_meshes):
        nV = np.shape(V)[0]
        V = np.vstack((V, V_list[i]))
        F = np.vstack((F, (nV + F_list[i])))

        if labels is not None:
            n_vertices = np.shape(V_list[i])[0]
            n_faces = np.shape(F_list[i])[0]
            VL = np.concatenate((VL, np.array([labels[i]] * n_vertices)))
            FL = np.concatenate((FL, np.array([labels[i]] * n_faces)))

    if labels is not None:
        return V, F, VL, FL

    return V, F


def load_meshes(nodes, in_dir):
    for node in nodes:
        in_files = [os.path.join(in_dir, '{}.obj'.format(x)) \
                for x in node['objs']]

        meshes = [trimesh.load(x) for x in in_files]
        V_list = [mesh.vertices for mesh in meshes]
        F_list = [mesh.faces for mesh in meshes]
        node['vertices'], node['faces'] = merge_meshes(V_list, F_list)

    return nodes

def compute_aabbox(nodes):
    for node in nodes:
        vertices = node["vertices"]

        min_x = np.min(vertices[:,0])
        max_x = np.max(vertices[:,0])
        min_y = np.min(vertices[:,1])
        max_y = np.max(vertices[:,1])
        min_z = np.min(vertices[:,2])
        max_z = np.max(vertices[:,2])

        double_s = np.array([max_x-min_x, max_y-min_y, max_z-min_z])
        c = np.array([min_x+double_s[0]/2.0, min_y+double_s[1]/2.0, min_z+double_s[2]/2.0])
        ax = np.array([1.,0.,0.], dtype=np.float32)
        ay = np.array([0.,1.,0.], dtype=np.float32) 

        aabbox = np.zeros(12, dtype=np.float32)
        aabbox[0:3] = c
        aabbox[3:6] = double_s
        aabbox[6:9] = ax
        aabbox[9:12] = ay

        aabbox = list(aabbox)
        # To make it json serializable, convert to python float
        for i in range(len(aabbox)):
            aabbox[i] = float(aabbox[i])

        node['box'] = aabbox

    return nodes    

def get_semantics(nodes):

    for node in nodes:

        ## Dummy --> this was not used!!!
        node["semantic_label"] = 0

    return nodes   

def face_areas(V, F):
    M = np.shape(F)[0]

    # Compute face areas.
    tri_verts = np.empty((M, 3, 3))
    for i in range(M):
        for j in range(3):
            tri_verts[i,j] = V[F[i,j]]

    areas = trimesh.triangles.area(tri_verts)

    return areas


def sample_points(V, F, VA, n_sample_points):
    # Sample faces with area weights.
    areas = face_areas(V, F)
    assert(np.sum(areas) > g_zero_tol)
    probs = areas / np.sum(areas)
    fids = np.random.choice(np.shape(F)[0], n_sample_points, p=probs)

    PA = np.zeros((3 * n_sample_points, np.shape(VA)[1]))

    for pid in range(n_sample_points):
        vids = F[fids[pid]]

        # Sample barycentric coordinates.
        weights = np.random.uniform(size=3)
        weights = weights / np.sum(weights)

        for i in range(3):      # Three vertices.
            for j in range(3):  # Three coordinates (x, y, z).
                PA[3 * pid + j] += weights[i] * VA[3 * vids[i] + j]

    return PA


def sample_all_node_points(nodes, n_sample_points, dense=False, target_only=False):
    n_nodes = len(nodes)
    assert(n_nodes > 0)

    for node in nodes:
        node['face_areas'] = face_areas(node['vertices'], node['faces'])

    node_areas = [np.sum(node['face_areas']) for node in nodes]
    assert(np.sum(node_areas) > g_zero_tol)
    probs = node_areas / np.sum(node_areas)
    labels = np.random.choice(n_nodes, n_sample_points, p=probs)


    non_zero_nodes = []

    for i in range(n_nodes):
        node = nodes[i]
        n_node_sample_points = np.sum(labels == i)

        if (n_node_sample_points == 0):
            print("Warning: node without sampled points found.")
            continue

        # For denser sampling in computing for connectivity
        if (dense):
            node['dense_points_mat'] = sample_points(node['vertices'], node['faces'],
                    node['vertices_mat'], n_node_sample_points)

            x0 = node['default_param']
            PA = node['dense_points_mat']
            node['dense_points'] = np.reshape(np.matmul(PA, x0),
                    (n_node_sample_points, 3), order='C')

        # For sampling and getting parametric models for the shape
        else:
            node['points_mat'] = sample_points(node['vertices'], node['faces'],
                    node['vertices_mat'], n_node_sample_points)

            x0 = node['default_param']
            PA = node['points_mat']
            node['points'] = np.reshape(np.matmul(PA, x0),
                    (n_node_sample_points, 3), order='C')

        non_zero_nodes.append(node)

    # return nodes
    return non_zero_nodes

def get_separating_axis(pc1, pc2):
    ## Get axis of connectivity by getting the separating plane
    ## return 0, 1, 2 for x, y, z axes

    pc1_xmin = np.min(pc1[:,0])
    pc1_xmax = np.max(pc1[:,0])
    pc1_ymin = np.min(pc1[:,1])
    pc1_ymax = np.max(pc1[:,1])
    pc1_zmin = np.min(pc1[:,2])
    pc1_zmax = np.max(pc1[:,2])

    pc2_xmin = np.min(pc2[:,0])
    pc2_xmax = np.max(pc2[:,0])
    pc2_ymin = np.min(pc2[:,1])
    pc2_ymax = np.max(pc2[:,1])
    pc2_zmin = np.min(pc2[:,2])
    pc2_zmax = np.max(pc2[:,2])

    ## Get intersection
    x_endpt1 = np.max([pc1_xmin, pc2_xmin])
    x_endpt2 = np.min([pc2_xmax, pc1_xmax])
    y_endpt1 = np.max([pc1_ymin, pc2_ymin])
    y_endpt2 = np.min([pc2_ymax, pc1_ymax])
    z_endpt1 = np.max([pc1_zmin, pc2_zmin])
    z_endpt2 = np.min([pc2_zmax, pc1_zmax])        


    x_intersection = np.abs(x_endpt2-x_endpt1)
    y_intersection = np.abs(y_endpt2-y_endpt1)
    z_intersection = np.abs(z_endpt2-z_endpt1)

    ## Get union
    x_endpt1 = np.min([pc1_xmin, pc2_xmin])
    x_endpt2 = np.max([pc2_xmax, pc1_xmax])
    y_endpt1 = np.min([pc1_ymin, pc2_ymin])
    y_endpt2 = np.max([pc2_ymax, pc1_ymax])
    z_endpt1 = np.min([pc1_zmin, pc2_zmin])
    z_endpt2 = np.max([pc2_zmax, pc1_zmax])        

    x_union = np.abs(x_endpt2-x_endpt1)
    y_union = np.abs(y_endpt2-y_endpt1)
    z_union = np.abs(z_endpt2-z_endpt1)

    ##IOU
    x_iou = x_intersection/x_union
    y_iou = y_intersection/y_union
    z_iou = z_intersection/z_union

    IOU = np.array([x_iou, y_iou, z_iou])
    print(IOU)

    ## Select the axis with the smallest IOU
    axis_idx = np.argmin(IOU)

    axis = ["x-axis", "y-axis", "z-axis"]
    print("Connectivity constraint at "+axis[axis_idx])
    print()

    return axis_idx

### Get keypoints based on the part bounding box that will define the connectivity joint
def get_part_keypoints(node, project=False):
    bbox = node["box"]
    keypoints = []

    box_center = bbox[:3]
    scale_x = np.array([bbox[3]/2.0, 0.0, 0.0])
    scale_y = np.array([0.0, bbox[4]/2.0, 0.0])
    scale_z = np.array([0.0, 0.0, bbox[5]/2.0])

    ## Box face centers
    keypoints.append(box_center-scale_x)
    keypoints.append(box_center+scale_x)
    keypoints.append(box_center-scale_y)
    keypoints.append(box_center+scale_y)
    keypoints.append(box_center-scale_z)
    keypoints.append(box_center+scale_z)

    ## Edge midpoints
    keypoints.append(box_center-scale_x-scale_y)
    keypoints.append(box_center-scale_x+scale_y)
    keypoints.append(box_center-scale_x-scale_z)
    keypoints.append(box_center-scale_x+scale_z)
    keypoints.append(box_center+scale_x-scale_y)
    keypoints.append(box_center+scale_x+scale_y)
    keypoints.append(box_center+scale_x-scale_z)
    keypoints.append(box_center+scale_x+scale_z)
    keypoints.append(box_center-scale_y-scale_z)
    keypoints.append(box_center-scale_y+scale_z)
    keypoints.append(box_center+scale_y-scale_z)
    keypoints.append(box_center+scale_y+scale_z)

    ## Box corners
    keypoints.append(box_center-scale_x-scale_y-scale_z)
    keypoints.append(box_center-scale_x-scale_y+scale_z)
    keypoints.append(box_center-scale_x+scale_y-scale_z)
    keypoints.append(box_center-scale_x+scale_y+scale_z)
    keypoints.append(box_center+scale_x-scale_y-scale_z)
    keypoints.append(box_center+scale_x-scale_y+scale_z)
    keypoints.append(box_center+scale_x+scale_y-scale_z)
    keypoints.append(box_center+scale_x+scale_y+scale_z)

    keypoints = np.array(keypoints)

    if (project):
        ## Projected keypoint to the nearest point in the point cloud
        pc = node["dense_points"]
        projected_keypoints = []

        for keypoint in keypoints:
            kp = np.expand_dims(keypoint, axis=0)
            dists = distance_matrix(pc, kp)
            idx = np.unravel_index(dists.argmin(), dists.shape)
            projected_kp = pc[idx[0]]

            projected_keypoints.append(projected_kp)

        projected_keypoints = np.array(projected_keypoints)

        return projected_keypoints

    else:
        return keypoints

### Get connected nodes and joint position
def get_connectivity(nodes, dense=False, single_axis_constraint=False, keypoint_based=False):
    n_nodes = len(nodes)

    connected_parts = []
    joints = []
    conn_axes = []
    for i in range(n_nodes):
        if dense:
            part_pc1 = nodes[i]["dense_points"]
        else:
            part_pc1 = nodes[i]["points"]

        for j in range(i+1, n_nodes):
            if dense:
                part_pc2 = nodes[j]["dense_points"]
            else:  
                part_pc2 = nodes[j]["points"]

            #pairwise distances
            dists = distance_matrix(part_pc1, part_pc2)

            closest_dist = dists.min()
            idx = np.unravel_index(dists.argmin(), dists.shape)

            if (closest_dist < g_adjacency_tol):

                ##Points that are adjacent (connected part)
                part1_pt = part_pc1[idx[0]]
                part2_pt = part_pc2[idx[1]]

                joint = (part1_pt + part2_pt)/2
                connected_parts.append((i, j))

                if (single_axis_constraint):
                    conn_axis = get_separating_axis(part_pc1, part_pc2)
                    conn_axes.append(conn_axis)

                ## Get the joint based on a keypoint pair
                elif (keypoint_based):
                    keypoints_i = get_part_keypoints(nodes[i], project=True)
                    keypoints_j = get_part_keypoints(nodes[j], project=True)

                    dists = distance_matrix(keypoints_i, keypoints_j)
                    closest_dist = dists.min()
                    idx = np.unravel_index(dists.argmin(), dists.shape)

                    part1_pt = keypoints_i[idx[0]]
                    part2_pt = keypoints_j[idx[1]]
                    joint = (part1_pt + part2_pt)/2
                    
                    print("Keypoint pair distance.")
                    print(closest_dist)
                    ## Print keypoint type of connection
                    print("Connected keypoints: "+KEYPOINT_LABEL[idx[0]]+" "+KEYPOINT_LABEL[idx[1]])
                    print()

                joints.append(joint)

    print("Number of parameters: " + str(n_nodes*6))
    if not single_axis_constraint:
        print("Number of constraints: " + str(len(connected_parts)*3))
    else:
        print("Number of constraints: " + str(len(conn_axes)))

    return connected_parts, joints, conn_axes


# Get connectivity constraints B_1p_1 = B_2p_2
# For predicted parameters p_1 and p_2
# B_i is computed using the joint and ith param box parameters
def get_linear_connectivity_constraints(nodes, connected_parts, joints):
    # Get the linear constraint equation for each pair of connected parts
    B_pairs = []

    for c in range(len(joints)):
        i, j = connected_parts[c]
        node1 = nodes[i]
        node2 = nodes[j]

        #joint point p
        p = joints[c]
        p = np.expand_dims(p, axis=0)

        B1 = compute_param_linear_equation(p, node1["box"])
        B2 = compute_param_linear_equation(p, node2["box"])

        B_pairs.append((B1, B2))

    return B_pairs

# Get full connectivity matrix B
# B_1p_1 - B_2p_2 = 0
# Construct a block diagonal matrix with rows [...B_i ... -B_j...] for each constraint
def get_connectivity_matrix(nodes, connected_parts, B_pairs, conn_axes, single_axis_constraint=False):
    n_constraints = len(connected_parts)
    n_nodes = len(nodes)

    if not single_axis_constraint:
        # Initialize (C*3 x K*6) matrix. (3x6) for each part constraint
        B = np.zeros((n_constraints*3, n_nodes*6))
    else:
        # Initialize (C x K*6) matrix. (3x6) for each part constraint
        B = np.zeros((n_constraints, n_nodes*6))  

    for c in range(len(connected_parts)):
        i, j = connected_parts[c]
        B1, B2 = B_pairs[c]

        if not single_axis_constraint:
            B[c*3:(c+1)*3, 6*i:6*(i+1)] = B1
            B[c*3:(c+1)*3, 6*j:6*(j+1)] = -B2

        else:
            curr_constraint_axis = conn_axes[c]
            B[c, 6*i:6*(i+1)] = B1[curr_constraint_axis, :]
            B[c, 6*j:6*(j+1)] = -B2[curr_constraint_axis, :]

    print(B.shape)
    return B

def get_constraint_projection_matrix(nodes, B):
    n_constraints = B.shape[0]
    n_nodes = len(nodes)
    n_params = n_nodes * 6

    dim_null_space = int(n_params - n_constraints)

    U, S, VT = np.linalg.svd(B)

    # Check linear dependent constraints
    ## Check eigenvalues that are small
    ## eigenvalues are sorted so just make the smallest one
    dependent_cols = np.argwhere(S < g_zero_tol)

    if (dependent_cols.shape[0]>0):
        # Linear dependent columns
        print("Linear dependent constraints found.")
        min_depedent_cols = np.min(dependent_cols)

        null_space_start_col = -dim_null_space-(n_constraints-min_depedent_cols)

    else:
        null_space_start_col = -dim_null_space

    # Check if null space is non-empty
    if (null_space_start_col >= 0):
        print("Null space does not exists. Cannot get constraint matrix")
        return []

    V = VT.T
    N = V[:, null_space_start_col:]
    NNT = np.matmul(N, N.T)

    print(N.shape)

    return NNT

def merge_nodes(nodes, model_id, target_only = False):
    shape = {}

    shape['vertices'], shape['faces'], \
            shape['vertex_labels'], shape['face_labels'] = merge_meshes(
            [node['vertices'] for node in nodes],
            [node['faces'] for node in nodes],
            np.arange(len(nodes)))

    shape['vertex_ids'] = np.array([nodes[label]['id'] \
            for label in shape['vertex_labels']])
    shape['face_ids'] = np.array([nodes[label]['id'] \
            for label in shape['face_labels']])

    shape['vertex_semantic'] = np.array([nodes[label]['semantic_label'] \
            for label in shape['vertex_labels']])

    shape['vertices_mat'] = scipy.linalg.block_diag(
            *[node['vertices_mat'] for node in nodes])

    shape['points'], _, shape['point_labels'], _ = merge_meshes(
            [node['points'] for node in nodes],
            [np.empty((0, 3), dtype=int) for node in nodes],
            np.arange(len(nodes)))
    shape['point_ids'] = np.array([nodes[label]['id'] \
            for label in shape['point_labels']])

    shape['point_semantic'] = np.array([nodes[label]['semantic_label'] \
            for label in shape['point_labels']])

    shape['points_mat'] = scipy.linalg.block_diag(
            *[node['points_mat'] for node in nodes])

    shape['default_param'] = np.concatenate(
            [node['default_param'] for node in nodes])

    return shape


def render_mesh(mesh_file, face_labels_file, snapshot_file):
    cmd = g_renderer + ' \\\n'
    cmd += ' --mesh=' + mesh_file + ' \\\n'
    cmd += ' --face_labels=' + face_labels_file + ' \\\n'
    cmd += ' --snapshot=' + snapshot_file + ' \\\n'
    cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
    cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
    cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
    cmd += ' >/dev/null 2>&1'
    os.system(cmd)
    snapshot_file += '.png'
    print("Saved '{}'.".format(snapshot_file))


def render_point_cloud(point_cloud_file, point_labels_file, snapshot_file):
    cmd = g_renderer + ' \\\n'
    cmd += ' --point_cloud=' + point_cloud_file + ' \\\n'
    cmd += ' --point_labels=' + point_labels_file + ' \\\n'
    cmd += ' --snapshot=' + snapshot_file + ' \\\n'
    cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
    cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
    cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
    cmd += ' >/dev/null 2>&1'
    os.system(cmd)
    snapshot_file += '.png'
    print("Saved '{}'.".format(snapshot_file))


def render_boxes(json_file, snapshot_file):
    cmd = g_renderer + ' \\\n'
    cmd += ' --structurenet_json=' + json_file + ' \\\n'
    cmd += ' --snapshot=' + snapshot_file + ' \\\n'
    cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
    cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
    cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
    cmd += ' >/dev/null 2>&1'
    os.system(cmd)
    snapshot_file += '.png'
    print("Saved '{}'.".format(snapshot_file))


def save_shape_data(leaves, shape, out_dir, model_id, target_only=False):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    ## Box Template ###
    # Save leaf node json file.

    leaves = [{'box': leaf['box'], 'id': leaf['id'], 'semantic_label' : leaf['semantic_label']} for leaf in leaves]

    out_json_file = os.path.join(out_dir, "boxes", str(model_id)+'_leaves.json')

    with open(out_json_file, 'w') as f:
        json.dump(leaves, f, indent=2)
    print("Saved '{}'.".format(out_json_file))

    # Render box template.
    boxes_snapshot_file = os.path.join(out_dir, "rendering_boxes", str(model_id)+'_boxes')
    render_boxes(out_json_file, boxes_snapshot_file)

    ### Mesh ###
    # Save mesh.
    out_mesh_file = os.path.join(out_dir, "mesh", str(model_id)+'_mesh.obj')
    mesh = trimesh.Trimesh(vertices=shape['vertices'], faces=shape['faces'])  
    mesh.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
    print("Saved '{}'.".format(out_mesh_file))

    # Save vertex ids.
    dummy = np.zeros(shape['vertex_ids'].shape)
    out_vertex_ids_file = os.path.join(out_dir, "mesh", str(model_id)+'_vertex_ids.txt')
    np.savetxt(out_vertex_ids_file, dummy, fmt='%d')
    print("Saved '{}'.".format(out_vertex_ids_file))

    # Save face ids.
    dummy = np.zeros(shape['face_ids'].shape)
    out_face_ids_file = os.path.join(out_dir, "mesh", str(model_id)+'_face_ids.txt')
    np.savetxt(out_face_ids_file, dummy, fmt='%d')
    print("Saved '{}'.".format(out_face_ids_file))

    # Render mesh.
    mesh_snapshot_file = os.path.join(out_dir, "rendering_mesh", str(model_id)+'_mesh')
    render_mesh(out_mesh_file, out_face_ids_file, mesh_snapshot_file)


    ### Point Cloud ###
    # Save point cloud.
    out_point_cloud_file = os.path.join(out_dir, "point_cloud", str(model_id)+'_points.xyz')
    np.savetxt(out_point_cloud_file, shape['points'], delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(out_point_cloud_file))

    # Save point ids.
    out_point_ids_file = os.path.join(out_dir, "point_cloud", str(model_id)+'_point_ids.txt')
    np.savetxt(out_point_ids_file, shape['point_ids'], fmt='%d')
    print("Saved '{}'.".format(out_point_ids_file))

    # Render point_cloud.
    points_snapshot_file = os.path.join(out_dir, "rendering_pc", str(model_id)+'_points')
    render_point_cloud(out_point_cloud_file, out_point_ids_file,
            points_snapshot_file)

    ##### Important part
    # Save HDF5 file.

    orig_ids = [leaf['id'] for leaf in leaves]
    box_params = np.stack([leaf['box'] for leaf in leaves])
    semantic_label = [leaf['semantic_label'] for leaf in leaves]

    if not target_only:
        select_keys = ['vertex_labels', 'face_labels', 'point_labels', 'faces',
                'vertices', 'vertices_mat', 
                'points', 'points_mat', 'default_param',
                'point_semantic', 'vertex_semantic', 'constraint_mat', 'constraint_proj_mat']
    
    # Only care about output shape and not the parametric model
    else:
        select_keys = ['vertex_labels', 'face_labels', 'point_labels', 'faces',
                'vertices', 'vertices_mat',
                'points', 'points_mat','default_param',
                'point_semantic', 'vertex_semantic']      

    out_h5_file = os.path.join(out_dir, "h5", str(model_id)+'_leaves.h5')
    with h5py.File(out_h5_file, 'w') as f:
        f.create_dataset('box_params', data=box_params, compression="gzip")
        f.create_dataset('orig_ids', data=orig_ids, compression="gzip")
        f.create_dataset('semantic_label', data=semantic_label, compression="gzip")
        for key in select_keys:
            f.create_dataset(key, data=shape[key], compression="gzip")
    # print(select_keys)
    print("Saved '{}'.".format(out_h5_file))
    #############

    #Output to a single image
    height = 1080
    width = 1920
    new_im = Image.new('RGBA', (width*2, height))
    im1 = Image.open(boxes_snapshot_file+".png")
    im2 = Image.open(mesh_snapshot_file+".png")
    images = [im1, im2]
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += width

    output_image_filename = os.path.join(out_dir, "rendering_both", str(model_id)+'_both.png')
    new_im.save(output_image_filename)    
    print("Saved '{}'.".format(output_image_filename))

def process_model(category, model_id, axis_aligned = False, target_only=False, single_axis_constraint=False, keypoint_based=False):
    print("Model ID: {}".format(model_id))

    structurenet_json = os.path.join(g_structurenet_input_dir,
        '{}_hier'.format(category), '{}.json'.format(model_id))
    assert(os.path.exists(structurenet_json))

    partnet_json = os.path.join(g_partnet_dir, str(model_id),
            'result_after_merging.json')
    assert(os.path.exists(partnet_json))

    # Collect leaf nodes.
    leaves = collect_leaf_nodes(structurenet_json)
    n_leaves = len(leaves)

    if not target_only:
        # Skip if the number of parts is greater than the threshold.
        if n_leaves <= 1 or n_leaves > g_max_num_parts:
            print("Warning: Number of parts greater than threshold: {:d}.".format(
                n_leaves))
            return []

    # Skip if scale is close to zero.
    for leaf in leaves:
        scales = np.array(leaf['box'][3:6])
        if np.any(scales < g_zero_tol):
            print("Warning: Scale less than zero threshold: {:f}.".format(
                np.amin(scales)))
            return []

    # Collect OBJ mesh files.
    leaves = find_corresponding_meshes(partnet_json, leaves)

    # Skip if meshes do not exist.
    for leaf in leaves:
        if 'objs' not in leaf:
            print("Warning: Mesh does not exist.")
            return []

    # Load meshes.
    partnet_mesh_dir = os.path.join(g_partnet_dir, str(model_id), 'objs')
    leaves = load_meshes(leaves, partnet_mesh_dir)

    # Use axis-aligned bounding box
    if (axis_aligned):
        leaves = compute_aabbox(leaves)

    # Get semantic labels
    leaves = get_semantics(leaves)

    # Compute parametric models.
    leaves = compute_param_models(leaves, target_only = target_only)
    if not leaves: return leaves

    # Sample points.
    leaves = sample_all_node_points(leaves, g_num_sample_points, target_only=target_only)
    if not leaves: return leaves

    # Merge meshes, point clouds, and linear equations.
    shape = merge_nodes(leaves, model_id, target_only=target_only)
    if shape is None: return []

    if not target_only:
        # Get mesh connectivity
        leaves = sample_all_node_points(leaves, 15000, dense=True)
        connected_parts, joints, conn_axes = get_connectivity(leaves, dense=True, single_axis_constraint=single_axis_constraint, keypoint_based=keypoint_based)

        # Get individual connectivity constraints
        B_is = get_linear_connectivity_constraints(leaves, connected_parts, joints)

        # Merge constraints
        B = get_connectivity_matrix(leaves, connected_parts, B_is, conn_axes, single_axis_constraint=single_axis_constraint)

        # Get constraint matrix by computing for the null space
        NNT = get_constraint_projection_matrix(leaves, B)
        if NNT is None: return []

        shape["constraint_mat"] = B
        shape["constraint_proj_mat"] = NNT
 
    # out_dir = os.path.join(g_structurenet_output_dir, category, str(model_id))
    out_dir = os.path.join(g_structurenet_output_dir, category)

    save_shape_data(leaves, shape, out_dir, model_id, target_only=target_only)

    return leaves

if __name__ == "__main__":
    np.random.seed(0)

    structurenet_json_list = glob.glob(os.path.join(g_structurenet_input_dir,
        '{}_hier'.format(args.category), '*.json'))

    model_id_list = [int(os.path.basename(os.path.splitext(x)[0])) \
            for x in structurenet_json_list]

    os.makedirs(os.path.join(g_structurenet_output_dir, args.category), exist_ok=True)

    ##Create subfolders if they dont exist
    os.makedirs(os.path.join(g_structurenet_output_dir, args.category, "rendering_boxes"), exist_ok=True)
    os.makedirs(os.path.join(g_structurenet_output_dir, args.category, "rendering_mesh"), exist_ok=True)
    os.makedirs(os.path.join(g_structurenet_output_dir, args.category, "rendering_pc"), exist_ok=True)
    os.makedirs(os.path.join(g_structurenet_output_dir, args.category, "rendering_both"), exist_ok=True)
    os.makedirs(os.path.join(g_structurenet_output_dir, args.category, "h5"), exist_ok=True)
    os.makedirs(os.path.join(g_structurenet_output_dir, args.category, "boxes"), exist_ok=True)
    os.makedirs(os.path.join(g_structurenet_output_dir, args.category, "mesh"), exist_ok=True)
    os.makedirs(os.path.join(g_structurenet_output_dir, args.category, "point_cloud"), exist_ok=True)

    # # Parallel processing for constraints.
    # num_cores = int(0.8 * multiprocessing.cpu_count())
    # results = Parallel(n_jobs=num_cores)(delayed(
    #     process_model)(args.category, model_id, axis_aligned=True) for model_id in model_id_list)

    # # Parallel processing. with constraints single axis
    # num_cores = int(0.8 * multiprocessing.cpu_count())
    # results = Parallel(n_jobs=num_cores)(delayed(
    #     process_model)(args.category, model_id, axis_aligned=True, single_axis_constraint=True) for model_id in model_id_list)

    # Parallel processing. with keypoint constraints
    num_cores = int(0.4 * multiprocessing.cpu_count())
    results = Parallel(n_jobs=num_cores)(delayed(
        process_model)(args.category, model_id, axis_aligned=True, keypoint_based=True) for model_id in model_id_list)

    # # Targets only, only care about point cloud
    # # Parallel processing.
    # num_cores = int(0.4 * multiprocessing.cpu_count())
    # results = Parallel(n_jobs=num_cores)(delayed(
    #     process_model)(args.category, model_id, axis_aligned=True, target_only=True) for model_id in model_id_list)

    print("Total number of models: " + str(len(model_id_list)))

