#!/usr/bin/python

#------------------------------------------------------------------------------
# Define all the global variables for the project
#------------------------------------------------------------------------------

from __future__ import division
import os, sys
BASE_DIR = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__))))


g_renderer = '/orion/u/mhsung/app/primitive-fitting/build/OSMesaRenderer'
g_azimuth_deg = -70
g_elevation_deg = 20
g_theta_deg = 0

g_partnet_dir = '/orion/group/PartNet/data_v0'
g_structurenet_root_dir = '/orion/u/mhsung/projects/deformation-space-learning/StructureNet'
g_structurenet_input_dir = os.path.join(g_structurenet_root_dir, 'raw')

output_fol = "/orion/u/mikacuy/part_deform/"
# g_structurenet_output_dir = os.path.join(output_fol, 'data_aabb_constraints')
# g_structurenet_output_dir = os.path.join(output_fol, 'data_aabb_constraints_singleaxis')
g_structurenet_output_dir = os.path.join(output_fol, 'data_aabb_constraints_keypoint')
# g_structurenet_output_dir = os.path.join(output_fol, 'data_aabb_constraints_randomseg_final')
# g_structurenet_output_dir = os.path.join(output_fol, 'data_aabb_constraints_randomseg_toydata')
# g_structurenet_output_dir = os.path.join(output_fol, 'data_aabb_constraints_symmetry')
# g_structurenet_output_dir = os.path.join(output_fol, 'data_aabb_all_models')

# g_structurenet_output_dir = os.path.join(output_fol, 'data_aabb_all_models_dense')

# g_structurenet_output_dir = os.path.join(output_fol, '0_teaser')

g_zero_tol = 1.0e-6
g_min_num_parts = 4
g_max_num_parts = 16
g_num_sample_points = 2048

# g_num_sample_points = 8192

# For connectivity
g_adjacency_tol = 5.0e-2

