# Joint Learning of 3D Shape Retrieval and Deformation
**[Joint Learning of 3D Shape Retrieval and Deformation](https://joint-retrieval-deformation.github.io)** 

Mikaela Angelina Uy, Vladimir G. Kim, Minhyuk Sung, Noam Aigerman, Siddhartha Chaudhuri and Leonidas Guibas

CVPR 2021

![pic-network](teaser3.png)

## Introduction
We propose a novel technique for producing high-quality 3D models that match a given target object image or scan. Our method is based on retrieving an existing shape from a database of 3D models and then deforming its parts to match the target shape. Unlike previous approaches that in- dependently focus on either shape retrieval or deformation, we propose a joint learning procedure that simultaneously trains the neural deformation module along with the embed- ding space used by the retrieval module. This enables our network to learn a deformation-aware embedding space, so that retrieved models are more amenable to match the tar- get after an appropriate deformation. In fact, we use the embedding space to guide the shape pairs used to train the deformation module, so that it invests its capacity in learn- ing deformations between meaningful shape pairs. Further- more, our novel part-aware deformation module can work with inconsistent and diverse part-structures on the source shapes. We demonstrate the benefits of our joint training not only on our novel framework, but also on other state- of-the-art neural deformation modules proposed in recent years. Lastly, we also show that our jointly-trained method outperforms various non-joint baselines.  Our project page can be found [here](https://joint-retrieval-deformation.github.io), and the arXiv version of our paper can be found [here](https://arxiv.org/abs/2101.07889).
```
@inproceedings{uy-joint-cvpr21,
      title = {Joint Learning of 3D Shape Retrieval and Deformation},
      author = {Mikaela Angelina Uy and Vladimir G. Kim and Minhyuk Sung and Noam Aigerman and Siddhartha Chaudhuri and Leonidas Guibas},
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2021}
  }
```

## Data download and preprocessing details
Dataset downloads can be found in the links below. These should be extracted in the project home folder.
1) Raw source shapes are [here](http://download.cs.stanford.edu/orion/joint_embedding_deformation/data_aabb_constraints_keypoint.tar).

2) Processed h5 and pickle files are [here](http://download.cs.stanford.edu/orion/joint_embedding_deformation/generated_datasplits.tar).

3) Targets:
   * \[Optional] (already processed in h5) [point cloud](http://download.cs.stanford.edu/orion/joint_embedding_deformation/data_aabb_all_models.tar)
   * Images: [chair](http://download.cs.stanford.edu/orion/joint_embedding_deformation/partnet_rgb_masks_chair.tar), [table](http://download.cs.stanford.edu/orion/joint_embedding_deformation/partnet_rgb_masks_table.tar), [cabinet](http://download.cs.stanford.edu/orion/joint_embedding_deformation/partnet_rgb_masks_storagefurniture.tar). You also need to modify the correct path for `IMAGE_BASE_DIR` in the image training and evaluation scripts.

4) Automatic segmentation (ComplementMe) 
    * Source shapes are [here](http://download.cs.stanford.edu/orion/joint_embedding_deformation/data_complementme_final.tar).
    * Processed h5 and pickle files are [here](http://download.cs.stanford.edu/orion/joint_embedding_deformation/generated_datasplits_complementme.tar).

For more details on the pre-processing scripts, please take a look at `run_preprocessing.py` and `generate_combined_h5.py`. `run_preprocessing.py` includes the details on how the connectivity constraints and projection matrices are defined. We use the `keypoint_based` constraint to define our source model constraints in the paper. 

The renderer used throughout the project can be found [here](https://github.com/mhsung/libigl-renderer). Please modify the paths, including the input and output directories, accordingly at `global_variables.py` if you want to process your own data.

## Pre-trained Models
The pretrained models for Ours and Ours w/ IDO, which uses our joint training approach can be found [here](). We also included the pretrained models of our structure-aware deformation-only network, which are trained on random source-target pairs used to initialize our joint training.

## Evaluation
Example commands to run the evaluation script are as follows. The flags can be changed as desired. `--mesh_visu` renders the output results into images, remove the flag to disable the rendering. Note that `--category` is the object category and the values should be set to "chair", "table", "storagefurniture" for classes chair, table and cabinet, respectively.

For point clouds:
```
python evaluate.py --logdir=ours_ido_pc_chair/ --dump_dir=dump_ours_ido_pc_chair/ --joint_model=1 --use_connectivity=1 --use_src_encoder_retrieval=1 --category=chair --use_keypoint=1 --mesh_visu=1

python evaluate_recall.py --logdir=ours_ido_pc_chair/ --dump_dir=dump_ours_ido_pc_chair/ --category=chair
```

For images:
```
python evaluate_images.py --logdir=ours_ido_img_chair/ --dump_dir=dump_ours_ido_img_chair/ --joint_model=1 --use_connectivity=1 --category=chair --use_src_encoder_retrieval=1 --use_keypoint=1 --mesh_visu=1

python evaluate_images_recall.py --logdir=ours_ido_img_chair/ --dump_dir=dump_ours_ido_img_chair/ --category=chair
```

## Training
* To train deformation-only networks on random source-target pairs, example commands are as follows:
```
# For point clouds
python train_deformation_final.py --logdir=log/ --dump_dir=dump/ --to_train=1 --use_connectivity=1 --category=chair --use_keypoint=1 --use_symmetry=1

# For images
python train_deformation_images.py --logdir=log/ --dump_dir=dump/ --to_train=1 --use_connectivity=1 --category=storagefurniture --use_keypoint=1 --use_symmetry=1
```
* To train our joint models without IDO (Ours), example commands are as follows:
```
# For point clouds
python train_region_final.py --logdir=log/ --dump_dir=dump/ --to_train=1 --init_deformation=1 --loss_function=regression --distance_function=mahalanobis --use_connectivity=1 --use_src_encoder_retrieval=1 --category=chair --model_init=df_chair_pc/ --selection=retrieval_candidates --use_keypoint=1 --use_symmetry=1

# For images
python train_region_images.py --logdir=log/ --dump_dir=dump/ --to_train=1 --use_connectivity=1 --selection=retrieval_candidates --use_src_encoder_retrieval=1 --category=chair --use_keypoint=1 --use_symmetry=1 --init_deformation=1 --model_init=df_chair_img/
```
* To train our joint models with IDO (Ours w/ IDO), example commands are as follows:
```
# For point clouds
python joint_with_icp.py --logdir=log/ --dump_dir=dump/ --to_train=1 --loss_function=regression --distance_function=mahalanobis --use_connectivity=1 --use_src_encoder_retrieval=1 --category=chair --model_init=df_chair_pc/ --selection=retrieval_candidates --use_keypoint=1 --use_symmetry=1 --init_deformation=1 --use_icp_pp=1 --fitting_loss=l2

# For images
python joint_icp_images.py --logdir=log/ --dump_dir=dump/ --to_train=1 --init_joint=1 --loss_function=regression --distance_function=mahalanobis --use_connectivity=1 --use_src_encoder_retrieval=1 --category=chair --model_init=df_chair_img/ --selection=retrieval_candidates --use_keypoint=1 --use_symmetry=1 --init_deformation=1 --use_icp_pp=1 --fitting_loss=l2
```
Note that our joint training approach is used by setting the flag `--selection=retrieval_candidates=1`.

## Related Work
This work and codebase is related to the following previous work:
* <a href="https://github.com/mikacuy/deformation_aware_embedding" target="_blank">Deformation-Aware Embedding 3D Model Embedding and Retrieval</a> by Uy et al. (ECCV 2020).

## License
This repository is released under MIT License (see LICENSE file for details).
