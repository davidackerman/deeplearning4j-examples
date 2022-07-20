from funlib.geometry import Coordinate
from dacapo.experiments.architectures import CNNectomeUNetConfig
from dacapo.experiments.tasks.distance_task_config import DistanceTaskConfig

import torch

input_voxel_size = (16, 16, 16)
output_voxel_size = (8, 8, 8)

channels = [
    "ecs",  # extra cellular space
    "plasma_membrane",
    "mito",
    "mito_membrane",
    "vesicle",
    "vesicle_membrane",
    "mvb",  # endosomes
    "mvb_membrane",
    "er",
    "er_membrane",
    "eres",
    "nucleus",
    "microtubules",
    "microtubules_out",
]

architecture_config = CNNectomeUNetConfig(
    name="CellMapArchitecture",
    input_shape=Coordinate(
        216, 216, 216
    ),  # can be changed
    eval_shape_increase=Coordinate(
        72, 72, 72
    ),  # can be changed
    fmaps_in=1,
    num_fmaps=12,
    fmaps_out=72,
    fmap_inc_factor=6,
    downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
    constant_upsample=True,
    upsample_factors=[(2, 2, 2)],
)

task_config = DistanceTaskConfig(
    name="DistancePrediction",
    # important
    channels=channels,
    scale_factor=50,  # target = tanh(distance / scale)
    # training
    mask_distances=True,
    # evaluation
    clip_distance=50,
    tol_distance=10,
)

# create backbone from config
architecture = architecture_config.architecture_type(architecture_config)

# initialize task from config
task = task_config.task_type(task_config)

# adding final layers/activations to create the model
model = task.create_model(architecture)


path_to_weights = "/nrs/cellmap/pattonw/crop_num_experiment/scratch_distances_many_all_many_8nm_upsample-unet_default__2/checkpoints/mito__f1_score"
weights = torch.load(path_to_weights, map_location="cpu")
model.load_state_dict(weights["model"])

import numpy as np
x = torch.rand((1, 1, 216,216,216))

# set model to input mode
model.eval()

# Export the model
torch.onnx.export(model,                     # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "cellmap_model.onnx",      # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}},
#verbose=True,
)
