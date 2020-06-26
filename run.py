import random
import os

from src.cropping import crop_mammogram
from src.optimal_centers import extract_optimal_center
from src.heatmaps import load_model
from src.heatmaps import produce_heatmaps
from src.modeling import load_run_save
from src.constants import MODELMODES

crop_mammogram(
    input_data_folder='data/images',
    exam_list_path='data/exam_list.pkl',
    cropped_exam_list_path='output/cropped_images/cropped_exam_list.pkl',
    output_data_folder='output/cropped_images',
    num_processes=10,
    num_iterations=100,
    buffer_size=50,
)

extract_optimal_center(
    cropped_exam_list_path='output/cropped_images/cropped_exam_list.pkl',
    data_prefix='output/cropped_images',
    output_exam_list_path='output/data.pkl',
    num_processes=10,
)

heatmap_parameters = dict(
    device_type='cpu',
    gpu_number=0,

    patch_size=256,

    stride_fixed=70,
    more_patches=5,
    minibatch_size=100,
    seed=0,

    initial_parameters='models/patch_model.p',
    input_channels=3,
    number_of_classes=4,

    data_file='output/data.pkl',
    original_image_path='output/cropped_images',
    save_heatmap_path=[os.path.join('output/heatmaps', 'heatmap_malignant'),
                       os.path.join('output/heatmaps', 'heatmap_benign')],

    heatmap_type=[0, 1],

    use_hdf5=False
)
random.seed(heatmap_parameters['seed'])
model, device = load_model(heatmap_parameters)
produce_heatmaps(model, device, heatmap_parameters)

model_parameters = {
    "device_type": 'cpu',
    "gpu_number": 0,
    "max_crop_noise": (100, 100),
    "max_crop_size_noise": 100,
    "image_path": 'output/cropped_images',
    "batch_size": 1,
    "seed": 0,
    "augmentation": True,
    "num_epochs": 10,
    "use_heatmaps": True,
    "heatmaps_path": 'output/heatmaps',
    "use_hdf5": False,
    "model_mode": MODELMODES.VIEW_SPLIT,
    "model_path": 'models/imageheatmaps_model.p',
}

load_run_save(
    data_path='output/data.pkl',
    output_path='output/imageheatmaps_predictions.csv',
    parameters=model_parameters,
)
