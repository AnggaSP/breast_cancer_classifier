import os, shutil

from flask import Flask, jsonify, request
from src.cropping.crop_single import crop_single_mammogram
from src.optimal_centers.get_optimal_center_single import get_optimal_center_single
from src.heatmaps.run_producer_single import produce_heatmaps
from src.modeling.run_model_single import run
from werkzeug import utils

# Metadata
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'input/'
app.config['OUTPUT_FOLDER'] = 'output/'
ALLOWED_EXTENSIONS = {'png'}

# Define the url for the API endpoint
@app.route('/classify', methods=['POST'])
def classify():
    # Check if image is defined in the request
    if 'image' not in request.files:
        # Inform the user that the image is not defined
        message = {
        'status': 400,
        'message': 'Image is not defined'
        }

        # Remove file before returning
        remove_file()

        return jsonify(message), 400

    # Then save it into a variable
    image = request.files['image']
    view = request.form['view']

    # Additional sanity check before we proceed
    if (image is not None and view is not None and image.filename != '' and
            allowed_file(image.filename)):
        imagename = utils.secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], imagename))

        # First phase: Crop the mammogram to the area of interest
        crop_single_mammogram(
            mammogram_path=os.path.join(app.config['UPLOAD_FOLDER'], imagename),
            view=str(view),
            horizontal_flip="NO",
            cropped_mammogram_path=os.path.join(app.config['OUTPUT_FOLDER'], 'cropped.png'),
            metadata_path=os.path.join(app.config['OUTPUT_FOLDER'], 'cropped_metadata.pkl'),
            num_iterations=100,
            buffer_size=50
        )

        # Second phase: Optimal center of the mammogram
        get_optimal_center_single(
            cropped_mammogram_path=os.path.join(app.config['OUTPUT_FOLDER'], 'cropped.png'),
            metadata_path=os.path.join(app.config['OUTPUT_FOLDER'], 'cropped_metadata.pkl')
        )

        # Third phase: Heatmap for benign and malignant cancer
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

            cropped_mammogram_path=os.path.join(app.config['OUTPUT_FOLDER'], 'cropped.png'),
            metadata_path=os.path.join(app.config['OUTPUT_FOLDER'], 'cropped_metadata.pkl'),
            heatmap_path_malignant=os.path.join(app.config['OUTPUT_FOLDER'], 'malignant_heatmap.hdf5'),
            heatmap_path_benign=os.path.join(app.config['OUTPUT_FOLDER'], 'benign_heatmap.hdf5'),

            heatmap_type=[0, 1],

            use_hdf5=False
        )
        produce_heatmaps(heatmap_parameters)

        # Fourth and final phase: Run the model and get the inference
        model_parameters = dict(
            view=str(view),
            model_path='models/ImageHeatmaps__ModeImage_weights.p',
            cropped_mammogram_path=os.path.join(app.config['OUTPUT_FOLDER'], 'cropped.png'),
            metadata_path=os.path.join(app.config['OUTPUT_FOLDER'], 'cropped_metadata.pkl'),
            device_type='cpu',
            gpu_number=0,
            max_crop_noise=(100, 100),
            max_crop_size_noise=100,
            batch_size=1,
            seed=0,
            augmentation=True,
            num_epochs=10,
            use_heatmaps=True,
            heatmap_path_benign=os.path.join(app.config['OUTPUT_FOLDER'], 'benign_heatmap.hdf5'),
            heatmap_path_malignant=os.path.join(app.config['OUTPUT_FOLDER'], 'malignant_heatmap.hdf5'),
            use_hdf5=False
        )
        result = run(model_parameters)

        # Map the data into dict with metadata for client
        message = {
        'status': 200,
        'message': 'OK',
        'result': result
        }

        # Remove file before returning
        remove_file()

        # Return it to the caller as JSON
        return jsonify(message)
    else:
        # Inform the user that the param is incorrect
        message = {
        'status': 400,
        'message': 'Parameter is incorrect'
        }

        # Remove file before returning
        remove_file()

        return jsonify(message), 400
        
# Allowed file extension to be uploaded in this endpoint
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Remove all the temporary files
def remove_file():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    for filename in os.listdir(app.config['OUTPUT_FOLDER']):
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

app.run(host="127.0.0.1", port=5000, debug=True)
