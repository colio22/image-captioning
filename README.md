# Re-Implementation of the Global Enhanced Transformer for Image Captioning

This project seeks to re-implement the work of Ji et al. in creating a transformer-based image captioning model that understands global context of given images. 
The goal of the project is to reproduce their CIDEr scores within 5 points.

## Setup Instructions

This project depends on external dataset files being present on the local machine. For this project, no internal image analysis or feature extraction was performed.
Instead, previously extracted features from the well-known COCO dataset are used. These pre-extracted features are provided at https://github.com/aimagelab/meshed-memory-transformer.

Before running the code, download the 53 GB HDF5 file containing the features from [coco_detections.hdf5](https://ailb-web.ing.unimore.it/publicfiles/drive/show-control-and-tell/coco_detections.hdf5) 
and store it in a place where they can easily be accessed by a machine with access to GPU. Placing the file on Google Drive is highly recommended so that Google Colab can be used in conjunction.

A zipped file containing the annotations for the extracted features must also be downloaded from [annotations.zip](https://ailb-web.ing.unimore.it/publicfiles/drive/meshed-memory-transformer/annotations.zip).
Extract the zipped file and place them in the same location as the HDF5 feature file.

Finally, ensure that this entire repository has been uploaded to the root level of your runtime (not inside of any other folders). The included Jupyter notebook that runs the project expects the repo to already 
be present before it executes.

The expected file tree in your runtime should look similar to the following before running the FinalProject.ipynb notebook:
```
|- drive (Google Drive mount)
|  |- ... (path to dataset files)
|  |  |- coco_detections.hdf5
|  |  |- annotations/
|- image_captioning (uploaded copy of this repo)
```

## Train a New Model

A Jupyter Notebook is included in the root level of the repository that will train and evaluate the model in a Google Colab GPU-connected runtime. Be aware that this process could take several hours.
The only change that will have to be made to use the notebook is to update the ```--features_path```, ```--annotation_folder```, and ```--save_path``` arguments in cells 3 and 4 to point 
to the Google Drive location where you stored the needed data files during setup.

To run the project manually without using the provided notebook, take the following steps:
1. Clone the project to a location where it can easily access the data files from setup.
2. Install the needed dependencies: ``` pip install -r requirements.txt```
3. Run the ```run_project.py``` script. The script accepts the following arguments:
   | Argument | Description |
   |----------|-------------|
   | --features_path | Path to the coco_detections.hdf5 file containing extracted features. REQUIRED |
   | --annotation_folder | Path to the annotations folder from setup. REQUIRED |
   | --exp_name | Name of the experiment. Trained models will be saved as ```<exp_name>.pth```. |
   | --save_path | Path to the directory in which to store generated files. Google Drive locations are recommended to ensure permanence of data. A ```.pth``` model file and a ```.json``` file of generated captions will be created in this location. |
   | --load_model | Path to existing model, if any. If this argument is used, training will be skipped and the provided model will be evaluated. If not used, a new model will be trained and evaluated. |
   | --batch_size | Batch size to use for training. Defaults to 50. |

   To train a fresh model, the following command is recommended:
   ```
   python run_project.py --batch_size 50 \
                         --features_path "<Path/to/coco_detections.hdf5>" \
                         --annotation_folder "<Path/to/annotations>" \
                         --save_path "<Permanent/file/location>"
   ```

## Evaluate an Existing Model

The repository contains an existing model that has already been trained at ```saved_models/GET.pth```. Training can be skipped
and the model can be tested directly using the following command:

```
python run_project.py --batch_size 50 \
                      --features_path "<Path/to/coco_detections.hdf5>" \
                      --annotation_folder "<Path/to/annotations>" \
                      --save_path "<Permanent/file/location>" \
                      --load_model "saved_models/GET.pth"
```

Note that testing will still take a long time. This is because in addition to validating loss, the test function also generates a
string caption for every item in the first two batches. After batch 0 and batch 1 complete, the process will go much faster.
