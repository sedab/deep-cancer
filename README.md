# Charrrrtreuse
DS-GA 1006 Capstone Project for Joyce Wu, Raúl Delgado Sánchez and Eduardo Fierro

## Steps to replicate work:

### 1. Data:

* Download the GDC data transfer API at https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
* Create a manifest by selecting Cases > CANCER_TYPE and Files > Data Type > Tissue Slide Image.
* Download the manifest into ```manifest_file```
* Run the command ```gdc-client download -m manifest_file``` in Terminal

### 2. Data processing:

* Run ```utils/0b_tileLoop_deepzoom2.py``` to tile the .svs images into .jpeg images
* Run ```utils/0d_SortTiles.py``` to sort the tiles into train, valid and test datasets
* Run ```utils/BuildTileDictionary.py``` to build a dictionary of slides that is used to map each slide to a 2D array of tile paths and the true label. This is used in the `aggregate` function during training and evaluation.

### 4. Train model:

* Run ```train.py``` to train with our CNN architecture, or run ```train_inception.py``` to train Google's Inception V3.
* sbatch files ```run_job.sh``` and ```run_job_inception.sh``` are example scripts to submit GPU jobs for running ```train.py``` and ```train_inception.py``` respectively.

### 5. Test model:

* Run ```test.py``` to evaluate a specific model on the test data, ```run_test.sh``` is the associated sbatch file.

## Additional resources:

### iPython Notebooks

* ```100RandomExamples.ipynb``` visualizes of 100 random examples of tiles in the datasets
* ```Final evaluation and viz.ipynb``` provides code for visualizing the output prediction of a model, and also for evaluating a model on the test set on CPU
* ```LungJsonDescription.ipynb``` explores the potential of metadata that can be used as extra information for training
* ```new_transforms_examples.ipynb``` visualizes a few examples of the data augmentation used for training. One can tune the data augmentation here.

### Progress and documentation

* Capstone Assignment 2.pdf, Capstone 11-9 Memo.pdf, Capstone 11-16 Update.pdf, Capstone 11-30 Update.pdf	are our submitted assignments for our Capstone course.



