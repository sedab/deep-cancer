# Charrrrtreuse
DS-GA 1006 Capstone Project for Joyce Wu, Raúl Delgado Sánchez and Eduardo Fierro Farah.

## Steps to replicate work:

### 1. Data:

* Download the GDC data transfer API at https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
* Create a manifest by selecting Cases > CANCER_TYPE and Files > Data Type > Tissue Slide Image.
* Download the manifest into ```manifest_file```
* Run the command ```gdc-client download -m manifest_file``` in Terminal

### 2. Data processing:

* Run ```Tiling/0b_tileLoop_deepzoom2.py``` to tile the .svs images into .jpeg images. To replicate this particular project, select the following specifications: 

```sh
python -u 0b_tileLoop_deepzoom2.py -s 512 -e 0 -j 28 -f jpeg -B 25 -o out_path "input_path/*/*svs"
```

This is, tile size of of 512x512 pixels, no overlap, 28 threads, jpeg files and 25% aloud background per tile. Also, replace "out_path" and "input_path" with the paths where the files will be saved and where the original images are respectively. 

* Run ```Tiling/0d_SortTiles.py``` to sort the tiles into train, valid and test datasets. This was run with the following specifications: 

```sh
python 0d_SortTiles.py --SourceFolder="input_path" --JsonFile="input_path_json/JSON_NAME.json" --Magnification=20 --MagDiffAllowed=0 --SortingOption=3 --PercentTest=15 --PercentValid=15 --PatientID=12 --nSplit 0
```

In this case, the option Magnification refers to the magnification at which the tiles should be considerted, and in this case with no margin where the magnification requested is not available (MagDiffAllowed 0). The Sorting option selected, 3, referts to split according to the type of cancer: (LUSC, LUAD, or Nomal Tissue). The split was specified as 70%-15%-15% (train-valid-test). PatientID refers to the number of characters from each image refers to a unique patient identifier; the code makes sure that the tiles corresponding to one patient are either on the test set, valid set or train set, but not divided among these categories. Finally, when nSplit>0, it overrides the existing PercentTest and PercentTest options, splitting the data into even n categories. 

* Run ```Tiling/BuildTileDictionary.py``` to build a dictionary of slides that is used to map each slide to a 2D array of tile paths and the true label. This is used in the `aggregate` function during training and evaluation. In this case, the only option available is "--data", and takes either (Lung/Breast/Kidney) and --file_path, pointing to the path where the tiles are stored. 

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



