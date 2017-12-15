# Charrrrtreuse
DS-GA 1006 Capstone Project for Joyce Wu, Raúl Delgado Sánchez and Eduardo Fierro Farah.

## Steps to replicate work:

### 1. Data:

* Download the GDC data transfer API at https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
* Create a manifest by selecting Cases > CANCER_TYPE and Files > Data Type > Tissue Slide Image.
* Download the manifest into ```manifest_file```
* Run the command ```gdc-client download -m manifest_file``` in Terminal

### 2. Data processing:

Note that data tiling and sorting scripts come from https://github.com/ncoudray/DeepPATH/. Please refer to the README within `DeepPATH_code` for the full range of options. 

#### 2.1. Data tiling
Run ```Tiling/0b_tileLoop_deepzoom2.py``` to tile the .svs images into .jpeg images. To replicate this particular project, select the following specifications:

```sh
python -u 0b_tileLoop_deepzoom2.py -s 512 -e 0 -j 28 -f jpeg -B 25 -o out_path "input_path/*/*svs"
```

* `-s 512`: Tile size of 512x512 pixels

* `-e 0`: Zero overlap in pixels for tiles

* `-j 28`: 28 CPU threads

* `-f jpeg`: jpeg files

* `-B 25`: 25% allowed background within a tile.

Replace `out_path` and `input_path` with the path to which the files will be saved and the path to where the original image files reside respectively. 

#### 2.2. Data sorting
Run `Tiling/0d_SortTiles.py` to sort the tiles into train, valid and test datasets. This was run with the following specifications: 

```sh
python 0d_SortTiles.py --SourceFolder="<INPUT_PATH>" --JsonFile="<JSON_FILE_PATH>" --Magnification=20 --MagDiffAllowed=0 --SortingOption=3 --PercentTest=15 --PercentValid=15 --PatientID=12 --nSplit 0
```

`--Magnification=20`: Magnification at which the tiles should be considered (20x)

`--MagDiffAllowed=0`: If the requested magnification does not exist for a given slide, take the nearest existing magnification but only if it is at +/- the amount allowed here (0)

`--SortingOption=3`: Sort according to type of cancer (types of cancer + Solid Tissue Normal)

`--PercentValid=15 --PercentTest=15` The percentage of data to be assigned to the validation and test set. In this case, it will result in a 70 / 15 / 15 % train-valid-test split.

`--PatientID=12` This option makes sure that the tiles corresponding to one patient are either on the test set, valid set or train set, but not divided among these categories.

`--nSplit=0` If nSplit > 0, it overrides the existing PercentTest and PercentTest options, splitting the data into n even categories. 

#### 2.3. Build tile dictionary

Run `Tiling/BuildTileDictionary.py --data <CANCER_TYPE> --path <ROOT_PATH>` to build a dictionary of slides that is used to map each slide to a 2D array of tile paths and the true label. This is used in the `aggregate` function during training and evaluation. `<CANCER_TYPE>` specifies the dataset such as `'Lung'`, `'Breast'`, or `'Kidney'`. `<ROOT_PATH>` points to the directory path for which the sorted tiles folder is stored in.

Note that this code assumes that the sorted tiles are stored in `<ROOT_PATH><CANCER_TYPE>TilesSorted`. If you do not follow this convention, you may need to modify this code.

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



