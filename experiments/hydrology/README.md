# MC-LSTM for Rainfall-Runoff Modeling

For the hydrology experiment, we added the MC-LSTM to the [neuralHydrology](https://github.com/neuralhydrology/neuralhydrology) Python library and used this library for the model training. To re-run the hydrology experiments, follow the step by step instructions below.

1. Clone or download the neuralHydrology library
2. Setup Python environment
3. Download CAMELS US dataset and extended Maurer forcings
4. Download config file and basin list
5. Adapt config file
6. Train a model
7. Evaluate a model
8. Ablation study

For the paper, we trained 10 model repetitions with different random seeds. In the main paper, we report the results derived from using the ensemble mean prediction. Appendix B.4.6 lists the results of the (average) single model performance in Table B.7.

## 1. Clone or download the neuralHydrology library

Download instructions (and documentation) can be found [here](https://neuralhydrology.readthedocs.io/en/latest/usage/quickstart.html).

## 2. Setup Python environment

We used the conda environment as specified in the neuralHydrology library to run the experiments. Speficially, we used the following [conda environment file](https://github.com/neuralhydrology/neuralhydrology/blob/master/environments/environment_cuda10_2.yml). If you prefer no to use Miniconda/Anaconda, make sure to install all dependencies as listed in this file. If you have Miniconda/Anaconda installedm you can create an environment from this file by:

```bash
conda env create -f environment_cuda10_2.yml
```

## 3. Download CAMELS US dataset and extended Maurer forcings

We used the CAMELS US dataset with the extended Maurer forcings for this experiment. 

1. Download the base CAMELS US dataset "CAMELS time series meteorology, observed flow, meta data (.zip)" from [this site](https://ral.ucar.edu/solutions/products/camels): direct link to [.zip file](https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_timeseries_v1p2_metForcing_obsFlow.zip)
2. Extract the `basin_dataset_public_v1p2` folder to your local system. This directory will be referred to as the "data directory" (or `data_dir`).
3. Download the "CAMELS Attributes (.zip)" for the same site of use [this link](https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/camels_attributes_v2.0.zip).
4. Extract the folder called `camels_attributes_v2.0` and place it within the CAMELS US data dir root directory.
5. Download the extended Maurer forcings from [here](https://www.hydroshare.org/resource/17c896843cf940339c3c3496d0c1c077/) of use [this direct file link](https://www.hydroshare.org/resource/17c896843cf940339c3c3496d0c1c077/data/contents/maurer_extended.zip).
6. Extract the folder called `maurer_extended` into the `basin_mean_forcing` folder that can be found in the root directory of the CAMELS US dataset.

## 4. Download config file and basin list

The neuralHydrology library uses yaml files to specify the run configurations (see the official [documentation](https://neuralhydrology.readthedocs.io/en/latest/index.html) for details). The config for training a single MC-LSTM with a random seed can be found [here (mclstm_run_config.yml)](mclstm_run_config.yml).
Additionally, you will need a file that lists all basins (rivers) from the CAMELS US dataset that we consider in this study. You can find the file [here (447_basin_file.txt)](447_basin_file.txt).

## 5. Adapt config file

The only things that have to be modified are the paths to a) the CAMELS US dataset and b) the basin file. Make sure to adapt the config argument `data_dir` to point to your local copy of the CAMELS US dataset and `train_basin_file`, `validation_basin_file`, `test_basin_file` to point to the file with the 447 basin ids (i.e. 447_basin_file.txt).

## 6. Train a model

Depending on if you have the neuralHydrology library installed or just downloaded/cloned locally you can start training a model by

```bash
nh-run train --config-file /path/to/mclstm_run_config.py
```
if you have the library installed, or

```bash
python neuralhydrology/nh_run.py train --config-file /path/to/mclstm_run_config.py
```
from within the neuralHydrology root directory if you don't have the library installed. You can append `--gpu-id` followed by an integer to specify a specific GPU for model trainnig (otherwise `cuda:0`, as specified in the config, will be used).

Training a model will create a run directory (under the `run_dir` config argument), which includes model checkpoint files, tensorboard log and logging files.

## 7. Evaluate a model

Give the run directory from the previous step, we can evaluate the trained model on the test period by

```bash
nh-run evaluate --run-directory /path/to/run-directory
```
if you have the library installed, or

```bash
python neuralhydrology/nh_run.py evaluate --run-directory /path/to/run-directory
```
from within the neuralHydrology root directory if you don't have the library installed. Afterwards, you can find the test results within the run directory at `test/model_epoch030/test_results.p`. The pickle file contains the simulations and observations for each river (basin id), as well as the NSE values. To compute other metrics, make use of the functions in `neuralhydrology/evaluation/metrics.py`. See the [documentation](https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.evaluation.metrics.html) for more details.

## 8. Ablation study

For the ablation study reported in the appendix, substitute the `mclstm.py` file content in the modelzoo of the neuralHydrology library (`neuralhydrology/modelzoo/mclstm.py`) with the content of the `mclstm_ablation.py` file. The different model configurations can then be defined in the config by:

- `mclstm_i_normaliser: sigmoid`: Uses a standard sigmoid function in the input gate instead of the normalised sigmoid (i.e. breaking mass conservation in the input gate).
- `mclstm_r_normaliser: linear`: Removes the activation function from the redistribution matrix (i.e. breaking mass conservation in the redistribution process).
- `subtract_outgoing_mass: False`: Does not remove outgoing mass from the system (i.e. breaking mass conservation in the output gate).