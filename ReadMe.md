# SSH Mapping IMT-Atlantique Data Challenge 2020

This repository contains codes and sample notebooks for downloading and processing the SSH mapping data challenge.

## Motivation

The goal is to investigate how to best reconstruct sequences of Sea Surface Height (SSH) maps from partial satellite altimetry observations. This data challenge follows an _Observation System Simulation Experiment_ framework: "Real" full SSH are from a numerical simulation with a realistic, high-resolution ocean circulation model: the reference simulation. Satellite observations are simulated by sampling the reference simulation based on realistic orbits of past, existing or future altimetry satellites. A baseline reconstruction method is provided, namely optimal interpolation (see below), and the practical goal of the challenge is:
* to beat this baseline according to scores also described below and in Jupyter notebooks.
* to build a webpage where other teams can dynamically run their method and confront their performance scores to other methods

### Reference simulation

The Nature Run (NR) used in this work corresponds to the NATL60 configuration  (Ajayi et al. 2020 doi:[10.1029/2019JC015827](https://doi.org/10.1029/2019JC015827)) of the NEMO (Nucleus for European Modeling of the Ocean) model. It is one of the most advanced state-of-the-art basin-scale high-resolution (1/60°) simulation available today, whose surface field effective resolution is about 7km.

### Observations

The SSH daily observations include:
* simulations of Topex-Poseidon, Jason 1, Geosat Follow-On, Envisat, and SWOT altimeter data. This nadir altimeters constellation was operating during the 2003-2005 period and is still considered as a historical optimal constellation in terms of spatio-temporal coverage.
* simulations of the upcoming SWOT mission (2021) providing 2D wide-swath observations to the along-track 1D nadir reference constellation. 
All the data (nadir & SWOT) are simulated based on the NATL60 baseline. Realistic observation errors can optionnaly be included in the interpolation.

### Optimal Interpolation (OI)

The DUACS system is an operational production of sea level products for the Marine (CMEMS)
and Climate (C3S) services of the E.U. Copernicus program, on behalf of the CNES french space
agency. It is mainly based on optimal interpolation techniques whose parameters are fully described
in Taburel et al. (2020). 

### Data training & evaluation sequence

All the datasets (NATL60 reference, nadir/SWOT, OI) are provided on the same regular grids with 0.05°x0.05° effective resolution. The dataset covers the period starting from 2012-10-01 to 2013-09-30.

* Regarding the evaluation period, the SSH interpolations will be assessed over the period from 2012-10-22 to 2012-12-02: 42 days, which is equivalent to two SWOT cycles in the SWOT science phase orbit.
* Regarding the learning period, the **reference data** can be used from 2013-01-02 to 2013-09-30. But the reference data between 2012-12-02 and 2013-01-02 should never be used so that any learning period or other method-related-training period can be considered uncorrelated to the evaluation period.
Last, for reconstruction methods that need a spin-up, the **observations** can be used from 2012-10-01 until the beginning of the evaluation period (21 days). This spin-up period is not included in the evaluation

![Data Sequence](figures/DC-data_availability.png)
 
## Quick start with DINAE code

In this github repository, a new end-to-end learning approach based on specifically designed neural networks (NN) for the interpolation problem is providef. The full code to read the data, run the model and display preliminary figures and scores is given. The outputs of the model for the evaluation period are provided in a NetCDF file, used for the post-processing of figures and scores.
You can follow the quickstart guide in [this notebook](https://github.com/CIA-Oceanix/2020a_IMT_SSH_mapping_NATL60/tree/master/notebooks/quickstart.ipynb) or launch it directly from <a href="https://binder.pangeo.io/v2/gh/imt-data-challenges/2020a_IMT_SSH_mapping_NATL60/master?filepath=quickstart.ipynb" target="_blank">binder</a>.

### Architecture of the code

Associated preprints:
- Fixed-point solver: https://arxiv.org/abs/1910.00556
- Gradient-based solvers using automatic differentiation: https://arxiv.org/abs/2006.03653

License: CECILL-C license

Copyright IMT Atlantique/OceaniX, contributor(s) : R. Fablet, 21/03/2020

Contact person: ronan.fablet@imt-atlantique.fr
This software is a computer program whose purpose is to apply deep learning
schemes to dynamical systems and ocean remote sensing data.
This software is governed by the CeCILL-C license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-C
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".
As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.
In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.
The fact that you are presently reading this means that you have had
knowledge of the CeCILL-C license and that you accept its terms.

### DINAE/mods/import_Datasets.py
Import data for OSSE-based experiments

### DINAE/mods/define_Models.py
Architecture of the NN:
* ConvAE.py: 2D-convolutional auto-encoder
* GENN.py: Gibbs-Energy NN

### dinae/mods/FP_solver.py
Train and evaluate AE model for OSSE-based experiments with FP-solver
* def_DINConvAE.py:
* save_Models.py:

### Results

Below is an illustration of the results obtained on the daily velocity SSH field
when interpolating pseudo irregular and noisy observations (top-right panels) corresponding to
along-track nadir with additional pseudo wide-swath SWOT observations built
from an idealized groundtruth (top-left panels) with state-of-the-art optimal interpolation
(bottom-left panels) and the newly proposed end-to-end learning approach (bottom-right panels):

       Nadir+SWOT
:-------------------------:
![Farmers Market Finder Demo](figs/animation_grads_OSSE.gif)

## Download the data
The data is hosted [here](!wget https://s3.eu-central-1.wasabisys.com/melody) with the following directory structure

```
.
|-- data
|   |-- dataset_nadir_0d_swot.nc
|-- ref
|   |-- NATL60-CJM165_GULFSTREAM_ssh_y2013.1y.nc
|-- oi
|   |-- ssh_NATL60_swot_4nadir.nc

```

To start out download the *daily reference, observation and OI* dataset using: 
```shell
wget https://s3.eu-central-1.wasabisys.com/melody/DATA.ncdf -O   "data.ncdf"
wget https://s3.eu-central-1.wasabisys.com/melody/REF.ncdf -O    "ref.ncdf"
wget https://s3.eu-central-1.wasabisys.com/melody/OI.ncdf -O     "oi.ncdf"
```

## Baseline and evaluation

### Baseline
The baseline mapping method is optimal interpolation (OI), whose results are already provided in [here](!wget https://s3.eu-central-1.wasabisys.com/melody/OI.ncdf)
   
### Evaluation

The evaluation of the mapping methods is based on the comparison of the SSH reconstructions with the *reference* dataset. It includes two scores, one based on the Root-Mean-Square Error (RMSE), the other based on Fourier wavenumber spectra. The evaluation notebook [`example_data_eval`](https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60/blob/master/notebooks/example_data_eval.ipynb) implements the computation of these two scores as they could appear in the dynamically updated webpage.

## Data processing

Cross-functional modules are gathered in the `src` directory. They include tools for plots, evaluation, writing and reading NetCDF files.

## Acknowledgement

The structure of this data challenge was to a large extent inspired by the [Boost-SWOT 2020 SSH Mapping Data Challenge](https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60).
