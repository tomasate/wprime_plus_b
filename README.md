# W' + b

[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="left">
  <img width="300" src="https://i.imgur.com/OWhX13O.jpg" />
</p>

Python package for analyzing W' + b in the electron and muon channels. The analysis uses a columnar framework to process input tree-based [NanoAOD](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD) files using the [coffea](https://coffeateam.github.io/coffea/) and [scikit-hep](https://scikit-hep.org) Python libraries.

## Processors

- [TriggerEfficiencyProcessor](processors/trigger_efficiency_processor.py) (trigger): Trigger efficiency processor that applies pre-selection and selection cuts (two bjets + one lepton + MET), and saves numerator and denominator as hist objects in a pickle file. 

- [TTBarControlRegionProcessor](processors/ttbar_processor.py) (ttbar): TTbar Control Region processor that applies pre-selection and selection cuts (two bjets + one lepton + MET), saves unbinned branches as parquet files and cutflow dictionaries as a pickle file.

To test locally first (recommended), can do e.g.:

```bash
python run.py --channel ele --sample TTTo2L2Nu --executor iterative --year 2017 --processor ttbar --nfiles 1 
```
Parquet and pickle files will be saved in the directory specified by the flag `--output_location`

To see a description of all script parameters type:

```bash
python run.py --help
```

General note: [coffea-casa](https://coffea-casa.readthedocs.io/en/latest/cc_user.html) is faster and more convenient, however still somewhat experimental so for large of inputs and/or processors which may require heavier cpu/memory Condor is recommended.


## Scale factors

We use the common json format for scale factors (SF), hence the requirement to install [correctionlib](https://github.com/cms-nanoAOD/correctionlib). The SF themselves can be found in the central [POG repository](https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration), synced once a day with CVMFS: `/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration`. A summary of their content can be found [here](https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/). The SF used in this analysis are (See [corrections](processors/corrections.py)):

* Pileup
* b-tagging
* Electron ID
* Electron Reconstruction
* Muon ID
* Muon Iso
* Muon TriggerIso
 

## Setting up coffea environments

<details><summary>Click to see details</summary>
<p>

#### Install miniconda (if you do not have it already)
In your lxplus area or in your local computer:
```
# download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# run and follow instructions  
bash Miniconda3-latest-Linux-x86_64.sh

# Make sure to choose `yes` for the following one to let the installer initialize Miniconda3
# > Do you wish the installer to initialize Miniconda3
# > by running conda init? [yes|no]
```
Verify the installation is successful by running conda info and check if the paths are pointing to your Miniconda installation. 
If you cannot run conda command, check if you need to add the conda path to your PATH variable in your bashrc/zshrc file, e.g.,
```
export PATH="$HOME/nobackup/miniconda3/bin:$PATH"
```
To disable auto activation of the base environment:
```
conda config --set auto_activate_base false
```

#### Set up a conda environment and install the required packages
```
# create a new conda environment
conda create -n coffea-env python=3.7

# activate the environment
conda activate coffea-env

# install packages
pip install numpy pandas coffea correctionlib pyarrow

# install xrootd
conda install -c conda-forge xrootd
```

</p>
</details>

## Data fileset

The fileset json files that contain a dictionary of the files per sample are in the `data/fileset` directory.

<details><summary>Click to see details</summary>
<p>

#### Re-making the input dataset files with DAS

```
# connect to lxplus with a port forward to access the jupyter notebook server
ssh <your_username>@lxplus.cern.ch localhost:8800 localhost:8800

# create a working directory and clone the repo (if you have not done yet)
git clone https://github.com/deoache/wprime_plus_b

# enable the coffea environment
conda activate coffea-env

# then activate your proxy
voms-proxy-init --voms cms --valid 100:00

# activate cmsset
source /cvmfs/cms.cern.ch/cmsset_default.sh

# open the jupyter notebook on a browser
cd data/fileset/
jupyter notebook --no-browser --port 8800
```

there should be a link looking like `http://localhost:8800/?token=...`, displayed in the output at this point, paste that into your browser.
You should see a jupyter notebook with a directory listing.

Open `filesetDAS.ipynb` and run it. The json files containing the datasets to be run should be saved in the same `data/fileset/` directory.
  
</p>
</details>
