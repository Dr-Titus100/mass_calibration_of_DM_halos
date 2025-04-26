# Statistical and Systematic Uncertainties in Weak Lensing Mass Calibration for Optically-Selected Galaxy Clusters.
* Authors: Titus Nyarko Nde (Boise State University, Boise, ID) and Hao-Yi Wu (Southern Methodist University, Dallas, TX)

## Abstract
In this paper, we present an approach to weak lensing mass calibration by incorporating systematic effects into cosmological simulations, addressing a common shortfall in such simulations. We consider three main systematic effects: miscentering, dilution effect, and projection effects, using data from the Mini-Uchuu and Cardinal simulations. We bin the data by richness and redshift. Our analysis reveals that using 5\% of the data as covariance (which mimics future high signal-to-noise data) yields the tightest constraints on mass and concentration. Also, lowering the concentration yields a similar effect in the density profile as the miscentering and dilution effects, which are small-scale effects. We demonstrate that by systematically incorporating these effects into simulations, we can accurately retrieve the actual parameters from observational data, enhancing the fidelity and applicability of simulations in cosmological research. Furthermore, our exploration of small and large radial scales reveals the differential impact of systematic effects, with the small-scale fits exhibiting better constraining power and agreement with the data. This is because the projection effect is a predominantly large-scale effect. We also found that, generally, the mass bias diminishes with the increase in redshift bins. However, within the same richness bin, the mass bias increases with an increase in the redshift bin. The insights gained from this study contribute to our enhanced understanding of the systematic effects and how to factor them into existing analytic models and simulations. This will allow for more accurate analyses of the latest data releases from astronomical surveys.

## How to Use Code
Note: I assume the user of this repo is already a Python user and at least has either Anaconda or Miniconda or Miniforge or Mamba installed. The user should have access to at least Jupyter Notebook or Jupyterlab. You can equally run the code using Visual studio code or Pycharm or Google colab. 

To use the code in this repo follow the following steps.
* Clone the repo using the following command.

```
git clone https://github.com/Dr-Titus100/mass_calibration_of_DM_halos.git
```
You may also fork the repo if interested.


* Install packages. All the packages used in this project and their respective versions are listed in the `requirements.txt` file. Activate your Python environment and run the following command in the terminal to install all the packages at once. Make sure you are in the right directory or provide the path to the `requirements.txt` file. 

```
pip install -r requirements.txt
```

Note: You may create a new Python environment to run this project. This ensures your existing projects are uninterrupted. To create a new Python environment use the following command:

```
conda create --name <env_name> python=<version>
```

## Alternative Approach
You can create a conda/mamba environment and install all necessary packages in one step by creating a `.yml` file. The `environment.yml` is provided in the same location as the `requirements.txt` file. This method is advantageous because you can list all the packages and the versions and also specifiy whether you want to install the package using `pip` or from the `conda-forge` channel. Run the command below to create a Python environment and install all necessary packages at once.

```
conda env create -f environment.yml
```
or
```
mamba env create -f environment.yml
```
if you are using mamba instead of conda. This will create a conda environment named `mass_cal`. Make sure you are in the same directory as the `environment.yml` file or provide the full path to it. Use the code below to activate your conda environment.

```
conda activate mass_cal
```

## Submitting Jobs
The MCMC code takes a long time to run. Hence, we do the computation by submitting a job on the Borah Super Computer at Boise State University. The files are in the repository `submitting_jobs`. It is advisable to run the `.py` file by submitting a job on a cluster. I have uploaded sample bash scripts in that repository, which I used to submit my job on the Borah Cluster. To submit a job, simply run the following command on a cluster node (make sure you are not on the login node).

```
sbatch cylinder_richness.sh
```
There are different files, so we can edit the file name as needed.

I have provided further comments in the `.sh` file. Those comments will be useful for anyone who wants to run the Python script. 

* If a module is missing, then you have to add a line in the `.sh` file to load the module.
```
module load <Path/to/module>
```





