#--------------- Packages ---------------#
import os
import sys
import emcee
import fitsio
import Corrfunc
import numpy as np
from classy import Class
import cluster_toolkit as ctk
from astropy.table import Table
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
# from colossus.halo import concentration
from colossus.cosmology import cosmology
from scipy.stats import norm#, multivariate_normal
from colossus.halo import profile_composite, concentration

# from cluster_toolkit import miscentering
import numdifftools as nd
import argparse
from schwimmbad import MPIPool 

#######################################################
#######################################################
def log_flat_prior(args):
    log10_M, c = args
    if 12.0<log10_M<16.0 and 2.0<c<10.0: #change from mvir to log_mvir
        return 0.0
    return -np.inf

def log_probability(params, data, cov, z):
    lp = log_flat_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    lp2 = lp + log_likelihood(params, data, cov, z)[0]
    if np.isnan(lp2):
        return -np.inf
    else: 
        return lp2

    
def log_likelihood(params, data, cov, z):
    log10_M, c = params
    M = (10**log10_M) #Msun/h #mass has no h, so we added the h here.

    a = 1/(1+z) # scale factor
    Rproj = np.logspace(-2, 2.4, num=1000, base=10) #Mpc/h Projected 2D radii. 
    R3d = np.logspace(-2, 3, num=1000, base=10) #Mpc/h comoving. 3D radii.
    
    # McClintock radial bins
    Rmin = 0.0323
    Rmax = 30
    nbins = 15
    Redges = np.logspace(np.log10(Rmin), np.log10(Rmax), nbins+1) #Projected radial bin edges
    Redges *= h*(1+z)

    """
    Note:
    Minimum Rproj for Sigma(Rproj) must be >= than min(r) of xi(r).
    Maximum Rproj for Sigma(Rproj) must be <= than max(r) of xi(r).
    Thus, the range of values for Rproj must be 
    equal to (or contained in) that of r
    """
    
    #Specify k and z
    # k = np.logspace(-5, 3, num=4000) #Mpc^-1 comoving
    k = np.logspace(-5, np.log10(k_max), num=4000) #Mpc^-1 comoving
    # Power spectrum
    Pnonlin = np.array([cosmo_ctk.pk(ki, z) for ki in k])#*h**3  #Mpc^3/h^3 comoving
    Plin = np.array([cosmo_ctk.pk_lin(ki, z) for ki in k])#*h**3  #Mpc^3/h^3 comoving
    kh = k/h #h/Mpc comoving
    # k /= h #h/Mpc comoving
    #P(k) are in Mpc^3/h^3 comoving
    #Thus, you will need to convert these to h/Mpc and (Mpc/h)^3 to use in the toolkit.
    Plin *= h**3
    Pnonlin *= h**3

    # NFW profile
    xi_nfw = ctk.xi.xi_nfw_at_r(R3d, M, c, Omega_m, delta=200) #delta=200 by default

    # # Matter-matter correlation function (matter auto-correlation)
    xi_mm = ctk.xi.xi_mm_at_r(R3d, kh, Pnonlin)

    # 2-halo correlation function
    bias_term = ctk.bias.bias_at_M(M, kh, Plin, Omega_m, delta=200) # Here, P must be linear. #delta=200 by default
    xi_2halo = ctk.xi.xi_2halo(bias_term, xi_mm)

    # Halo-matter correlation function
    xi_hm = ctk.xi.xi_hm(xi_nfw, xi_2halo)

    # Sigma (computed from xi_hm)
    Sigma = ctk.deltasigma.Sigma_at_R(Rproj, R3d, xi_hm, M, c, Omega_m, delta=200) #delta=200 by default
    # Sigma = ctk.deltasigma.Sigma_nfw_at_R(Rproj, M, c, Omega_m, delta=200)

    # DeltaSigma (excess surface density)
    DS = ctk.deltasigma.DeltaSigma_at_R(Rproj, Rproj, Sigma, M, c, Omega_m, delta=200) #delta=200 by default
    ave_DS = ctk.averaging.average_profile_in_bins(Redges, Rproj, DS)
    
    model = ave_DS[sel]*h*(1+z)**2 #convert to Msun/pc^2 physical
        
    # covariance
    icov_cut = np.linalg.inv(cov)
    
    icov_cut = icov_cut[sel]
    icov_cut = icov_cut[:,sel]
    
    # Difference between data and model and the likelihood.
    data_diff = data - model
    data_likelihood = -0.5*np.dot(data_diff, np.dot(icov_cut, data_diff))
    return data_likelihood, model

def run_mcmc(data, params, nwalkers, nsteps, burnin, cov, z, readerfile): 
    init_pts = params
    ndim = len(init_pts) #number of params we want to calibrate

    # MPI parallelization
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    
        # # Set up the backend
        if not os.path.isfile(readerfile):
            backend = emcee.backends.HDFBackend(readerfile)
            backend.reset(nwalkers, ndim)
            pos = [init_pts + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
        else:
            backend = emcee.backends.HDFBackend(readerfile)
            pos = None

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(np.array(data), cov, z), 
                                        backend=backend)
        sampler.run_mcmc(pos, nsteps, progress=True);

    af = sampler.acceptance_fraction
    print("Mean acceptance fraction:", np.mean(af))
    return sampler

if __name__ == "__main__":
    # Set cosmology
    cosmo_params = {'flat':True, 'H0':70, 'Om0':0.286, 'Ob0':0.047, 'sigma8':0.82, 'ns':0.96}
    cosmology.addCosmology('cardinal', cosmo_params)
    cosmo = cosmology.setCosmology('cardinal')
    h = cosmo_params['H0']/100

    ## Cluster toolkit cosmology
    #Begin by specifying the cosmology
    Omega_b = cosmo_params['Ob0']
    Omega_m = cosmo_params['Om0']
    Omega_cdm = Omega_m - Omega_b
    sigma8 = cosmo_params['sigma8']
    h = cosmo_params['H0']/100 # McClintock h value
    # A_s = 2.1e-9 #np.exp(3.064)/1e10 NB: ln(1e10*As)=3.064
    n_s = cosmo_params['ns']
    # z = 0.3

    #Create a params dictionary
    #Need to specify the max wavenumber
    k_max = 1000 #UNITS: 1/Mpc
    params = {
             'output':'mPk',
             'h':h,
             'sigma8':sigma8,
             'n_s':n_s,
             'Omega_b':Omega_b,
             'Omega_cdm':Omega_cdm,
             'P_k_max_1/Mpc':k_max,
             'z_max_pk':1.0, #Default value is 10
             'non linear':'halofit'
    }

    #Initialize the cosmology and compute everything
    cosmo_ctk = Class()
    cosmo_ctk.set(params)
    cosmo_ctk.compute()
    
    ###############################################################
    ###############################################################
    filepath = "/global/u2/t/titus/Titus/Lensing/codes/notebooks/cardinal/mcmc_results/"
    data_path = "/global/u2/t/titus/Titus/Lensing/data/"
    cov_path = '/global/u2/t/titus/Titus/Lensing/data/McClintock_data/desy1_tamas/'
    
    # Radial bins
    Rmin_phys_mpc = 0.0323
    Rmax_phys_mpc = 30
    nbins_phys_mpc = 15
    lnrp_bins_phys_mpc = np.linspace(np.log(Rmin_phys_mpc), np.log(Rmax_phys_mpc), nbins_phys_mpc+1)
    rp_bins_phys_mpc = np.exp(lnrp_bins_phys_mpc)
    rp_phys_mpc = np.sqrt(rp_bins_phys_mpc[:-1]*rp_bins_phys_mpc[1:])
    rp = rp_phys_mpc
    sel = rp > 0.1
    
    ### Load the halos with the DS and Sigma profiles
    halos_ds =  Table(np.load(data_path+"stacked_DS_halo_run.npy", allow_pickle=True))
    
    ###############################################################
    ###############################################################
    parser = argparse.ArgumentParser(description='Command-line arguments')
    parser.add_argument('--run_name', type=str, default="_test_run0", help='add to file name')
    
    args = parser.parse_args()
    run_name = args.run_name
    
    z_lower = [0.2, 0.2, 0.2, 0.2, 0.35, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5, 0.5]
    z_upper = [0.35, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7]
    lambda_lower = [20, 30, 45, 60, 20, 30, 45, 60, 20, 30, 45, 60]
    lambda_upper = [30, 45, 60, 500, 30, 45, 60, 500, 30, 45, 60, 500]
    l_list = [3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6]
    z_list = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    
    for i in range(len(halos_ds)):
        readerfile = filepath+f"cardinal_mcmc{run_name}_{z_lower[i]}_{z_upper[i]}_{lambda_lower[i]}_{lambda_upper[i]}.h5"
        cov_file = cov_path+f"full-unblind-v2-mcal-zmix_y1subtr_l"+str(l_list[i])+"_z"+str(z_list[i])+"_dst_cov.dat"
        cov = np.loadtxt(cov_file)

        log10_M = halos_ds["Mvir"][i]
        z = halos_ds["Redshift"][i]
        
        #######################
        c = halos_ds["cvir"][i]
        true_params = [log10_M, c]
        start = [14.20, 3.9] #initial guess, starting point for MCMC walkers.
        ds = log_likelihood(params = true_params, data = np.zeros(len(rp[sel])), cov = cov, z = z)[1]
        sampler = run_mcmc(data = ds, params = start, nwalkers = 32, nsteps = 10000, 
                           burnin = 1000, cov = cov, z = z, readerfile = readerfile)

# mpirun -np 8 python cardinal_ctk_halorun_m19_cov.py --run_name _testrun_ctk_halorun_modelasdata_m19_cov
# mpirun -np 8 python cardinal_ctk_halorun_m19_cov.py --run_name _fullrun_ctk_halorun_modelasdata_m19_cov


