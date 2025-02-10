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
    lp2 = lp + log_likelihood(params, data, cov, z)
    if np.isnan(lp2):
        return -np.inf
    else: 
        return lp2
    
def log_likelihood(params, data, cov, z):
    log10_M, c = params
    M = (10**log10_M) #Msun/h #mass has no h, so we added the h here.

    z = 0.3
    a = 1/(1+z) # scale factor
    
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
    
    Rproj = np.logspace(-2, 2.4, num=1000, base=10) #Mpc/h Projected 2D radii. 
    R3d = np.logspace(-2, 3, num=1000, base=10) #Mpc/h comoving. 3D radii.
    
    # rp in comoving Mpc/h
    Rmin = 0.0323 
    Rmax = 30 
    nbins = 15
    
    # rp1 = rp_
    Redges = np.logspace(np.log10(Rmin), np.log10(Rmax), nbins+1) #Projected radial bin edges
    Redges *= h*(1+z) #Converted to Mpc/h comoving

    """
    Note:
    Minimum Rproj for Sigma(Rproj) must be >= than min(r) of xi(r).
    Maximum Rproj for Sigma(Rproj) must be <= than max(r) of xi(r).
    Thus, the range of values for Rproj must be 
    equal to (or contained in) that of r
    """
    
    # NFW profile
    xi_nfw = ctk.xi.xi_nfw_at_r(R3d, M, c, Omega_m)
    
    # Matter-matter correlation function (matter auto-correlation)
    xi_mm = ctk.xi.xi_mm_at_r(R3d, kh, Pnonlin)
    
     # 2-halo correlation function
    bias_term = ctk.bias.bias_at_M(M, kh, Plin, Omega_m) # Here, P must be linear.
    xi_2halo = ctk.xi.xi_2halo(bias_term, xi_mm)
    
    # Halo-matter correlation function
    xi_hm = ctk.xi.xi_hm(xi_nfw, xi_2halo)
    
    # Sigma (computed from xi_hm)
    Sigma = ctk.deltasigma.Sigma_at_R(Rproj, R3d, xi_hm, M, c, Omega_m) #Sigma
    
    # DeltaSigma (excess surface density)
    # mass = mass/h
    DS_theory = ctk.deltasigma.DeltaSigma_at_R(Rproj, Rproj, Sigma, M, c, Omega_m) #DeltaSigma
    # print(DS_theory)
    ave_DS = ctk.averaging.average_profile_in_bins(Redges, Rproj, DS_theory)
    ave_DS *= h*(1+z)**2 #convert to Msun/pc^2 physical
    model = ave_DS[sel]
        
    scale_sel0 = (rp>=3.0)&(rp<999)
    scale_sel = scale_sel0[sel]
    # covariance
    icov_cut = np.linalg.inv(cov)
    
    icov_cut = icov_cut[sel][scale_sel]
    icov_cut = icov_cut[:,sel][:, scale_sel]
    
    # Difference between data and model and the likelihood.
    data_diff = (data - model)[scale_sel]
    data_likelihood = -0.5*np.dot(data_diff, np.dot(icov_cut, data_diff))
    return data_likelihood

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

############################################################
############################################################
if __name__ == "__main__":
    output_loc = '/global/u2/t/titus/Titus/Lensing/output/'
    nbody_loc = '/global/u2/t/titus/Titus/Lensing/data/'
    lensing_loc = '/global/u2/t/titus/Titus/Lensing/data/McClintock_data/desy1_tamas/'
    filepath = '/pscratch/sd/t/titus/MiniUchuu/mcmc_results/'
    data_path = '/pscratch/sd/t/titus/data/'
    
    # saved_ds_profiles_dic = nbody_loc+"saved_mini_uchuu_profiles_newdata_dic.npy"
    # converted_mini_uchuu_dic = {} # a dictionary for mini Uchuu profiles after adding systematics to them
    # saved_boost_profiles_dic = nbody_loc+"saved_boost_profiles_newdata_dic.npy"
    # boost_dic = {} # boost factor profile computed from M19; to be used as input data in the MCMC
    ########################################################
    
    # Set cosmology (colossus)
    cosmo_params = {'flat': True, 'H0': 67.74, 'Om0': 0.3089, 'Ob0': 0.0486, 'sigma8': 0.8159, 'ns': 0.9667}
    cosmology.addCosmology('MiniUchuu', cosmo_params)
    cosmo = cosmology.setCosmology('MiniUchuu')
    h = cosmo_params['H0']/100 
    
    z=0.3
    
    ##########################################################
    parser = argparse.ArgumentParser(description='Command-line arguments.')
    parser.add_argument('--redshift', type=int, default=0, help='redshift bin')
    parser.add_argument('--start', type=int, default=0, help='starting point of the loop')
    parser.add_argument('--end', type=int, default=4, help='end point of the loop')
    parser.add_argument('--sys_name', type=str, default="_sys0", help='give file name')
    parser.add_argument('--depth', type=str, default="30", help='the cylinder depth')
    args = parser.parse_args()
    i = args.redshift
    start = args.start
    end = args.end
    sys_name = args.sys_name
    cyl_depth = args.depth
    
    ##########################################################
    # Comoputed DS profiles with systematics
    data_ds = np.load(data_path+f'mini_uchuu_ds_cylinder_richness_selection_d{cyl_depth}_no_pecvel_changing_vol_heidi_no_sys_final2.npy', allow_pickle=True).item()
    
    # Mean mass of each bin
    mass = np.load(data_path+f'mini_uchuu_mass_cylinder_richness_selection_d{cyl_depth}_no_pecvel_changing_vol_heidi_final2.npy', allow_pickle=True).item()
    
    # mass of bins
    mean_mass_true0 = mass['l3_z0']
    mean_mass_true1 = mass['l4_z0']
    mean_mass_true2 = mass['l5_z0']
    mean_mass_true3 = mass['l6_z0']

    print('mean mass 1:', f'{mean_mass_true0:e}')
    print('mean mass 2:', f'{mean_mass_true1:e}')
    print('mean mass 3:', f'{mean_mass_true2:e}')
    print('mean mass 4:', f'{mean_mass_true3:e}\n')
    
    #Start by specifying the cosmology (cluster toolkit)
    Omega_b = cosmo_params['Ob0']
    Omega_m = cosmo_params['Om0']
    Omega_cdm = Omega_m - Omega_b
    sigma8 = cosmo_params['sigma8']
    # h = cosmo_params['H0']/100 # McClintock h value
    A_s = 2.1e-9 
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
             'non linear':'halofit',
    }

    #Initialize the cosmology and compute everything
    cosmo_ctk = Class()
    cosmo_ctk.set(params)
    cosmo_ctk.compute()
    
    for j in range(start,end):
        print(f'Richness bin {j+1} in progress...')
        readerfile = filepath+f"cylinder_richness_{sys_name}_mcmc_results_l"+str(j+3)+"_z"+str(i)+".h5"

        
        # Data from McClintock et. al., 2019.
        tamas = np.loadtxt(lensing_loc+"full-unblind-v2-mcal-zmix_y1subtr_l"+str(j+3)+"_z"+str(i)+"_profile.dat")
        dst_cov = np.loadtxt(lensing_loc+"full-unblind-v2-mcal-zmix_y1subtr_l"+str(j+3)+"_z"+str(i)+"_dst_cov.dat")
        boost_data99 = np.loadtxt(lensing_loc+"full-unblind-v2-mcal-zmix_y1clust_l"+str(j+3)+"_z"+str(i)+"_zpdf_boost.dat")[:,1] #Boost factor data
        Be = np.loadtxt(lensing_loc+"full-unblind-v2-mcal-zmix_y1clust_l"+str(j+3)+"_z"+str(i)+"_zpdf_boost.dat")[:,2] #boost error
        boost_cov99 = np.loadtxt(lensing_loc+"full-unblind-v2-mcal-zmix_y1clust_l"+str(j+3)+"_z"+str(i)+"_zpdf_boost_cov.dat")   
        
        
        ##################################################################
        ##################################################################
        # radial bins
        rp = np.array([0.04221139,  0.06651455,  0.10516441,  0.16537565,  0.26069825, 
                       0.41202975,  0.65045543,  1.02563815,  1.61857299,  2.55379112, 
                       4.02731114,  6.34991963, 10.00980414, 15.784786  , 24.87216972])
        sel = rp >= 0.2
        ds0 = data_ds["l"+str(j+3)+"_z"+str(i)] # DeltaSigma profile
        ds = ds0[sel]
        # boost_data = ctk.boostfactors.boost_nfw_at_R(rp, B0_orig, Rs_orig)
        
        # converted_mini_uchuu_dic[lam_z_bin] = ds
        # boost_dic[lam_z_bin] = boost_data

        # print("ds", ds)
        # print("shape", ds.shape)
        
    # np.save(saved_ds_profiles_dic, converted_mini_uchuu_dic)
    # np.save(saved_boost_profiles_dic, boost_dic)
        
        #################################################################
        #################################################################
        # Initial guess for the MCMC
        start_params = np.array([14.60, 5.0]) # mass, concentration.
        nsteps = 10000
        burnin = int(0.1*nsteps)
        sampler = run_mcmc(data = ds, params = start_params, 
                           nwalkers = 32, nsteps = nsteps, burnin = burnin, 
                           cov = dst_cov, z=z, readerfile = readerfile)

    
    # Added the factor of h*(1+z) to all Rs in boost data, boost cov, Redges, except boost model, not radial cut.
    # mpirun -np 8 python cylinder_richness_phys_units_large_scale.py --redshift 0 --start 0 --end 4 --sys_name _mini_uchuu_cylinder_richness_large_scale_depth30_final02 --depth 30
    
    # mpirun -np 8 python cylinder_richness_phys_units_large_scale.py --redshift 0 --start 0 --end 4 --sys_name _mini_uchuu_cylinder_richness_large_scale_depth30_final_final --depth 30

