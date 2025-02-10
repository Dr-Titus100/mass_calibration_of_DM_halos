#!/usr/bin/env python
import timeit
start = timeit.default_timer()
start_master = start * 1
import numpy as np
import pandas as pd
from astropy.io import fits
import os
import sys
import glob
import yaml
from concurrent.futures import ProcessPoolExecutor
import shutil
from astropy.table import Table, vstack
from numba import njit

#### my functions ####
# python make_gal_cat_Heidi_fast.py path/yml/file
# python make_gal_cat_Heidi_fast_final.py mini_uchuu_fid_hod.yml
# sys.path.append('/bsuhome/shuleicao/pythonscripts/Unify_HOD_pipeline/python_scripts/utils/')
from arg_parser_utils import parse_arguments
args = parse_arguments()
from fid_hod import Ngal_S20_poisson_numba

# def check_and_convert(arr):
#     """Check endianness and data type, and convert if necessary."""
#     # Convert to little-endian if it's big-endian
#     if arr.dtype.byteorder == '>':
#         arr = arr.byteswap().newbyteorder()

#     # Convert to float32 if it's not
#     if arr.dtype != np.float32:
#         arr = arr.astype(np.float32)
#     return arr

def check_and_convert(arr):
    """Check endianness and data type, and convert if necessary."""
    # Convert to little-endian if it's big-endian
    if arr.dtype.byteorder == '>':
        arr = arr.byteswap().newbyteorder()
    return arr

def load_parameters(yml_file):
    with open(yml_file, 'r') as stream:
        try:
            para = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error loading YAML file: {exc}")
    return para

def get_readcat(nbody, nbody_loc, redshift):
    if nbody == 'mini_uchuu':
        from read_mini_uchuu import ReadMiniUchuu
        return ReadMiniUchuu(nbody_loc, redshift)
    elif nbody == 'uchuu':
        from read_uchuu import ReadUchuu
        return ReadUchuu(nbody_loc, redshift)
    elif nbody == 'abacus_summit':
        from read_abacus_summit import ReadAbacusSummit
        return ReadAbacusSummit(nbody_loc, redshift)
    elif nbody == 'tng_dmo':
        from read_tng_dmo import ReadTNGDMO
        halofinder = para.get('halofinder', 'rockstar')
        return ReadTNGDMO(nbody_loc, halofinder, redshift)
    else:
        raise ValueError(f"Unsupported nbody value: {nbody}")

para = load_parameters(args.yml_file)
HOD_index = args.HOD_index
redshift = para['redshift']
output_loc = os.path.join(para['output_loc'], f'HOD_z{redshift}_{HOD_index}_{args.para_name}/')
os.makedirs(output_loc, exist_ok=True)

model_name = para['model_name']
out_path = os.path.join(output_loc, f'model_{model_name}')
seed = args.seed
np.random.seed(seed=seed) # for scipy

galaxy_path = os.path.join(out_path, f'gals_seed{seed}.fit')
Mmin = 2e10

alpha = args.alpha
f_cen = args.f_cen
logMmin = args.logM_min
logM0 = args.logM0
sigma_logM = args.sigma_logM
logM1 = args.logM1
pec_vel = True #args.pec_vel

sat_from_part = para.get('sat_from_part', False)
if sat_from_part == True:
    from draw_sat_positions_from_particles_layer import DrawSatPositionsFromParticlesLayer
    dsp_part = DrawSatPositionsFromParticlesLayer(args.yml_file)
    Mmin_part = 10**12.5 # only draw particles for halos above this mass

from draw_sat_position import DrawSatPosition
dsp_mc = DrawSatPosition(args.yml_file, seed=seed)

if not os.path.isdir(out_path): os.makedirs(out_path)
if not os.path.isdir(out_path+'/temp/'): os.makedirs(out_path+'/temp/')

with open(f'{out_path}/para.yml', 'w') as outfile:
    yaml.dump(para, outfile, default_flow_style=False)

print('output is at ' + out_path)
readcat = get_readcat(para['nbody'], para['nbody_loc'], redshift)

readcat.read_halos(Mmin, pec_vel=True)
boxsize = readcat.boxsize
px_halo_all = check_and_convert(readcat.xh)
py_halo_all = check_and_convert(readcat.yh)
pz_halo_all = check_and_convert(readcat.zh)

mass_all = check_and_convert(readcat.mass_200m)
hid_all = check_and_convert(readcat.hid)

if pec_vel == True:
    vx_halo_all = check_and_convert(readcat.vx)
    vy_halo_all = check_and_convert(readcat.vy)
    vz_halo_all = check_and_convert(readcat.vz)

@njit
def calc_one_layer_numba(px_halo_all, py_halo_all, pz_halo_all, mass_all, hid_all, pec_vel, 
                         vx_halo_all, vy_halo_all, vz_halo_all, pz_min, pz_max, is_last_bin):
    if is_last_bin:
        sel = (pz_halo_all >= pz_min) & (pz_halo_all <= pz_max)
    else:
        sel = (pz_halo_all >= pz_min) & (pz_halo_all < pz_max)
        
    px_halo_sub = px_halo_all[sel]
    py_halo_sub = py_halo_all[sel]
    pz_halo_sub = pz_halo_all[sel]
    if pec_vel:
        vx_halo_sub = vx_halo_all[sel]
        vy_halo_sub = vy_halo_all[sel]
        vz_halo_sub = vz_halo_all[sel]
    else:
        vx_halo_sub = np.zeros_like(px_halo_sub)
        vy_halo_sub = np.zeros_like(py_halo_sub)
        vz_halo_sub = np.zeros_like(pz_halo_sub)
        
    mass_sub = mass_all[sel]
    hid_sub = hid_all[sel]

    Ncen = np.zeros(len(mass_sub))
    Nsat = np.zeros(len(mass_sub))
    for i in range(len(mass_sub)):
        Ncen[i], Nsat[i] = Ngal_S20_poisson_numba(mass_sub[i], alpha, logM1, logM0, logMmin, sigma_logM)
        
    is_central = Ncen > 0.5
    has_satellites = Nsat > 0.5

    central_indices = np.where(is_central)[0]
    n_central = len(central_indices)
    n_satellite = np.sum(Nsat[has_satellites].astype(np.int32))  

    # Allocate arrays to hold the output
    hid_out = np.empty(n_central + n_satellite, dtype=hid_sub.dtype)
    m_out = np.empty(n_central + n_satellite, dtype=mass_sub.dtype)
    px_out = np.empty(n_central + n_satellite, dtype=px_halo_sub.dtype)
    py_out = np.empty(n_central + n_satellite, dtype=py_halo_sub.dtype)
    pz_out = np.empty(n_central + n_satellite, dtype=pz_halo_sub.dtype)
    vx_out = np.empty(n_central + n_satellite, dtype=vx_halo_sub.dtype)
    vy_out = np.empty(n_central + n_satellite, dtype=vy_halo_sub.dtype)
    vz_out = np.empty(n_central + n_satellite, dtype=vz_halo_sub.dtype)
    iscen_out = np.empty(n_central + n_satellite, dtype=np.int32)  
    from_part_out = np.empty(n_central + n_satellite, dtype=np.int32)  

    # Fill central galaxies data
    hid_out[:n_central] = hid_sub[central_indices]
    m_out[:n_central] = mass_sub[central_indices]
    px_out[:n_central] = px_halo_sub[central_indices]
    py_out[:n_central] = py_halo_sub[central_indices]
    pz_out[:n_central] = pz_halo_sub[central_indices]
    vx_out[:n_central] = vx_halo_sub[central_indices]
    vy_out[:n_central] = vy_halo_sub[central_indices]
    vz_out[:n_central] = vz_halo_sub[central_indices]
    iscen_out[:n_central] = 1
    from_part_out[:n_central] = 1

    satellite_indices = np.where(has_satellites)[0]
    satellite_mass = mass_sub[satellite_indices]
    satellite_hid = hid_sub[satellite_indices]
    satellite_Nsat = Nsat[satellite_indices].astype(np.int32)  

    return (hid_out, m_out, px_out, py_out, pz_out, vx_out, vy_out, vz_out, 
            iscen_out, from_part_out, central_indices, satellite_indices, 
            satellite_mass, satellite_hid, satellite_Nsat, 
            px_halo_sub, py_halo_sub, pz_halo_sub, vx_halo_sub, vy_halo_sub, vz_halo_sub)

def calc_one_layer(pz_min, pz_max, is_last_bin, px_halo_all, py_halo_all, pz_halo_all, mass_all, hid_all, vx_halo_all=None, vy_halo_all=None, vz_halo_all=None):
    (hid_out, m_out, px_out, py_out, pz_out, vx_out, vy_out, vz_out, 
     iscen_out, from_part_out, central_indices, satellite_indices, 
     satellite_mass, satellite_hid, satellite_Nsat, 
     px_halo_sub, py_halo_sub, pz_halo_sub, vx_halo_sub, vy_halo_sub, vz_halo_sub) = calc_one_layer_numba(
        px_halo_all, py_halo_all, pz_halo_all, mass_all, hid_all, pec_vel, 
        vx_halo_all, vy_halo_all, vz_halo_all, pz_min, pz_max, is_last_bin)
    
    # Process satellites using the original logic
    start_idx = len(central_indices)
    for i, idx in enumerate(satellite_indices):
        num_sat = satellite_Nsat[i]
        px = np.zeros(num_sat)
        py = np.zeros(num_sat)
        pz = np.zeros(num_sat)
        vx = np.zeros(num_sat)
        vy = np.zeros(num_sat)
        vz = np.zeros(num_sat)
        if sat_from_part and satellite_mass[i] >= Mmin_part:
            px, py, pz, vx, vy, vz = dsp_part.draw_sats(satellite_mass[i], num_sat, px_halo_sub[idx], py_halo_sub[idx], pz_halo_sub[idx])
        else:
            px, py, pz = dsp_mc.draw_sat_position(satellite_mass[i], num_sat)
            vx, vy, vz = dsp_mc.draw_sat_velocity(satellite_mass[i], num_sat)
            px += px_halo_sub[idx]
            py += py_halo_sub[idx]
            pz += pz_halo_sub[idx]
            vx += vx_halo_sub[idx]
            vy += vy_halo_sub[idx]
            vz += vz_halo_sub[idx]

        end_idx = start_idx + num_sat
        hid_out[start_idx:end_idx] = satellite_hid[i]
        m_out[start_idx:end_idx] = satellite_mass[i]
        px_out[start_idx:end_idx] = px
        py_out[start_idx:end_idx] = py
        pz_out[start_idx:end_idx] = pz
        vx_out[start_idx:end_idx] = vx
        vy_out[start_idx:end_idx] = vy
        vz_out[start_idx:end_idx] = vz
        iscen_out[start_idx:end_idx] = 0
        from_part_out[start_idx:end_idx] = 0
        start_idx = end_idx

    # Modulo operation to keep positions within box
    px_out = np.mod(px_out, boxsize)
    py_out = np.mod(py_out, boxsize)
    pz_out = np.mod(pz_out, boxsize)

    return (hid_out, m_out, px_out, py_out, pz_out, vx_out, vy_out, vz_out, iscen_out, from_part_out)

n_parallel = 100
n_layer = boxsize / n_parallel

def calc_one_bin(ibin):
    pz_min = ibin * n_layer
    pz_max = (ibin + 1) * n_layer
    is_last_bin = (ibin == n_parallel - 1)
    ofname = f'{out_path}/temp/gals_{pz_min}_{pz_max}.fits'
    if True: # not os.path.exists(ofname):
        results = calc_one_layer(pz_min, pz_max, is_last_bin,
                                 px_halo_all, py_halo_all, pz_halo_all, 
                                 mass_all, hid_all,
                                 vx_halo_all, vy_halo_all, vz_halo_all)
        hid_out, m_out, px_out, py_out, pz_out, vx_out, vy_out, vz_out, iscen_out, from_part_out = results
        table = Table([hid_out, m_out, px_out, py_out, pz_out, vx_out, vy_out, vz_out, iscen_out, from_part_out], 
                      names=('haloid', 'Mvir', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'iscen', 'from_part'))
        table.write(ofname, overwrite=True)

def merge_files():
    fname_list = glob.glob(f'{out_path}/temp/gals_*.fits')
    tables = [Table.read(fname) for fname in fname_list]
    combined_table = vstack(tables)
    combined_table.sort('Mvir', reverse=True)
    combined_table.write(galaxy_path, overwrite=True)
    shutil.rmtree(f'{out_path}/temp')


if __name__ == '__main__':
    start = timeit.default_timer()
    print('make_gal_cat.py prep took', '%.2g'%((start - start_master)/60), 'mins')

    with ProcessPoolExecutor() as pool:
        for result in pool.map(calc_one_bin, range(n_parallel)):
            if result: print(result)  # output error

    stop = timeit.default_timer()
    print('galaxies took', '%.2g'%((stop - start)/60), 'mins')

    start = stop
    merge_files()
    stop = timeit.default_timer()
    print('merging took', '%.2g'%((stop - start)/60), 'mins')

    dtime = (stop - start_master)/60.
    print(f'total time {dtime:.2g} mins')
