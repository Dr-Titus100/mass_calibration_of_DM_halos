#!/usr/bin/env python
import timeit
start = timeit.default_timer()
import numpy as np
import pandas as pd
from scipy import spatial
import os
import sys
import glob
import fitsio
import yaml
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor
import argparse

sys.path.append('/bsuhome/shuleicao/pythonscripts/Unify_HOD_pipeline/python_scripts/utils/')
from arg_parser_utils import parse_arguments
args = parse_arguments()

from periodic_boundary_condition import periodic_boundary_condition
from periodic_boundary_condition import periodic_boundary_condition_halos

# parser = argparse.ArgumentParser()
# parser.add_argument('yml_file', type=str, help='Path to the YAML file')
# parser.add_argument('--depth', type=float, default=None, help='Optional: Override depth from the YAML file')
# parser.add_argument('--pec_vel', action='store_true', help="Set to True to consider peculiar velocities.")
# parser.add_argument('--los', type=str, default='z', choices=['x', 'y', 'z'], help="Line-of-sight direction. Default is 'z'.")
# parser.add_argument('--new', action='store_true', help="Use new dataset.")
# # parser.add_argument('--suffix', type=str, default='simul', help="Suffix for fitting methods, simul: simultaneous fit for both sigma_logM and logM_min; sigma_#: sigma_logM is fixed as #.")
# parser.add_argument('--HOD_index', type=int, default=0, help='HOD model index')
# args = parser.parse_args()

new_prefix = 'new_' if args.new else ''

with open(args.yml_file, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# If depth is provided from the command line, use it. Otherwise, use the value from the YAML file.
depth = args.depth if args.depth is not None else para['depth']
pec_vel = args.pec_vel
suffix = args.suffix if args.suffix is not None else None

if pec_vel:
    rich_name = f'd{depth}_pecvel'
else:
    rich_name = f'd{depth}'

redshift = para['redshift']
perc = para['perc']
use_rlambda = para['use_rlambda']
use_pmem = para.get('use_pmem', False)
los = args.los

save_members = para.get('save_members', False)
sat_from_part = para.get('sat_from_part', False)

print('use pmem:', use_pmem)
print('pec vel:', pec_vel)
print('sat from part:', sat_from_part)

Mmin = 10**12.5

n_parallel_z = 1 # NOTE! cannot do more than one yet.
n_parallel_x = 10 # TODO: check n_parallel_x != n_parallel_y
n_parallel_y = 10

n_parallel = n_parallel_z * n_parallel_x * n_parallel_y

# z_padding_halo = 0
# z_padding_gal = 0

output_loc = para['output_loc']
model_name = para['model_name']

model_suffix = f"d{args.depth}"
if args.pec_vel:
    model_suffix += "_pecvel"

if pec_vel:
    prefix = '_pecvel'
else:
    prefix = ''
HOD_index = args.HOD_index
optimize_flag = args.optimize
if not optimize_flag:
    output_loc = os.path.join(para['output_loc'], f'HOD_z{redshift}_{HOD_index}_{args.para_name}/')
else:
    output_loc = os.path.join(para['output_loc'], f'Optimize_HOD_z{redshift}_{HOD_index}_{args.para_name}/')
os.makedirs(output_loc, exist_ok=True)

if suffix is None:
    out_path = f'{output_loc}/model_{model_name}_{model_suffix}/'
else:
    out_path = f'{output_loc}/model_{model_name}_{model_suffix}_{suffix}/'
    
if not os.path.isdir(out_path):
    os.makedirs(out_path)
if not os.path.isdir(f'{out_path}/temp/'):
    os.makedirs(f'{out_path}/temp/')

# read in halos
if para['nbody'] == 'mini_uchuu':
    from read_mini_uchuu import ReadMiniUchuu
    readcat = ReadMiniUchuu(para['nbody_loc'], redshift)

if para['nbody'] == 'uchuu':
    from read_uchuu import ReadUchuu
    readcat = ReadUchuu(para['nbody_loc'], redshift)

if para['nbody'] == 'abacus_summit':
    sys.path.append('../abacus_summit')
    from read_abacus_summit import ReadAbacusSummit
    readcat = ReadAbacusSummit(para['nbody_loc'], redshift)

if para['nbody'] == 'tng_dmo':
    from read_tng_dmo import ReadTNGDMO
    halofinder = para.get('halofinder', 'rockstar')
    readcat = ReadTNGDMO(para['nbody_loc'], halofinder, redshift)
    print('halofinder', halofinder)

boxsize = readcat.boxsize
OmegaM = readcat.OmegaM
hubble = readcat.hubble
readcat.read_halos(Mmin, pec_vel=pec_vel)
hid_in = readcat.hid
mass_in = readcat.mass

OmegaDE = 1 - OmegaM
Ez = np.sqrt(OmegaM * (1+redshift)**3 + OmegaDE)

if los == 'z':
    x_halo_in = readcat.xh
    y_halo_in = readcat.yh
    z_halo_in = readcat.zh
    if pec_vel == True:
        z_halo_in += (1+redshift) * readcat.vz / Ez / 100.
if los == 'x':
    x_halo_in = readcat.yh
    y_halo_in = readcat.zh
    z_halo_in = readcat.xh
    if pec_vel == True:
        z_halo_in += (1+redshift) * readcat.vx / Ez / 100.
if los == 'y':
    x_halo_in = readcat.zh
    y_halo_in = readcat.xh
    z_halo_in = readcat.yh
    if pec_vel == True:
        z_halo_in += (1+redshift) * readcat.vy / Ez / 100.

del readcat

if use_pmem == True:
    use_cylinder = False
    which_pmem = para.get('which_pmem')
    if which_pmem == 'myles3':
        from pmem_weights_myles3 import pmem_weights
    if which_pmem == 'buzzard':
        from pmem_weights_buzzard import pmem_weights

    depth = -1
    dz_max = 0.5 * boxsize * Ez / 3000. # need to be smaller than half box size, otherwise the same galaxies will be counted twice
    print('dz_max', dz_max)
else:
    use_cylinder = True

scale_factor = 1./(1.+redshift)

# read in galaxies
gal_cat_format = para.get('gal_cat_format', 'fits')

if gal_cat_format == 'fits':
    if suffix is None:
        gal_fname = f'{output_loc}/model_{model_name}/{new_prefix}gals.fit'
    else:
        gal_fname = f'{output_loc}/model_{model_name}/{new_prefix}gals_{suffix}.fit'
    with fits.open(gal_fname) as hdul:
        data = hdul[1].data  # Assuming the data is in the first extension
        header = hdul[1].header
        if los == 'z':
            x_gal_in = data['px']
            y_gal_in = data['py']
            z_gal_in = data['pz']
            if pec_vel:
                z_gal_in += (1+redshift) * data['vz'] / Ez / 100.
        if los == 'x':
            x_gal_in = data['py']
            y_gal_in = data['pz']
            z_gal_in = data['px']
            if pec_vel:
                z_gal_in += (1+redshift) * data['vx'] / Ez / 100.
        if los == 'y':
            x_gal_in = data['pz']
            y_gal_in = data['px']
            z_gal_in = data['py']
            if pec_vel:
                z_gal_in += (1+redshift) * data['vy'] / Ez / 100.

#### periodic boundary condition ####
# x_padding = 3
# y_padding = 3
if los == 'z':
    x_padding, y_padding, z_padding = 3, 3, 0
elif los == 'x':
    x_padding, y_padding, z_padding = 0, 3, 3
elif los == 'y':
    x_padding, y_padding, z_padding = 3, 0, 3

# x_halo, y_halo, z_halo, hid, mass = periodic_boundary_condition_halos(
#     x_halo_in, y_halo_in, z_halo_in, 
#     boxsize, x_padding, y_padding, 0, hid_in, mass_in)

x_halo, y_halo, z_halo, hid, mass = periodic_boundary_condition_halos(
    x_halo_in, y_halo_in, z_halo_in, 
    boxsize, x_padding, y_padding, z_padding, hid_in, mass_in)

sort = np.argsort(-mass)
hid = hid[sort]
mass = mass[sort]
x_halo = x_halo[sort]
y_halo = y_halo[sort]
z_halo = z_halo[sort]

# x_gal, y_gal, z_gal = periodic_boundary_condition(
#     x_gal_in, y_gal_in, z_gal_in,
#     boxsize, x_padding, y_padding, 0)

x_gal, y_gal, z_gal = periodic_boundary_condition(
    x_gal_in, y_gal_in, z_gal_in,
    boxsize, x_padding, y_padding, z_padding)

class CalcRichness(object): # one pz slice at a time
    def __init__(self, pz_min, pz_max, px_min=0, px_max=boxsize, py_min=0, py_max=boxsize):
        self.pz_min = pz_min
        self.pz_max = pz_max
        self.px_min = px_min
        self.px_max = px_max
        self.py_min = py_min
        self.py_max = py_max

#         sel_gal = (z_gal > pz_min - z_padding_gal) & (z_gal < pz_max + z_padding_gal)
        sel_gal = (z_gal > pz_min - z_padding) & (z_gal < pz_max + z_padding)
        if px_min > 0 or px_max < boxsize or py_min > 0 or py_max < boxsize: # further dicing the pz slice
            sel_gal &= (x_gal > px_min - x_padding) & (x_gal < px_max + x_padding)
            sel_gal &= (y_gal > py_min - y_padding) & (y_gal < py_max + y_padding)

        self.x_gal = x_gal[sel_gal]
        self.y_gal = y_gal[sel_gal]
        self.z_gal = z_gal[sel_gal]
        
#         sel_halo = (z_halo > pz_min - z_padding_halo) & (z_halo < pz_max + z_padding_halo)
        sel_halo = (z_halo > pz_min - z_padding) & (z_halo < pz_max + z_padding)
        if px_min > 0 or px_max < boxsize or py_min > 0 or py_max < boxsize:
            sel_halo &= (x_halo > px_min - x_padding) & (x_halo < px_max + x_padding)
            sel_halo &= (y_halo > py_min - y_padding) & (y_halo < py_max + y_padding)

        self.x_halo = x_halo[sel_halo]
        self.y_halo = y_halo[sel_halo]
        self.z_halo = z_halo[sel_halo]
        self.hid = hid[sel_halo]
        self.mass = mass[sel_halo]

        #### sort again, just to be safe ####
        sort = np.argsort(-self.mass)
        self.hid = self.hid[sort]
        self.mass = self.mass[sort]
        self.x_halo = self.x_halo[sort]
        self.y_halo = self.y_halo[sort]
        self.z_halo = self.z_halo[sort]

        gal_position = np.dstack([self.x_gal, self.y_gal])[0]
        gal_tree = spatial.cKDTree(gal_position)

        halo_position = np.dstack([self.x_halo, self.y_halo])[0]
        halo_tree = spatial.cKDTree(halo_position)

        rmax_tree = 2
        self.indexes_tree = halo_tree.query_ball_tree(gal_tree, r=rmax_tree)
        self.gal_taken = np.zeros(len(self.x_gal)) # for percolation

    def get_richness(self, i_halo):
        gal_ind = self.indexes_tree[i_halo]
        x_cen = self.x_halo[i_halo]
        y_cen = self.y_halo[i_halo]
        z_cen = self.z_halo[i_halo]

        #### step 1: cut the LOS ####
        z_gal_gal_ind = self.z_gal[gal_ind]
        d_pbc0 = z_gal_gal_ind - z_cen
        d_pbc1 = z_gal_gal_ind + boxsize - z_cen
        d_pbc2 = z_gal_gal_ind - boxsize - z_cen

        dz0 = d_pbc0 * Ez / 3000.
        dz1 = d_pbc1 * Ez / 3000.
        dz2 = d_pbc2 * Ez / 3000.

        if use_cylinder == True and depth > 0:
            sel_z0 = (np.abs(d_pbc0) < depth)
            sel_z1 = (np.abs(d_pbc1) < depth)
            sel_z2 = (np.abs(d_pbc2) < depth)
            sel_z = sel_z0 | sel_z1 | sel_z2
            sel_z = sel_z & (self.gal_taken[gal_ind] < 1e-4)
            dz0 = dz0[sel_z]
            dz1 = dz1[sel_z]
            dz2 = dz2[sel_z]

        elif use_pmem == True and depth == -1:
            sel_z0 = (np.abs(dz0) < dz_max)
            sel_z1 = (np.abs(dz1) < dz_max)
            sel_z2 = (np.abs(dz2) < dz_max)
            sel_z = sel_z0 | sel_z1 | sel_z2
            sel_z = sel_z & (self.gal_taken[gal_ind] < 0.8) # TODO: percolation threshold?
            dz0 = dz0[sel_z]
            dz1 = dz1[sel_z]
            dz2 = dz2[sel_z]

        else:
            print('BUG!!')

        #### step 2: calculate radius ####
        r = (self.x_gal[gal_ind][sel_z] - x_cen)**2 + (self.y_gal[gal_ind][sel_z] - y_cen)**2 
        r = np.sqrt(r)

        #### step 3: iteratively calculating r_lambda ####
        if use_rlambda == True:
            rlam_ini = 1
            rlam = rlam_ini
            for iteration in range(100):
                if use_cylinder == True and depth > 0:
                    ngal = len(r[r < rlam])
                elif use_pmem == True or depth == -1:
                    pmem0 = pmem_weights(dz0, r/rlam, dz_max=dz_max)
                    pmem1 = pmem_weights(dz1, r/rlam, dz_max=dz_max)
                    pmem2 = pmem_weights(dz2, r/rlam, dz_max=dz_max)
                    pmem = pmem0 + pmem1 + pmem2
                    ngal = np.sum(pmem)
                else:
                    print('BUG!!')

                rlam_old = rlam
                rlam = (ngal/100.)**0.2 / scale_factor # phys -> comoving
                if abs(rlam - rlam_old) < 1e-5 or rlam < 1e-6:
                    break
        else: 
            rlam = radius # fixed aperture

        #### Step 4: do percolation ####
        if rlam > 0:
            sel_mem = (r < rlam)
            if perc == True and len(gal_ind) > 0:
                if use_cylinder == True:
                    self.gal_taken[np.array(gal_ind)[sel_z][sel_mem]] = 1
                if use_pmem == True: # probabilistic percolation
                    self.gal_taken[np.array(gal_ind)[sel_z][sel_mem]] += pmem[sel_mem]

        #### Step 5 (optional): save the member galaxies ####
        if save_members == True:
            if rlam > 0:
                if use_cylinder == True and depth > 0: # no repeat
                    self.x_gal_mem = self.x_gal[gal_ind][sel_z][sel_mem]
                    self.y_gal_mem = self.y_gal[gal_ind][sel_z][sel_mem]
                    self.z_gal_mem = self.z_gal[gal_ind][sel_z][sel_mem]
                    dz_all = np.array([dz0[sel_mem], dz1[sel_mem], dz2[sel_mem]])
                    arg = np.array([np.argmin(np.abs(dz_all), axis=0)]) # find the smallest absolute value
                    self.dz_out = np.take_along_axis(dz_all, arg, axis=0) # cool numpy function!
                    self.dz_out = np.concatenate(self.dz_out)
                    self.r_out = r[sel_mem]/rlam 
                    self.p_gal_mem = self.x_gal_mem * 0 + 1
                    self.pmem_out = self.x_gal_mem * 0 + 1

                elif use_pmem == True or depth == -1: # each gal: repeat 3 times for PBC
                    self.x_gal_mem = np.tile(self.x_gal[gal_ind][sel_z][sel_mem], 3)
                    self.y_gal_mem = np.tile(self.y_gal[gal_ind][sel_z][sel_mem], 3)
                    self.z_gal_mem = np.tile(self.z_gal[gal_ind][sel_z][sel_mem], 3)
                    # save duplicate galaxies for dz0, dz1, and dz2
                    self.dz_out = np.concatenate([dz0[sel_mem], dz1[sel_mem], dz2[sel_mem]])
                    self.r_out = np.tile(r[sel_mem]/rlam, 3) 
                    self.p_gal_mem = pmem[sel_mem]
                    self.pmem_out = np.concatenate([pmem0[sel_mem], pmem1[sel_mem], pmem2[sel_mem]])
                    
                    sel = (self.pmem_out > 1e-6)
                    self.x_gal_mem = self.x_gal_mem[sel]
                    self.y_gal_mem = self.y_gal_mem[sel]
                    self.z_gal_mem = self.z_gal_mem[sel]
                    self.dz_out = self.dz_out[sel]
                    self.r_out = self.r_out[sel]
                    self.pmem_out = self.pmem_out[sel]

                    if max(self.p_gal_mem) > 1:
                        print('max(self.p_gal_mem)', max(self.p_gal_mem), 'BUG!: double counting galaxies.')
                        exit()
                else:
                    print('BUG')

        return rlam, ngal

    def measure_richness(self):
        nh = len(self.x_halo)

        #### richness files:  ####
        if suffix is None:
            ofname1 = f'{out_path}/temp/{new_prefix}richness_{rich_name}_pz{self.pz_min:.0f}_{self.pz_max:.0f}_px{self.px_min:.0f}_{self.px_max:.0f}_py{self.py_min:.0f}_{self.py_max:.0f}.dat'
        else:
            ofname1 = f'{out_path}/temp/{new_prefix}richness_{rich_name}_{suffix}_pz{self.pz_min:.0f}_{self.pz_max:.0f}_px{self.px_min:.0f}_{self.px_max:.0f}_py{self.py_min:.0f}_{self.py_max:.0f}.dat'
        outfile1 = open(ofname1, 'w')
        outfile1.write('#hid, mass, px, py, pz, rlam, lam \n')

        #### member files: only write header (optional) ####
        if save_members == True:
            if suffix is None:
                ofname2 = f'{out_path}/temp/{new_prefix}members_{rich_name}_pz{self.pz_min:.0f}_{self.pz_max:.0f}_px{self.px_min:.0f}_{self.px_max:.0f}_py{self.py_min:.0f}_{self.py_max:.0f}.dat'
            else:
                ofname2 = f'{out_path}/temp/{new_prefix}members_{rich_name}_{suffix}_pz{self.pz_min:.0f}_{self.pz_max:.0f}_px{self.px_min:.0f}_{self.px_max:.0f}_py{self.py_min:.0f}_{self.py_max:.0f}.dat'
            outfile2 = open(ofname2, 'w')
            outfile2.write('#hid, dz, r/rlam, pmem \n')
            outfile2.close()

        for ih in range(nh):
            rlam, lam = self.get_richness(ih)
            if lam > 0 and \
                self.z_halo[ih] > self.pz_min and self.z_halo[ih] < self.pz_max and \
                self.x_halo[ih] > self.px_min and self.x_halo[ih] < self.px_max and \
                self.y_halo[ih] > self.py_min and self.y_halo[ih] < self.py_max:

                outfile1.write('%12i %15e %12g %12g %12g %12g %12g \n'%(self.hid[ih], self.mass[ih], self.x_halo[ih], self.y_halo[ih], self.z_halo[ih], rlam, lam))

                #### save members (append) (optional) #### 
                if save_members == True:
                    self.dz_out *= 3000. / Ez # convert back to comoving distance
                    self.hid_mem = self.x_gal_mem * 0 + self.hid[ih]
                    data = np.array([self.hid_mem, self.x_gal_mem, self.y_gal_mem, self.z_gal_mem, self.dz_out, self.r_out, self.pmem_out]).transpose()
                    with open(ofname2, "ab") as f:
                        np.savetxt(f, data, fmt='%12i %12g %12g %12g %12g %12g %12g')

        outfile1.close()



z_layer_thickness = boxsize / n_parallel_z
x_cube_size = boxsize / n_parallel_x
y_cube_size = boxsize / n_parallel_y

def calc_one_bin(ibin):
    iz = ibin // (n_parallel_x * n_parallel_y)
    ixy = ibin % (n_parallel_x * n_parallel_y)
    ix = ixy // n_parallel_x
    iy = ixy % n_parallel_x
    pz_min = iz*z_layer_thickness
    pz_max = (iz+1)*z_layer_thickness
    px_min = ix*x_cube_size
    px_max = (ix+1)*x_cube_size
    py_min = iy*y_cube_size
    py_max = (iy+1)*y_cube_size

    if suffix is None:
        ofname = f'{out_path}/temp/{new_prefix}richness_{rich_name}_pz{pz_min:.0f}_{pz_max:.0f}_px{px_min:.0f}_{px_max:.0f}_py{py_min:.0f}_{py_max:.0f}.dat'
    else:
        ofname = f'{out_path}/temp/{new_prefix}richness_{rich_name}_{suffix}_pz{pz_min:.0f}_{pz_max:.0f}_px{px_min:.0f}_{px_max:.0f}_py{py_min:.0f}_{py_max:.0f}.dat'

    if os.path.exists(ofname) == False:
        cr = CalcRichness(pz_min=pz_min, pz_max=pz_max, px_min=px_min, px_max=px_max, py_min=py_min, py_max=py_max)
        cr.measure_richness()

def merge_files_richness():
    if suffix is None:
        fname_list = glob.glob(f'{out_path}/temp/{new_prefix}richness_{rich_name}_pz*.dat')
    else:
        fname_list = glob.glob(f'{out_path}/temp/{new_prefix}richness_{rich_name}_{suffix}_pz*.dat')
    nfiles = len(fname_list)
    if nfiles < n_parallel:
        print('missing ', n_parallel - nfiles, 'files, not merging')
    else:
        hid_out = []
        m_out = []
        x_out = []
        y_out = []
        z_out = []
        rlam_out = []
        lam_out = []

        for fname in fname_list:
            data = pd.read_csv(fname, delim_whitespace=True, dtype=np.float64, comment='#', 
                            names=['haloid', 'mass', 'px', 'py', 'pz', 'rlam', 'lam'])
            hid_out.extend(data['haloid'])
            m_out.extend(data['mass'])
            x_out.extend(data['px'])
            y_out.extend(data['py'])
            z_out.extend(data['pz'])
            rlam_out.extend(data['rlam'])
            lam_out.extend(data['lam'])
            
        hid_out = np.array(hid_out)
        m_out = np.array(m_out)
        x_out = np.array(x_out)
        y_out = np.array(y_out)
        z_out = np.array(z_out)
        rlam_out = np.array(rlam_out)
        lam_out = np.array(lam_out)

        sel = np.argsort(-m_out)

        cols = [
            fits.Column(name='haloid', format='K' ,array=hid_out[sel]),
#             fits.Column(name='M200m', format='E',array=m_out[sel]),
            fits.Column(name='Mvir', format='E',array=m_out[sel]),
            fits.Column(name='px', format='D' ,array=x_out[sel]),
            fits.Column(name='py', format='D',array=y_out[sel]),
            fits.Column(name='pz', format='D',array=z_out[sel]),
            fits.Column(name='Rlambda', format='D',array=rlam_out[sel]),
            fits.Column(name='lambda', format='D',array=lam_out[sel]),
        ]
        coldefs = fits.ColDefs(cols)
        tbhdu = fits.BinTableHDU.from_columns(coldefs)
        if suffix is None:
            tbhdu.writeto(f'{out_path}/{new_prefix}richness_{rich_name}.fit', overwrite=True)
            os.system(f'rm -rf {out_path}/temp/{new_prefix}richness_{rich_name}_pz*.dat')
        else:
            tbhdu.writeto(f'{out_path}/{new_prefix}richness_{rich_name}_{suffix}.fit', overwrite=True)
            os.system(f'rm -rf {out_path}/temp/{new_prefix}richness_{rich_name}_{suffix}_pz*.dat')

def merge_files_members():
    if suffix is None:
        fname_list = glob.glob(f'{out_path}/temp/{new_prefix}members_{rich_name}_pz*.dat')
    else:
        fname_list = glob.glob(f'{out_path}/temp/{new_prefix}members_{rich_name}_{suffix}_pz*.dat')
    nfiles = len(fname_list)
    if nfiles < n_parallel:
        print('missing ', n_parallel - nfiles, 'files, not merging')
    else:
        print('nfiles', nfiles)
        hid_out = []
        x_out = []
        y_out = []
        z_out = []
        dz_out = []
        r_out = []
        pmem_out = []

        for fname in fname_list:
            data = pd.read_csv(fname, delim_whitespace=True, dtype=np.float64, comment='#', 
                            names=['haloid', 'px', 'py', 'pz', 'dz', 'r', 'pmem'])
            hid_out.extend(data['haloid'])
            x_out.extend(data['px'])
            y_out.extend(data['py'])
            z_out.extend(data['pz'])
            dz_out.extend(data['dz'])
            r_out.extend(data['r'])
            pmem_out.extend(data['pmem'])
            
        hid_out = np.array(hid_out)
        x_out = np.array(x_out)
        y_out = np.array(y_out)
        z_out = np.array(z_out)
        dz_out = np.array(dz_out)
        r_out = np.array(r_out)
        pmem_out = np.array(pmem_out)

        cols = [
            fits.Column(name='haloid', format='K' ,array=hid_out),
            fits.Column(name='px_gal', format='D' ,array=x_out),
            fits.Column(name='py_gal', format='D',array=y_out),
            fits.Column(name='pz_gal', format='D',array=z_out),
            fits.Column(name='dz_gal', format='D',array=dz_out),
            fits.Column(name='r_over_rlambda', format='D',array=r_out),
            fits.Column(name='pmem', format='D',array=pmem_out),
        ]
        coldefs = fits.ColDefs(cols)
        tbhdu = fits.BinTableHDU.from_columns(coldefs)
        if suffix is None:
            tbhdu.writeto(f'{out_path}/{new_prefix}members_{rich_name}.fit', overwrite=True)
            os.system(f'rm -rf {out_path}/temp/{new_prefix}members_{rich_name}_pz*.dat')
        else:
            tbhdu.writeto(f'{out_path}/{new_prefix}members_{rich_name}_{suffix}.fit', overwrite=True)
            os.system(f'rm -rf {out_path}/temp/{new_prefix}members_{rich_name}_{suffix}_pz*.dat')


if __name__ == '__main__':
    #calc_one_bin(0)
    
    stop = timeit.default_timer()
    print('calc_richness.py prep took', '%.2g'%((stop - start)/60), 'mins')
    start = stop
    
    n_cpu = os.cpu_count()
    n_repeat = int(np.ceil(n_parallel/n_cpu))
    for i_repeat in range(n_repeat):
        with ProcessPoolExecutor() as pool:
            for result in pool.map(calc_one_bin, range(i_repeat*n_cpu, min(n_parallel, (i_repeat+1)*n_cpu))):
                if result: print(result)  # output error

    stop = timeit.default_timer()
    print('richness took', '%.2g'%((stop - start)/60), 'mins')
    start = stop

    merge_files_richness()
    if save_members == True:
        merge_files_members()

    stop = timeit.default_timer()
    print('merging took', '%.2g'%((stop - start)/60), 'mins')
    
