import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_cylinder', action='store_true', help='Flag to run cylinder clusters')
    parser.add_argument('--use_redmapper_chunhao', action='store_true', help='Flag to run chunhao redmapper')
########################################################################
    parser.add_argument('--pec_vel', action='store_true', help="Set to True to consider peculiar velocities.")
    #     parser.add_argument('--los', type=str, default='z', choices=['x', 'y', 'z'], help="Line-of-sight direction. Default is 'z'.")
    parser.add_argument('--new', action='store_true', help="Use new satellite HOD method.")
    parser.add_argument('--alpha', type=float, default=1.0, help='Slope of satellite occupation power law')
    parser.add_argument('--logM_min', type=float, default=12.0, help='Minimum halo mass to host a central')
    parser.add_argument('--sigma_logM', type=float, default=0.1, help='Width of central transition')
    parser.add_argument('--logM0', type=float, default=11.7, help='Satellite cut-off mass')
    parser.add_argument('--logM1', type=float, default=12.9, help='Minimum halo mass to host a satellite')
    parser.add_argument('--f_cen', type=float, default=1.0, help='Central incompleteness fraction')
    # parser.add_argument('--yml_file', type=str, default='/bsuhome/tnde/Lensing/codes/notebooks/mini_uchuu/mini_uchuu_fid_hod.yml', help='Path to the YAML file')
    parser.add_argument('yml_file', type=str, nargs='?', default=None, help='Path to the YAML file')

    parser.add_argument('--redshift', type=float, default=0.3, help='Specify redshift')
    parser.add_argument('--h', type=float, default=0.6774, help='Specify reduced Hubble constant')
    parser.add_argument('--Ob', type=float, default=0.0486, help='Specify cosmo baryon density parameter, \Omega_b')
    parser.add_argument('--Om', type=float, default=0.3089, help='Specify cosmo mass density parameter, \Omega_m')
    parser.add_argument('--ns', type=float, default=0.9667, help='Specify spectral index of the scalar power spectrum, n_s')
    parser.add_argument('--sigma8', type=float, default=0.8159, help='Specify root mean square of the amplitude of matter (perturbations) power spectrum averaged over spheres of 8 Mpc/h radius, \sigma_8')
    parser.add_argument('--nbody_loc', type=str, default='/bsuhome/hwu/scratch/uchuu/MiniUchuu/')
    parser.add_argument('--output_loc', type=str, default='/bsuscratch/tnde/MiniUchuu_mock_d30_no_pecvel_final_changing_vol_heidi_final2')
    parser.add_argument('--model_name', type=str, default='fid_hod_z0.3')
    parser.add_argument('--nbody', type=str, default='mini_uchuu')

    parser.add_argument('--depth', type=float, default=30.0, help='Optional: Override depth from the YAML file')
    parser.add_argument('--los', type=str, default='z', choices=['x', 'y', 'z'], help="Line-of-sight direction. Default is 'z'.")
    # parser.add_argument('--new', action='store_true', help="Use new dataset.")
    parser.add_argument('--suffix', type=str, default=None, help="Suffix for fitting methods, simul: simultaneous fit for both sigma_logM and logM_min; sigma_#: sigma_logM is fixed as #.")
    parser.add_argument('--HOD_index', type=int, help='HOD model index')
    parser.add_argument('--para_name', type=str, default='joblib_new', help="Suffix for parallel methods")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
########################################################################

    if args.use_cylinder and args.depth is None:
        parser.error("--depth must be specified when using --use_cylinder")

    return args
