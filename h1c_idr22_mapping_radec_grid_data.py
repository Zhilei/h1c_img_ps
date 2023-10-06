from glob import glob
import numpy as np
from astropy.time import Time
from astropy.io import fits
from astropy.table import Table
import time
from astropy import constants as const
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import copy
import pickle
from itertools import product
from pyuvdata import UVData
import os
from pygdsm import GlobalSkyModel2016
from direct_optimal_mapping import optimal_mapping_radec_grid, data_conditioning

RAD2HR = 12/np.pi
HR2DEG = 15

data_type = 'h1c_idr22' # 'validation', 'h1c_idr32'
val_type = 'sum' # 'true_eor', 'true_foregrounds', 'true_sum', 'sum', only useful when data_type == 'validation'
# version = '2023Feb23'
# n_int = 100 # number of integrations
# n_ant = 320
# map_type = '%dint%dant'%(n_int, n_ant)
band = 'band1'
split = 'even'

sequence = 'forward'
nthread = 15

if data_type == 'h1c_idr22':
    OUTPUT_FOLDER = '/nfs/esc/hera/zhileixu/optimal_mapping/h1c_idr22/radec_grid/%s/%s'%(band, split)
elif data_type == 'validation':
#     OUTPUT_FOLDER = '/nfs/esc/hera/zhileixu/optimal_mapping/h1c_idr22/radec_grid/validation/%s/%s/%s'%(val_type, version, map_type)
    OUTPUT_FOLDER = '/nfs/esc/hera/zhileixu/optimal_mapping/h1c_idr22/radec_grid/validation/%s/data'%(val_type)
OVERWRITE = False

print('Data type:', data_type)
if data_type == 'validation':
    print('Validation type:', val_type)
print('Mapping para.:', sequence, band, split) 
# print('Number of integrations:', n_int)
# print('Number of antennas:', n_ant)
print('overwrite:', OVERWRITE)
print('Number of threads:', nthread)
print(OUTPUT_FOLDER)

def radec_map_making(files, ifreq, ipol,
                     p_mat_calc=True, 
                     select_ant=False):

    t0 = time.time()
    ra_center_deg = 29.6 # 29.6 for idr22 3:8, 24.2 for idr22 0:9
    dec_center_deg = -30.7
    ra_rng_deg = 32
    n_ra = 64
    dec_rng_deg = 8
    n_dec = 16
    
    # Fields
    # field 1: 1.25 -- 2.7 hr
    # field 2: 4.5 -- 6.5 hr
    # field 3: 8.5 -- 10.75 hr
    lst_ends = np.array([1.25, 2.7]) / RAD2HR
    
    sky_px = optimal_mapping_radec_grid.SkyPx()
    px_dic = sky_px.calc_radec_pix(ra_center_deg, ra_rng_deg, n_ra, 
                            dec_center_deg, dec_rng_deg, n_dec)
    uv_org = UVData()
    uv_org.read(files, freq_chans=ifreq, polarizations=ipol)
#     print('Start lst array:', np.unique(uv_org.lst_array) * RAD2HR)
    uv_org.select(lst_range=lst_ends, inplace=True)
#     print('Selected LSTs (hr):', np.unique(uv_org.lst_array) * RAD2HR)
    if select_ant:
        ant_sel = np.array([  1,  12,  13,  14,  23,  25,  26,  27,  36,  37,  38,  39,  40,
                             41,  51,  52,  55,  65,  66,  68,  70,  71,  82,  83,  84,  85,
                             86,  87,  88, 120, 121, 123, 124, 137, 138, 140, 141, 142, 143])

        uv_org.select(antenna_nums=ant_sel, inplace=True, keep_all_metadata=False)
    start_flag = True
    time_arr = np.unique(uv_org.time_array)[:]
    freq = uv_org.freq_array[0, 0]
    if split == 'even':
        time_arr_sel = time_arr[::2]
    elif split == 'odd':
        time_arr_sel = time_arr[1::2]
    
    for time_t in time_arr_sel:
        #print(itime, time_t, end=';')
        uv = uv_org.select(times=[time_t,], keep_all_metadata=False, inplace=False)

        # Data Conditioning
        dc = data_conditioning.DataConditioning(uv, 0, ipol)
        dc.bl_selection()
        dc.noise_calc()
        n_vis = dc.uv_1d.data_array.shape[0]
        if dc.rm_flag() is None:
            #print('All flagged. Passed.')
            continue
        dc.redundant_avg()
        bl_max = np.sqrt(np.sum(dc.uv_1d.uvw_array**2, axis=1)).max()
        radius2ctr = np.radians(hp.rotator.angdist(np.array([px_dic['ra_deg'].flatten(), px_dic['dec_deg'].flatten()]), 
                                                   [np.mean(px_dic['ra_deg']), np.mean(px_dic['dec_deg'])], lonlat=True))
        radius2ctr = radius2ctr.reshape(px_dic['ra_deg'].shape)
        
        opt_map = optimal_mapping_radec_grid.OptMapping(dc.uv_1d, px_dic)

        file_name = OUTPUT_FOLDER+\
        '/h1c_idr22_%s_%.2fMHz_pol%d_radec_grid_RA%dDec%d.p'%(split, freq/1e6, ipol, 
                                                                              ra_rng_deg, dec_rng_deg)

        if OVERWRITE == False:
            if os.path.exists(file_name):
                print(file_name, 'existed, return.')
                return

        opt_map.set_a_mat(uvw_sign=1)
#         print(opt_map.a_mat.shape)
        opt_map.set_inv_noise_mat(dc.uvn, norm=True)
        map_vis = np.matmul(np.conjugate(opt_map.a_mat.T), 
                            np.matmul(opt_map.inv_noise_mat, 
                                      opt_map.data))
        map_vis = np.real(map_vis)
        beam_weight = np.matmul(np.conjugate((opt_map.beam_mat).T), 
                                np.diag(opt_map.inv_noise_mat),)
        
        beam_sq_weight = np.matmul(np.conjugate((opt_map.beam_mat**2).T), 
                                np.diag(opt_map.inv_noise_mat),)

        if p_mat_calc:
            opt_map.set_p_mat()
        else:
            opt_map.p_mat = np.nan

        if start_flag:
            map_sum = copy.deepcopy(map_vis)
            beam_weight_sum = copy.deepcopy(beam_weight)
            beam_sq_weight_sum = copy.deepcopy(beam_sq_weight)
            p_sum = copy.deepcopy(opt_map.p_mat)
            start_flag=False
        else:
            map_sum += map_vis
            beam_weight_sum += beam_weight
            beam_sq_weight_sum += beam_sq_weight
            p_sum += opt_map.p_mat

    if start_flag == True:
        print(f'ifreq:{ifreq} no unflagged data available.')
        return

    result_dic = {'px_dic':px_dic,
                  'map_sum':map_sum,
                  'beam_weight_sum':beam_weight_sum,
                  'beam_sq_weight_sum':beam_sq_weight_sum,
                  'n_vis': n_vis,
                  'p_sum': p_sum,
                  'freq': freq,
                  'bl_max':bl_max,
                  'radius2ctr': radius2ctr,
                 }
    with open(file_name, 'wb') as f_t:
        pickle.dump(result_dic, f_t, protocol=4) 
    print(f'ifreq:{ifreq} finished in {time.time() - t0} seconds.')
    return

if __name__ == '__main__':
    if data_type == 'h1c_idr22':
        #H1C part
        data_folder = '/nfs/esc/hera/H1C_IDR22/IDR2_2_pspec/v2/one_group/data'
        files = np.array(sorted(glob(data_folder+'/zen.grp1.of1.LST.*.HH.OCRSLP2X.uvh5')))[2:8]
    elif data_type == 'h1c_idr32':
        data_folder = '/nfs/esc/hera/H1C_IDR32/LSTBIN/all_epochs'
        files = np.array(sorted(glob(data_folder+'/zen.grp1.of1.LST.*.sum.uvh5')))
    elif data_type == 'validation':
        if val_type == 'sum':
            data_folder = '/nfs/esc/hera/Validation/test-4.0.0/pipeline/LSTBIN/%s'%val_type
            files = np.array(sorted(glob(data_folder+'/zen.grp1.of1.LST.*.HH.OCRSLPX.uvh5')))[:3]
        elif val_type == 'true_sum':
            data_folder = '/nfs/esc/hera/Validation/test-4.0.0/pipeline/LSTBIN/%s'%val_type
            files = np.array(sorted(glob(data_folder+'/zen.grp1.of1.LST.*.HH.OCRSL.uvh5')))[:3]
        elif val_type == 'true_foregrounds':
            data_folder = '/nfs/esc/hera/Validation/test-4.0.0/pipeline/LSTBIN/%s'%val_type
            files = np.array(sorted(glob(data_folder+'/zen.grp1.of1.LST.*.HH.OCRSL.uvh5')))[:3]
        elif val_type == 'true_eor':
            data_folder = '/nfs/esc/hera/Validation/test-4.0.0/pipeline/LSTBIN/%s'%val_type
            files = np.array(sorted(glob(data_folder+'/zen.eor.LST.*.HH.uvh5')))[:3]
        elif val_type == 'honggeun_gsm':
            data_folder = '/nfs/ger/proj/hera/hgkim/simulations/dipole_GSM_nside256/2023Feb23'
            files = np.array(sorted(glob(data_folder+'/sim.2458116.*_GSM2008_nside256_J2000.uvh5')))[:] # from 19.5 to 27.8 deg RA
        else:
            print('Wrong validation type.')
    else:
        print('Wrong data type.')
        
    print('%d Files being mapped:\n'%len(files), files)
    if band == 'band1':
        ifreq_arr = np.arange(175, 335, dtype=int) #band1
    elif band == 'band2':
        ifreq_arr = np.arange(515, 695, dtype=int) #band2
    else:
        raise RuntimeError('Wrong input for band.')
#     ifreq_arr = np.arange(180)
#     ifreq_arr = np.arange(77, 79)    
    ipol_arr = [-5]
    if sequence == 'forward':
        args = product(np.expand_dims(files, axis=0), ifreq_arr[:], ipol_arr)
    elif sequence == 'backward':
        args = product(np.expand_dims(files, axis=0), ifreq_arr[::-1], ipol_arr)
    else:
        raise RuntimeError('Sequence should be either forward or backward.')

#     for args_t in args:
#         print(args_t)
#         radec_map_making(*args_t)

    pool = multiprocessing.Pool(processes=nthread)
    pool.starmap(radec_map_making, args)
    pool.close()
    pool.join()
