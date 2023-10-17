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
suffix = "OCRSLP2XTK"
field = 'field1'
band = 'band2'
split = 'odd'
ipol_arr = [1,]

sequence = 'forward'
nthread = 15

OUTPUT_FOLDER = '/nfs/esc/hera/zhileixu/optimal_mapping/h1c_idr22/radec_grid/%s/%s/%s'%(field, band, split)
OVERWRITE = False

# Fields
# field 1: 1.25 -- 2.7 hr
if field == 'field1':
    lst_ends = np.array([1.25, 2.7]) / RAD2HR
# field 2: 4.5 -- 6.5 hr
elif field == 'field2':
    lst_ends = np.array([4.5, 6.5]) / RAD2HR
# field 3: 8.5 -- 10.75 hr
elif field == 'field3':
    lst_ends = np.array([8.5, 10.75]) / RAD2HR
else:
    print('Wrong field is given')   

ra_center_deg = np.degrees(np.mean(lst_ends)) # 29.6 for idr22 3:8, 24.2 for idr22 0:9
dec_center_deg = -30.7
ra_rng_deg = 45
n_ra = 90
dec_rng_deg = 16
n_dec = 32

sky_px = optimal_mapping_radec_grid.SkyPx()
px_dic = sky_px.calc_radec_pix(ra_center_deg, ra_rng_deg, n_ra, 
                        dec_center_deg, dec_rng_deg, n_dec)

print('Data type:', data_type)
print('Mapping para.:', sequence, field, band, split, ipol_arr)
print('Pixelization: %.1fdeg X %.1fdeg (%d X %d)'%(ra_rng_deg, dec_rng_deg, n_ra, n_dec), ', centering at (%.2fdeg, %.2fdeg).'%(ra_center_deg, dec_center_deg))
print('overwrite:', OVERWRITE)
print('Number of threads:', nthread)
print(OUTPUT_FOLDER)

def radec_map_making(files, ifreq, ipol,
                     p_mat_calc=True, 
                     select_ant=False):
    
    t0 = time.time()  
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
        '/h1c_idr22_%s_%s_%s_%.2fMHz_pol%d_radec_grid_RA%dDec%d.p'%(suffix, field, split, freq/1e6, ipol, 
                                                                              ra_rng_deg, dec_rng_deg)

        if OVERWRITE == False:
            if os.path.exists(file_name):
                print(file_name, 'existed, return.')
                return

        opt_map.set_a_mat(uvw_sign=1)
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
    #H1C part
    data_folder = '/nfs/esc/hera/H1C_IDR22/IDR2_2_pspec/v2/one_group/data'
    files = np.array(sorted(glob(data_folder+'/zen.grp1.of1.LST.*.HH.%s.uvh5'%suffix)))
    if field == 'field1':
        files = files[1:3]
    elif field == 'field2':
        files = files[3:6]
    elif field == 'field3':
        files = files[7:10]
    else:
        print('Wrong field is given.')
        
    print('%d Files being mapped:\n'%len(files), files)
    if band == 'band1':
        ifreq_arr = np.arange(175, 335, dtype=int) #band1
    elif band == 'band2':
        ifreq_arr = np.arange(515, 695, dtype=int) #band2
    else:
        raise RuntimeError('Wrong input for band.')   
    if sequence == 'forward':
        args = product(np.expand_dims(files, axis=0), ifreq_arr[:], ipol_arr)
    elif sequence == 'backward':
        args = product(np.expand_dims(files, axis=0), ifreq_arr[::-1], ipol_arr)
    else:
        raise RuntimeError('Sequence should be either forward or backward.')

    for args_t in args:
#         print(args_t)
        radec_map_making(*args_t)

    pool = multiprocessing.Pool(processes=nthread)
    pool.starmap(radec_map_making, args)
    pool.close()
    pool.join()
