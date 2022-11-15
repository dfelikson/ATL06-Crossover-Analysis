#!/usr/bin/env python
import numpy as np

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import pyproj
import sys
import pointCollection as pc
import glob
import re
import h5py

def read_xovers(xover_dir, verbose=False, wildcard='*', r_limits=[0, 1.e7], delta_t_limit=2592000): ##{{{
    """
    read_xovers: Read all the crossover files in a directory (or matching a glob)

    Inputs:
        xover_dir: directory to search
        verbose: set to 'true' to see errors reading crossover files (default is silent)
        wildcard: default is '*'.  Specify to match selected files
        r_limits: limits for the polar stereographic coordinates of the tile files.  
            Default of [0, 1e7] eliminates crossovers with lat=0 (a common error in early versions)  
        delta_t_limit: set to limit time difference of crossovers, in seconds.  Default of 2592000 is 1 month

    Outputs:
        v: dict of nx2 matrices, giving ATL06 parameters interpolated to the crossover locations.  The first
            column gives the value for the first measurement in the crossover, the second the value from the second.
        delta: dict of nx2 matrices, giving ATL06 parameter differences between the crossover measurents, late minus early
        meta: metadata values at the crossovers
    """


    tiles=glob.glob(xover_dir+'/*.h5')
    
    with h5py.File(tiles[0],'r') as h5f:
        fields=[key for key in h5f['data_0'].keys()]

    xformer_pol2ll=pyproj.Transformer.from_crs(3031, 4326)
    D=[]
    meta=[]

    tile_re=re.compile('E(.*)_N(.*).h5')
    for tile in glob.glob(xover_dir+'/'+wildcard+'.h5'): 
        m=tile_re.search(tile)

        lat, lon = xformer_pol2ll.transform(m.group(1),m.group(2))
        if (-75 >= lat >= -85) & (120 <= lon <= 140):
          print("lat",lat,"lon",lon)  
          if m is not None:
              r2=np.float(m.group(1))**2+np.float(m.group(2))**2
              if (r2 < r_limits[0]**2) or (r2 > r_limits[1]**2):
                  continue
          try:
              this_D=[pc.data().from_h5(tile, field_dict={gr : fields}) for gr in ['data_0','data_1']]
              this_meta=pc.data().from_h5(tile, field_dict={None:['slope_x', 'slope_y','grounded']})
              if delta_t_limit is not None:
                  good=np.abs(this_D[1].delta_time[:,0]-this_D[0].delta_time[:,0]) < delta_t_limit
                  for Di in this_D:
                      Di.index(good)
                  this_meta.index(good)

          except KeyError:
              if verbose:
                  print("failed to read " + tile)
              continue

          D.append(this_D)
          meta.append(this_meta)
#        print(m.group(1),m.group(2))
#            if north:
#      polarEPSG=3413
#    else:
#      polarEPSG=3031

    meta=pc.data().from_list(meta)
    v={}
    for field in fields:
        vi=[]
        for Di in D:
            vi.append(np.r_[[np.sum(getattr(Di[ii], field)*Di[ii].W, axis=1) for ii in [0, 1]]])
        v[field]=np.concatenate(vi, axis=1).T
    delta={field:np.diff(v[field], axis=1) for field in fields}
    bar={field:np.mean(v[field], axis=1) for field in fields}
    return v,  delta,  bar, {key:getattr(meta, key) for key in meta.fields}
##}}}
#-------------------------------------------------------------------------------
# main code.  Parse the only input argument (the crossover directory)

xover_dir=sys.argv[1]
plot_title=sys.argv[2]
ATL06_fields=['BP','LR','W','cycle_number','rgt','h_li','h_li_sigma','x','y','spot']

v, delta, bar, meta = read_xovers(xover_dir)#, wildcard='E400_N-500')
meta['slope_mag']=np.abs(meta['slope_x']+1j*meta['slope_y'])

snow_conf = (bar['atl06_quality_summary'][:]<0.01) & (meta['slope_mag'][:]<0.02)
valid = (bar['atl06_quality_summary'][:]<0.01) & (meta['slope_mag'][:]<0.02)
valid_0 = (valid & (v['bsnow_conf'][:,0] > 0) & (v['bsnow_conf'][:,1] < 0))
valid_1 = (valid & (v['bsnow_conf'][:,1] > 0) & (v['bsnow_conf'][:,0] < 0))


bsnow_h_valid = v['bsnow_h'][valid_0,0]
delta_h_valid = delta['h_li'][valid_0]
bsnow_h_valid_1 = v['bsnow_h'][valid_1,1]
delta_h_valid_1 = delta['h_li'][valid_1]

#print(np.shape(bsnow_h_valid_0),np.shape(bsnow_h_valid_1))
#print(np.shape(delta_h_valid_0),np.shape(delta_h_valid_1))

#for i in range(bsnow_h_valid_1[:,]):
#    bsnow_h_valid = bsnow_h_valid + bsnow_h_valid_1[i]

#for i in range(delta_h_valid_1):
#    delta_h_valid = delta_h_valid + delta_h_valid_1[i]

bsnow_h_valid = np.append(bsnow_h_valid, bsnow_h_valid_1)
delta_h_valid = np.append(delta_h_valid, delta_h_valid_1)
# Compute Mean, Median and Median Absolute Deviation
mean_bsnow = np.nanmean(np.abs(delta_h_valid))
median_bsnow = np.nanmedian(np.abs(delta_h_valid))
mad_bsnow = np.nanmedian(np.abs(delta_h_valid - np.median(delta_h_valid)))
#mean_bsnow_1 = np.nanmean(np.abs(bsnow_h_valid_1))
#median_bsnow_1 = np.nanmedian(np.abs(bsnow_h_valid_1))
#mad_bsnow_1 = np.median(np.abs(delta_h_valid_1 - np.median(delta_h_valid_1)))

#mean_bsnow = np.c_[mean_bsnow_0, mean_bsnow_1]
#median_bsnow = np.c_[median_bsnow_0, median_bsnow_1]
#mad_bsnow = np.c_[mad_bsnow_0, mad_bsnow_1]
#print("Mean",mean_bsnow,"Median",median_bsnow_0,"Median Abs. Dev 0",mad_bsnow_0)
#print("Mean_1",mean_bsnow_1,"Median_1",median_bsnow_1,"Median Abs. Dev 1",mad_bsnow_1)
label = 'blowing snow height\nMean track=%.6f, Median track=%.6f, MAD track=%.6f' % (mean_bsnow,median_bsnow,mad_bsnow)
# Add the subplot
fig = plt.figure(dpi=300)

plt.scatter(bsnow_h_valid,delta_h_valid, s=2., cmap='RdBu')
#plt.scatter(bsnow_h_valid_1,delta_h_valid_1, s=2., cmap='RdBu')

plt.title(plot_title)
plt.xlabel(label)
plt.ylabel("delta height")
#plt.text(1.5,1.5, label, size=10, ha="center")
#import pdb; pdb.set_trace()
plt.savefig(plot_title + '.png', bbox_inches='tight')
sys.exit()
