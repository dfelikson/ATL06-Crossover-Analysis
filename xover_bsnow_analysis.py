#!/usr/bin/env python
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
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
    cycles = ['03','04','05','06','07','08','09','10','11','12','13','14','15','16']
#    cycles = ['03','04','05','06']

    for cycle in cycles:
      tiles=glob.glob(xover_dir+'/c'+cycle+'_1_extra_segments/*.h5')
#      tiles=glob.glob(xover_dir+'/*.h5')
      with h5py.File(tiles[0],'r') as h5f:
          fields=[key for key in h5f['data_0'].keys()]

      xformer_pol2ll=pyproj.Transformer.from_crs(3031, 4326)
      D=[]
      meta=[]

      tile_re=re.compile('E(.*)_N(.*).h5')
      for tile in glob.glob(xover_dir+'/c'+cycle+'_1_extra_segments/'+wildcard+'.h5'): 
#      for tile in glob.glob(xover_dir+'/'+wildcard+'.h5'):                   
          m=tile_re.search(tile)
          lat, lon = xformer_pol2ll.transform(m.group(1),m.group(2))

          if (-75 >= lat >= -85) & (120 <= lon <= 140):
      
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
optical_depth = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
od_counter = 0
v['bsnow_od'][:,0]=np.ma.masked_invalid(v['bsnow_od'][:,0])
v['bsnow_od'][:,1]=np.ma.masked_invalid(v['bsnow_od'][:,1])
# Loop through a range of optical depths
for od in optical_depth:

  h_li_diff_meds = [0,0,0,0,0,0]
  print('Blowing snow optical depth > {:3.1f}'.format(od))

  for i in range(0,6,1):
  # Find rows in v where either:
  #  -> v['spot'][:,0] == i and v['bsnow_conf'][:,0] > 0 (spot i along track 1 with blowing snow)    
    valid_spot_0 = (valid & ((v['spot'][:,0] == i+1) & (v['bsnow_conf'][:,1] < 0) & (v['bsnow_conf'][:,0] > 0) & (v['bsnow_od'][:,0] > od)))
    valid_spot_1 = (valid & ((v['spot'][:,1] == i+1) & (v['bsnow_conf'][:,0] < 0) & (v['bsnow_conf'][:,1] > 0) & (v['bsnow_od'][:,1] > od)))
  

    # Find rows in v where either:
    #  -> v['spot'][:,0] == i and v['bsnow_conf'][:,0] > 0 (spot i along track 1 with blowing snow)
    #  -> v['spot'][:,1]
    if len(v['h_li'][valid_spot_0,0]) == 0 & len(v['h_li'][valid_spot_0,1]) == 0:
      continue
    else:   
      h_li_diff_0 = v['h_li'][valid_spot_0,0] - v['h_li'][valid_spot_0,1]
      h_li_diff_1 = v['h_li'][valid_spot_1,1] - v['h_li'][valid_spot_1,0]
      h_li_diff   = np.append(h_li_diff_0, h_li_diff_1)
      h_li_diff_meds[i] = np.median(h_li_diff) 
      print(' spot {:d} median height difference: {:+7.4f} m ({:6d} points)'.format(i+1, h_li_diff_meds[i], len(h_li_diff)))
  print("===================================================================")
  fig = plt.figure(dpi=300)
  ax = fig.add_subplot(111)
  tick_name = ['spot1','spot2','spot3','spot4','spot5','spot6']
  color_bar = ['blue','orange','grey','yellow','green','red']
  plt.bar(tick_name,h_li_diff_meds,color=color_bar)
  plt.xticks(tick_name) 
  plt.ylim([-0.13,0.00])
#  plt.legend(tick_name,loc=4)
  plt.title("clr-clr")
  plt.xlabel("laser spot")
  plt.ylabel("median delta height (m)")
  plt.savefig(plot_title + '_' + str(od_counter)  + '.png', bbox_inches='tight')
  od_counter = od_counter + 1
sys.exit()
#delta_h_valid = np.append(delta_h_valid_v0, delta_h_valid_v1)

# Compute Mean, Median and Median Absolute Deviation
#mean_delta_h = np.nanmean(delta_h_valid)
#median_delta_h = np.nanmedian(delta_h_valid)
#mad_delta_h = np.nanmedian(np.abs(delta_h_valid - np.median(delta_h_valid)))

#label = 'Delta Height - Mean=%.6f m, Median=%.6f m, MAD=%.6f m' % (mean_delta_h,median_delta_h,mad_delta_h)
# Add the subplot
fig = plt.figure(dpi=300)

# Create the default figure
#fig = Figure(figsize=[10, 7])
#FigureCanvas(fig)


ax = fig.add_subplot(111)
#plt.hist(delta_h_valid, bins = 50)

#delta_h_valid = delta['h_li'][valid_0]
#delta_h_valid_1 = delta['h_li'][valid_1]

print(delta_h_valid)
#bsnow_h_valid = np.append(bsnow_h_valid, bsnow_h_valid_1)
#delta_h_valid_2 = np.append(delta_h_valid, delta_h_valid_1)
#combined = np.vstack((bsnow_h_valid,delta_h_valid)).T
#print(np.shape(combined))

#plt.hist(delta_h_valid_2, bins = 50)
#plt.hist(delta_h_valid, bins = 50)
#plt.scatter(bsnow_h_valid,delta_h_valid, s=2., cmap='RdBu')
#plt.scatter(bsnow_h_valid_1,delta_h_valid_1, s=2., cmap='RdBu')
#ax1 = sns.heatmap(combined, vmin=0, vmax=1)
tick_name = ['gt1','gt2','gt3','gt4','gt5','gt6']
plt.bar(tick_name,delta_h_valid)
plt.xticks(tick_name)


plt.title(plot_title)
plt.xlabel("groundtracks")
plt.ylabel("delta height (m)")
#plt.text(1.5,1.5, label, size=10, ha="center")
#import pdb; pdb.set_trace()
plt.savefig(plot_title + '.png', bbox_inches='tight')
#figure = ax1.get_figure()
#figure.savefig('bsnow_heatmap.png', dpi=400)
sys.exit()
