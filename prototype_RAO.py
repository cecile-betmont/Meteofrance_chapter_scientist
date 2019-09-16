#!/usr/bin/env python
# coding: utf-8

''' first prototype to demonstrate how python ingests, treats and plots data for IPCC Chapter 8 '''
# [1] Python downloads simulations from CMIP-6 archive
#		Here, water evaporation from soil [kg m-2 s-1] from the IPSL-CM6A-LR model on relative coarese grid (250 km), covering the period 1850/01-2149/12 is used.
# [2] Computations of multi-year means using python's CDO interface
#		1995-2014 is used as reference period
#		relative mean changes to this reference period are computed for periods 2021-2040, 2041-2060, 2081-2100
# [3] Plotting of relative mean changes in %, using python's matplotlib and pylab libraries
# [4] Plotting of (smoothed) time series of global evaporation flux
# 
# C Thales Services, Labege, France. 13/09/2019. 
# ------------------------------------------------------------------------




''' import libraries and modules '''
# [1] load modules providing a portable way of using operating system dependent functionality
import os
import glob

# [2] load wget module to download files
import wget 

# [3] load numerical tools (numpy & pandas)  and graphic/plotting moduls (pylab, matplotlib)
import pandas as pd
import numpy as np 
from pylab import * 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.basemap import Basemap, cm, shiftgrid

# [4] Brewer colors
from palettable.colorbrewer.diverging import BrBG_11 # select colors for diverging quantities. Follows visual guidelines of IPCC!

# [5] import netcdf4 library
from netCDF4 import Dataset
import netCDF4

# [6] import Python interface to CDO
from cdo import *
cdo=Cdo()






''' set some parameters '''
tmin = [1995,2021,2041,2081] # time periods of multi-year averages. Here, 1995-2014 is reference period
tmax = [2014,2040,2060,2100]
dt   = 1
fontsize_cb = 6 # colorbar fontsize in plots 
fontsize_tt = 8 # title fontise
dirpath = os.getcwd() # set current directory as working directory 
boundmax = 10 # upper limit of relative change in colorbar in %
boundmin = -10 # lowerlimit of relative change in colorbar in %
step     = 1 # step in colorbar




''' get CMIP6 simulations '''
url="http://vesg.ipsl.upmc.fr/thredds/fileServer/cmip6/CMIP/IPSL/IPSL-CM6A-LR/abrupt-4xCO2/r1i1p1f1/Lmon/evspsblsoi/gr/v20190118/evspsblsoi_Lmon_IPSL-CM6A-LR_abrupt-4xCO2_r1i1p1f1_gr_185001-214912.nc"
filename=wget.download(url)
#filename='evspsblsoi_Lmon_IPSL-CM6A-LR_abrupt-4xCO2_r1i1p1f1_gr_185001-214912.nc'
dataset=Dataset(filename)







''' multi year averages using CDO'''
avg_files = []
# split file into year
cdo.splityear(input=filename,output='tfiles_')
# get all generated files
tempfiles=(glob.glob("tfiles_*.nc"))
for i in range(len(tmin)):
	avg_period=np.arange(tmin[i],tmax[i]+dt,dt)
	avg_file=[]
	for period in avg_period:
		file=next((s for s in tempfiles if 'tfiles_'+str(period)+'.nc' in s), None)
		avg_file.append(file)
	cdo.cat(input=avg_file,output='out.nc')

	outfile='avg_'+str(avg_period[0])+'_'+str(avg_period[len(avg_period)-1])+'.nc4'
	cdo.timmean(input='out.nc',output=outfile)
	avg_files.append(outfile)
# clean unused files
os.remove(dirpath+"/"+"out.nc")
for file in tempfiles:
	os.remove(dirpath+"/"+str(file))
for file in (glob.glob("avg_*.nc")):
	os.remove(dirpath+"/"+str(file))






''' plot relative changes in mean Water Evaporation from Soil'''
# read reference data
dataset= Dataset(avg_files[0])
print dataset.file_format
print dataset.Conventions
ref    = dataset.variables['evspsblsoi'][:]
ref    = np.ma.array(ref, mask=ref==ref.fill_value)

# define same colorbar for all plots
cmap = BrBG_11.mpl_colormap 
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
           'Custom cmap', cmaplist, cmap.N)
bounds = np.arange(boundmin, boundmax+step, step)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# calculate relative change to reference period and plot map
for i in (x+1 for x in range(len(avg_files[1:]))):
	
	# read mult-year averages	
	dataset= Dataset(avg_files[i])
	var    = dataset.variables['evspsblsoi'][:]
	var    = np.ma.array(var, mask=var==var.fill_value)
	lon    = dataset.variables['lon'][:]
	lat    = dataset.variables['lat'][:]

	# relative difference with respect to reference period
	diff   = np.squeeze(100.*(var-ref)/ref)

	# shift longitudes to [-180,180] grid required for basemap
	diff, lon = shiftgrid(180., diff, lon, start=False)

	# --------------------------------
	# plot with basemap and matplotlib
	fig = plt.figure(figsize=(6,6))

	m = Basemap(llcrnrlon=-180,llcrnrlat=-90,urcrnrlon=180,urcrnrlat=90,
    	        rsphere=(6378137.00,6356752.3142), 
               	resolution='l',projection='robin', 
	           	lat_0=0,lon_0=0,lat_ts=10)

	# draw coastlines and country boundaries, edge of map.
	m.drawcoastlines(linewidth=0.2, linestyle='solid', color='k')
	m.drawcountries(linewidth=0.2, linestyle='solid', color='k')

	# convert lat/lon to map projection coordinates yi/xi
	lat, lon= np.meshgrid(lat, lon)
	xi, yi  = m(lon, lat)

	# plot data on map. Here, no interpolation or smoothing
	cs = m.pcolormesh(xi,yi,np.transpose(diff), cmap=cmap, norm=norm)

    # Add Colorbar to plot
	cbar = m.colorbar(cs, location='bottom', pad="2%")
	cbar.set_label('%',fontsize=fontsize_cb )
	cbar.ax.tick_params(labelsize=fontsize_cb,direction='in',length=9) 
	cbar.set_ticks(bounds)
	cbar.set_ticklabels(bounds)

	# Add Title to plot
	title='Change in Water Evaporation from Soil ('+str(tmin[i])+'-'+str(tmax[i])+')'
	plt.title(title,fontsize=fontsize_tt)

	# save figure to file
	plt.savefig(((title.replace(" ", "_")).replace("(", "")).replace(")", "")+'.png', dpi=350,bbox_inches='tight')   
	plt.close("all")
	
	# ------- end of plot ----------


# clean unused files
for file in (glob.glob("avg_*.nc4")):
	os.remove(dirpath+"/"+str(file))








''' plot time series of global evaporation flux'''
# read the data
dataset= Dataset(filename)
var    = dataset.variables['evspsblsoi'][:]
var    = np.ma.array(var, mask=var==var.fill_value)
lon    = dataset.variables['lon'][:]
lat    = dataset.variables['lat'][:]
time   = dataset.variables['time']
time   = netCDF4.num2date(time[:], time.units, time.calendar) # convert "hours since" to datetime object

# compute global mean for each month
global_mean_var = []
for i in range(len(time)):
	global_mean_var.append(np.ma.mean(var[i,:,:]))

# convert global mean time series into a pandas time series
d = pd.Series(global_mean_var, time)

# apply moving average. 1-yr window
m = pd.Series(d).rolling(window=12).mean()
e = pd.Series(d).rolling(window=12).std()

# plot time series
fig = plt.figure(figsize=(6,6))
ax = plt.gca()

ax.spines["top"].set_visible(False)  # Remove the plot frame lines. 
ax.spines["right"].set_visible(False)  

ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 

plt.plot(time,m,linestyle='-',linewidth=1,color='black') # line plot
plt.fill_between(time,  m-e, m+e,color='lightgrey') # shading

# yaxis
ax.tick_params('both', length=10, labelsize=fontsize_cb)
ax.ticklabel_format(style='sci',axis='y')
class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)
plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.0e"))
ax.set_ylabel(r'$\mathrm{kg \thickspace m^{-2} s^{-1}}$',fontsize=fontsize_cb)  

# xaxis
ax.set_xlim(pd.Timestamp('1995-01-01'),pd.Timestamp('2100-12-31')) # limit plot range to 1995-01-01 to 2100-12-31
dates = pd.date_range('1995', '2100', freq=pd.DateOffset(years=10)) # a tick every 10 years
ax.xaxis.set_ticks(dates)
ax.xaxis.set_ticklabels(dates.strftime('%Y'))

# title
plt.title('Global water evaporation flux from soil (1995-2100)',fontsize=fontsize_tt)

# aspect ratio of figure
def fixed_aspect_ratio(ax1,ratio):
    '''
    Set a fixed aspect ratio on matplotlib plots
    regardless of axis units
    '''
    x0,x1 = ax1.get_xlim()
    y0,y1 = ax1.get_ylim()
    ax1.set_aspect(ratio*abs(x1-x0)/abs(y1-y0))
fixed_aspect_ratio(ax,.3)

# save figure to file
plt.savefig('TimeSeries_WaterEvap_1995-2100.png', dpi=350,bbox_inches='tight')   
plt.close("all")


