# -*- coding: utf-8 -*-
"""
KSMB-SMB Satellite Workshop (June 27-29, IBS, Daejeon)
Tutorials for Recent Advances in Methods of Biomedical Mathematics 
Created on Mon Jun 24 14:08:19 2024
@author: Eder Zavala (with credits to Dr Jin Hyun Cheong and Professor Jon E. Froehlich)

As Beyonce said: "if you liked it, put a citation on it":

Grant, A. D., Upton, T. J., Terry, J. R., Smarr, B. L., & Zavala, E. (2022). 
Analysis of wearable time series data in endocrine and metabolic research. 
Current Opinion in Endocrine and Metabolic Research, 25, 100380.

Kim, D. W., Zavala, E., & Kim, J. K. (2020). 
Wearable technology and systems modeling for personalized chronotherapy. 
Current Opinion in Systems Biology, 21, 9-15.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import (inset_axes,InsetPosition,mark_inset)
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import matplotlib.dates as mdates
import seaborn as sns
# import glob, os
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore") 

# from pandas.plotting import lag_plot
# from pandas.plotting import autocorrelation_plot
from scipy.signal import hilbert, butter, filtfilt, find_peaks
from scipy.fftpack import fft,fftfreq,ifft
from statsmodels.graphics.tsaplots import plot_acf
# from scipy.stats import zscore
#from numpy import trapz
from scipy.integrate import simps
#from sklearn import metrics
from dtw import dtw, accelerated_dtw
# from matplotlib.dates import DateFormatter, HourLocator
# from datetime import datetime, timedelta

#%% ANALYSIS SETTINGS

# Save plots and output files?
save = 'n' # y/n

# Headers of time series to compare
# x_analyte = 'Activity'
# y_analyte = 'Light'
x_analyte = 'Glucose_1'
y_analyte = 'Glucose_2'

#%% PLOT SETTINGS AND AUXILIARY FUNCTIONS

plt.rcParams.update({'font.size': 15})

xa_color = 'tab:blue'
ya_color = 'indianred'

xa_ls = '-'
ya_ls = '-'

# Time lagged cross correlation
def crosscorr(datax, datay, lag=0, wrap=False, method='spearman'):  # Default values. <lag> is a range centered at zero.
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
        print(datax.corr(shiftedy))
    else:
        return datax.corr(datay.shift(lag), method='spearman')

# Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    c = filtfilt(b, a, data)
    return c

def set_ratio_ylim(ax1,ax2):
    yratio_lo = 0
    yratio_hi = 2

    ax1.set_ylim(yratio_lo,yratio_hi)
    ax2.set_ylim(yratio_lo,yratio_hi)
    ax1.set_ylabel(x_analyte+'/'+y_analyte)

def fmt_xaxes(ax):
    hours = mdates.HourLocator(interval=3)
    fmt = mdates.DateFormatter('%H')
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(fmt)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize='small')

#%% RUN

print()
print('RUNNING...')
print('Synchrony analysis of two dynamic hormone profiles')
print()
print('Comparing '+x_analyte+' vs '+y_analyte)

# Read list of files
# filelist=glob.glob('CGM_*.csv')
# print()
# print('Available data files: ', filelist)

# Select file to analyse
# file = 'MotionWatch_demo.csv'
file = 'CGM_demo.csv'

#%% LOAD DATA

# Read csv file into a dataframe, create a sub-dataframe with the hormones to compare
# data_df=pd.read_csv(file, parse_dates=[['Date','Time']], index_col=0, header=0, dayfirst=True)
data_df=pd.read_csv(file, parse_dates=['Date_Time'], index_col=0, header=0, dayfirst=True)
print()
print('-----------------------------------')
print('Analysing file: ' +str(file))
print('-----------------------------------')

# Filter by date
# data_df_filtered = data_df.loc[data_df.index.normalize() == pd.Timestamp('2019-05-16')]

# Select two hormones to compare (e.g. plasma vs MD)
df = data_df[[x_analyte, y_analyte]].copy()
# df = data_df_filtered[[x_analyte, y_analyte]].copy()
time = df.index

#%% CALCULATIONS

# Interpolate data (limit specifies the max number of consecutive NaNs to fill)
df = df.interpolate(limit=3,limit_area='inside').sort_index()

# Ratios
df_ratio = df[x_analyte]/df[y_analyte].dropna()

# Area Under the Curve
xa = np.array(df[x_analyte].dropna())
ya = np.array(df[y_analyte].dropna())
x_auc = simps(xa, dx=15)
y_auc = simps(ya, dx=15)
auc_ratio = x_auc/y_auc
print()
print(f"Simpson's AUC for {x_analyte} =",x_auc)
print(f"Simpson's AUC for {y_analyte} =",y_auc)
print(f"AUC ratio {x_analyte}/{y_analyte} =",auc_ratio)

window_size = 4  # Time step is 10 min so a window_size=6 equals 1 hr

# Zscores
df_zscore = (df - df.mean())/df.std() #standardise the data in each column

# Zscore ratios
df_zratio = (df_zscore[x_analyte]/df_zscore[y_analyte].dropna())

# Zscore rolling mean
df_zrm = df_zscore.rolling(window_size, center=True, win_type='gaussian').mean(std=6)

# Compute rolling window synchrony
rolling_corr = df_zrm[x_analyte].rolling(window=window_size, center=True).corr(df_zrm[y_analyte])

# 1st order discrete differentiation of Z-score
df_zdiff = df_zscore.diff()

# Spearman correlation coefficient and p-value
r, p = stats.spearmanr(df, nan_policy='omit')
print()
print(f"Spearman r = {r} and p-value = {p}")

# Spearman correlation of first order differentials
sc1d = df.diff().corr(method='spearman')
print()
print(f"Spearman correlation of 1st order differentials: \n {sc1d}")

# Sample rate and desired cutoff frequencies (in Hz)
fs = 3.
lowcut  = .01
highcut = .5
order = 1

# Fourier spectral analysis
df_fft = fft(df)  # Fast Fourier Transform
df_psd = np.abs(df_fft)**2  # Power Spectrum Density
natfreqs = fftfreq(len(df_psd), 1)  # Extract frequencies from the PSD
j = natfreqs > 0  # Keep positive elements only

# Find frequency peaks using FFT
# max_freq = natfreqs[np.argmax(np.log10(df_psd[:,0]))]  # Using x_analyte to calculate this
peaks,_ = find_peaks(np.log10(df_psd[:,0]),prominence=0.1)  # Using x_analyte to calculate this
loc_max = natfreqs[peaks]
x = loc_max > 0  # Keep positive elements only
# loc_max = loc_max[x]
max_freq = loc_max[x][0]

# T_max = 1/loc_max[np.where(loc_max == max_freq)[0][0]-1]
# T_min = 1/loc_max[np.where(loc_max == max_freq)[0][0]+1]
# T_min2 = 1/loc_max[np.where(loc_max == max_freq)[0][0]+2]

# Recover original signal using FFT
df_fft_bis = df_fft.copy()
df_slow = np.real(ifft(df_fft_bis))  # Inverse FFT

# Filter freqs from original signal
# df_fft_bis[np.abs(natfreqs) > max_freq] = 0  # Cut frequencies higher than the fundamental frequency
df_fft_bis[np.abs(natfreqs) < max_freq] = 0  # Cut frequencies lower than the fundamental frequency
df_slow_bis = np.real(ifft(df_fft_bis))  # Inverse FFT

#%% PLOT: Interpolated data

fig1 = plt.figure(1, facecolor='white')
plt.clf()

ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212,sharex=ax1)

ax1.plot(df.index,df[x_analyte],xa_ls,lw=2,color=xa_color,label=x_analyte)
ax2.plot(df.index,df[y_analyte],ya_ls,lw=2,color=ya_color,label=y_analyte)

ax1.set_xlabel('')
ax2.set_xlabel('')

ax1.set_ylabel('[mmol/L]')
ax2.set_ylabel('[mmol/L]')
ax1.set_xlim(min(df.index),max(df.index))
ax1.legend(frameon=False,fontsize=10,loc='upper left')
ax2.legend(frameon=False,fontsize=10,loc='upper left')
fmt_xaxes(ax1)
fmt_xaxes(ax2)

plt.xlabel('Clock time (hr)')
plt.text(-0.12, -0.4, 'Source: '+str(file), transform=ax2.transAxes, fontsize=6)

fig1.tight_layout()
plt.show()

#%% PLOT: Ratio of x_analyte:y_analyte

fig2 = plt.figure(2, facecolor='white')
plt.clf()

ax1 = fig2.add_subplot(111)

ax2 = plt.axes([0,0,1,1])    # Create a set of inset Axes: these should fill the bounding box allocated to them
ip = InsetPosition(ax1,[0.9,0.0,0.3,1.0])    # Manually set the position and relative size of the inset axes within ax1
ax2.set_axes_locator(ip)

ax1.plot(df.index,df_ratio,'-',lw=2,color='k',label='Ratio')
ax1.axhline(1.0, color='dimgrey', linestyle='--',lw=1)
ax2 = df_ratio.plot.box(label='')
ax2.axis('off')

ax1.set_xlabel('Clock time (hr)')
ax1.set_xlim(min(df.index),max(df.index))
ax1.legend(frameon=False,fontsize=10,loc='upper left')
set_ratio_ylim(ax1,ax2)
fmt_xaxes(ax1)

plt.text(0.8, 0.1, r'$\mu \pm \sigma = $'+str(round(df_ratio.mean(),2))+r' $\pm$ '+str(round(df_ratio.std(),2)), transform=ax2.transAxes, fontsize=10, rotation=90)
plt.text(-0.12, -0.17, 'Source: '+str(file), transform=ax1.transAxes, fontsize=6)

# fig2.tight_layout()
plt.show()

#%% PLOT: Zscores

fig3 = plt.figure(3, facecolor='white')
plt.clf()

ax1 = fig3.add_subplot(111)

ax1.plot(df.index,df_zscore[x_analyte],xa_ls,lw=2,color=xa_color,label=x_analyte)
ax1.plot(df.index,df_zscore[y_analyte],ya_ls,lw=2,color=ya_color,label=y_analyte)

ax1.set_xlabel('')
ax1.set_ylabel('Z-score')
ax1.set_xlim(min(df.index),max(df.index))
ax1.legend(frameon=False,fontsize=10,loc='upper left')
fmt_xaxes(ax1)

plt.xlabel('Clock time (hr)')
plt.text(-0.12, -0.17, 'Source: '+str(file), transform=ax1.transAxes, fontsize=6)

fig3.tight_layout()
plt.show()

#%% PLOT: Rolling mean Z-score and Pearson r

fig4 = plt.figure(4, facecolor='white')
plt.clf()

ax1 = fig4.add_subplot(211)
ax2 = fig4.add_subplot(212,sharex=ax1)

overall_pearson_r = df.corr().iloc[0,1]
print(f"Overall Pearson r: {overall_pearson_r}")

ax1.plot(df.index,df_zrm[x_analyte],xa_ls,lw=2,color=xa_color,label=x_analyte)
ax1.plot(df.index,df_zrm[y_analyte],ya_ls,lw=2,color=ya_color,label=y_analyte)
ax2.plot(df.index,rolling_corr,'-',lw=2,color='k',label='')

ax1.set_xlabel('')
ax2.set_xlabel('')
ax1.set_ylabel('Z-score ($\overline{\mu}$) \n $_{RW='+str(window_size*15)+' min}$')
ax2.set_ylabel(r'Pearson $r_{X,Y}$')
ax1.set_xlim(min(df.index),max(df.index))
ax1.legend(frameon=False,fontsize=10,loc='upper left')
ax1.set_title(f'Overall Pearson r = {np.round(overall_pearson_r,2)}', fontdict={'fontsize': 10})

fmt_xaxes(ax1)
fmt_xaxes(ax2)

plt.xlabel('Clock time (hr)')
plt.text(-0.15, -0.4, 'Source: '+str(file), transform=ax2.transAxes, fontsize=6)

fig4.tight_layout()
plt.show()

#%% PLOT: 1st-order diff

fig5 = plt.figure(5, facecolor='white')
plt.clf()

ax1 = fig5.add_subplot(111)

# First order differencing (discrete derivative) to remove diurnal trend
ax1.plot(df.index,df_zdiff[x_analyte],xa_ls,lw=2,color=xa_color,label=x_analyte)
ax1.plot(df.index,df_zdiff[y_analyte],ya_ls,lw=2,color=ya_color,label=y_analyte)
ax1.hlines(0,min(df.index),max(df.index),lw=1,color='k',linestyle='--')

ax1.set_xlabel('')
ax1.set_ylabel('1st(O) diff Z-score')
ax1.set_xlim(min(df.index),max(df.index))
ax1.legend(frameon=False,fontsize=10,loc='best')
fmt_xaxes(ax1)

plt.xlabel('Clock time (hr)')
plt.text(-0.18, -0.17, 'Source: '+str(file), transform=ax1.transAxes, fontsize=6)

fig5.tight_layout()
plt.show()

#%% PLOT: Autocorrelations

fig6 = plt.figure(6, facecolor='white')
plt.clf()

ax1 = fig6.add_subplot(211)
ax2 = fig6.add_subplot(212,sharex=ax1)

plot_acf(df[x_analyte].values,ax=ax1,missing='conservative',title='',label=x_analyte)
plot_acf(df[y_analyte].values,ax=ax2,missing='conservative',title='',label=y_analyte)

ax1.set_ylabel(x_analyte)
ax2.set_ylabel(y_analyte)
ax1.tick_params(labelbottom=False)
plt.xlabel('Lags (1 lag = 15 min)')
plt.text(0.7, 0.7, '95% confidence cone', transform=ax2.transAxes, fontsize=8)
plt.text(-0.09, -0.4, 'Source: '+str(file), transform=ax2.transAxes, fontsize=6)

fig6.tight_layout()
plt.show()

#%% PLOT: Time lagged cross correlation to find peak synchronicity (assumes 10 minute interpolation)

fig7 = plt.figure(7, facecolor='white')
plt.clf()

ax1 = fig7.add_subplot(111)

# Load data
datax = df.dropna()[x_analyte]
datay = df.dropna()[y_analyte]

# Resample at 1 min resolution and interpolate using a cubic spline
dataxx = datax.resample('T').interpolate('cubic')
datayy = datay.resample('T').interpolate('cubic')

# Set number of samples to shift in each direction
samples = 180  # Time step is now 1 min so samples=60 equals 1 hr
s_space = range(-samples,samples+1) # Sampling space, centered at 0 shift

# Calculate time lagged cross correlation (tlcc)
tlcc = [crosscorr(dataxx,datayy,lag) for lag in s_space]

centretick = s_space.index(0)
peaktick = np.argmax(tlcc)
offset = (abs(centretick-peaktick))

ax1.plot(tlcc)
ax1.axvline((centretick),color='k',linestyle='--',label='Centre')
ax1.axvline(peaktick,color='r',linestyle='--',label='Peak sync')

ticks = range(0,int(len(s_space)),60)
ticklabels = range(-(len(s_space)//(2*60)),(len(s_space)//(2*60))+1,1)
ax1.set_xticks(ticks)
ax1.set_xticklabels(ticklabels)
ax1.set_xlabel('Time lag (hr)')
ax1.set_ylabel('Spearman r')
ax1.legend(frameon=False,fontsize=10,loc='upper right')

if centretick > peaktick:
    plt.text(0.05, 0.85, x_analyte+' leads \n by '+str(offset)+' min', transform=ax1.transAxes, fontsize=10)
else:
    plt.text(0.05, 0.85, y_analyte+' leads \n by '+str(offset)+' min', transform=ax1.transAxes, fontsize=10)
plt.text(-0.2, -0.17, 'Source: '+str(file), transform=ax1.transAxes, fontsize=6)

fig7.tight_layout()
plt.show()

#%% PLOT: Windowed time lagged cross correlation

fig8 = plt.figure(8, facecolor='white', figsize=(10,8))
plt.clf()

ax1 = fig8.add_subplot(111)

# Windowed time lagged cross correlation (wtlcc)
# Set number of samples to shift in each direction
samples = 12  # Time step is 15 min so sample=4 equals 1 hr
s_space = range(-samples,samples+1) # Sampling space, centered at 0 shift
centretick = s_space.index(0)
splits = 4
samples_per_split = int(len(df.dropna())/splits)
wtlcc=[]
for t in range(0, splits):
    datax = df.dropna()[x_analyte].iloc[(t)*samples_per_split:(t+1)*samples_per_split]
    datay = df.dropna()[y_analyte].iloc[(t)*samples_per_split:(t+1)*samples_per_split]
    tlcc = [crosscorr(datax,datay,lag) for lag in s_space]
    wtlcc.append(tlcc)
wtlcc = pd.DataFrame(wtlcc)

ax1 = sns.heatmap(wtlcc,cmap='RdBu_r')
cbar = ax1.collections[0].colorbar
cbar.ax.set_ylabel('WTLCC')
ax1.axvline((centretick),color='w',linestyle='--',lw=3,label='Centre')

ticks = range(0,int(len(s_space)),6)
ticklabels = range(-(len(s_space)//(2*6)),(len(s_space)//(2*6))+1,1)
ax1.set_xticks(ticks)
ax1.set_xticklabels(ticklabels)
ax1.set_xlabel('Time lag (hr)')
ax1.set_ylabel('Window epochs')

plt.text(-0.07, -0.08, 'Source: '+str(file), transform=ax1.transAxes, fontsize=6)

fig8.tight_layout()
plt.show()

#%% PLOT: Rolling window time lagged cross correlation

fig9 = plt.figure(9, facecolor='white', figsize=(10,8))
plt.clf()

ax1 = fig9.add_subplot(111)

# Rolling windowed time lagged cross correlation (wtlcc)
# Set number of samples to shift in each direction
samples = int(len(df)/2)  # Time step is 15 min so sample=4 equals 1 hr
s_space = range(-samples,samples+1) # Sampling space, centered at 0 shift
centretick = s_space.index(0)

window_size = samples #determines the number of epochs
t_start = 0
t_end = t_start + window_size
step_size = 1
rwtlcc=[]
while t_end < int(len(df.dropna())):
    datax = df.dropna()[x_analyte].iloc[t_start:t_end]
    datay = df.dropna()[y_analyte].iloc[t_start:t_end]
    tlcc = [crosscorr(datax,datay,lag) for lag in s_space]
    rwtlcc.append(tlcc)
    t_start = t_start + step_size
    t_end = t_end + step_size
rwtlcc = pd.DataFrame(rwtlcc)

ax1 = sns.heatmap(rwtlcc,cmap='RdBu_r')
cbar = ax1.collections[0].colorbar
cbar.ax.set_ylabel('RWTLCC')
ax1.axvline((centretick),color='w',linestyle='--',lw=3,label='Centre')

ticks = range(0,int(len(s_space)),6)
# ticklabels = range(-(len(s_space)//(2*6)),(len(s_space)//(2*6))+1,1)
ticklabels = range(-(len(s_space)//(2*6)+1),(len(s_space)//(2*6))+1,1)
ax1.set_xticks(ticks)
ax1.set_xticklabels(ticklabels)

ax1.set_xlabel('Time lag (hr)')
ax1.set_ylabel('Epochs')

fig9.tight_layout()
plt.show()

#%% PLOT: Instantaneous Phase Synchrony

fig10 = plt.figure(10, facecolor='white', figsize=(10,8))
plt.clf()

ax1 = fig10.add_subplot(411)
ax2 = fig10.add_subplot(412,sharex=ax1)
ax3 = fig10.add_subplot(413,sharex=ax1)
ax4 = fig10.add_subplot(414,sharex=ax1)

#  Apply Butterworth filter, Hilbert transform, and calculate phase synchrony
datax = df.dropna()[x_analyte]
datay = df.dropna()[y_analyte]
c1 = butter_bandpass_filter(datax,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
c2 = butter_bandpass_filter(datay,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
al1 = np.angle(hilbert(c1),deg=False)
al2 = np.angle(hilbert(c2),deg=False)
phase_synchrony = 1-np.sin(np.abs(al1-al2)/2)
N = len(al1)

# Plot filtered data
ax1.plot(datax.index,c1,xa_ls,lw=2,color=xa_color,label=x_analyte)
ax2.plot(datax.index,c2,ya_ls,lw=2,color=ya_color,label=y_analyte)

# Plot angle
ax3.plot(datax.index,al1,xa_ls,lw=2,color=xa_color,label=x_analyte)
ax3.plot(datax.index,al2,ya_ls,lw=2,color=ya_color,label=y_analyte)

# Plot phase synchrony
ax4.plot(datax.index,phase_synchrony,'-',lw=2,color='k')

ax1.set_ylabel(x_analyte)
ax2.set_ylabel(y_analyte)
ax3.set_ylabel('Angle')
ax4.set_ylabel('IPS')
ax1.set_xlim(min(datax.index),max(datax.index))
fmt_xaxes(ax1)
fmt_xaxes(ax2)
fmt_xaxes(ax3)
fmt_xaxes(ax4)

plt.xlabel('Clock time (hr)')
plt.text(-0.1, -0.4, 'Source: '+str(file), transform=ax4.transAxes, fontsize=6)

fig10.tight_layout()
plt.show()

#%% PLOT: FFT

fig11 = plt.figure(11, facecolor='white', figsize=(10,7))
plt.clf()

ax1 = fig11.add_subplot(311)
ax2 = fig11.add_subplot(312)
ax2b = ax2.twinx()
ax3 = fig11.add_subplot(313)
ax3b = ax3.twinx()

ax1.plot(natfreqs[j],10*np.log10(df_psd[j,0]),xa_ls,lw=2,color=xa_color,label=x_analyte)
ax1.plot(natfreqs[j],10*np.log10(df_psd[j,1]),ya_ls,lw=2,color=ya_color,label=y_analyte)
# for i in loc_max:
    # ax1.axvline(i, color='k', linestyle='--',lw=1)
ax1.axvline(max_freq, color='r', linestyle='--',lw=2)
# ax1.axvline(1/T_max, color='k', linestyle='--',lw=1)
# ax1.axvline(1/T_min, color='k', linestyle='--',lw=1)
# ax1.axvline(1/T_min2, color='k', linestyle='--',lw=1)
ax1.set_xlabel('Frequency (hr$^{-1}$)')
ax1.set_ylabel('PSD (dB)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize='small')
ax1.legend(frameon=False,fontsize=10,loc='best')
ax1.text(0.82, 0.4, 'F$_{max}$ = '+str(round(max_freq,2))+' hr$^{-1}$', transform=ax1.transAxes, fontsize=10)

ax2.plot(df.index,df_slow[:,0],xa_ls,lw=2,color=xa_color,label=x_analyte)
ax2b.plot(df.index,df_slow[:,1],ya_ls,lw=2,color=ya_color,label=y_analyte)
ax2.set_ylabel(x_analyte+' iFFT')
ax2.set_xlim(min(df.index),max(df.index))
ax2b.set_ylabel(y_analyte+' iFFT')
fmt_xaxes(ax2)
fmt_xaxes(ax2b)
ax2b.spines['right'].set_visible(True)
# ax2.text(0.8, 0.5, 'Inverse FFT \nT = '+str(round(1/max_freq,2))+' hr at F$_{max}$ \nT$_{lo}$ = '+str(round(T_min,2))+' hr\nT$_{lo2}$ = '+str(round(T_min2,2))+' hr\nT$_{hi}$ = '+str(round(T_max,2))+' hr', transform=ax2.transAxes, fontsize=10)
ax2.text(0.82, 0.1, 'Inverse FFT \nT = '+str(round(1/max_freq,2))+' hr at F$_{max}$', transform=ax2.transAxes, fontsize=10)

ax3.plot(df.index,df_slow_bis[:,0],xa_ls,lw=2,color=xa_color,label=x_analyte)
ax3b.plot(df.index,df_slow_bis[:,1],ya_ls,lw=2,color=ya_color,label=y_analyte)
ax3.set_xlabel('Clock time (hr)')
ax3.set_ylabel(x_analyte+' iFFT')
ax3.set_xlim(min(df.index),max(df.index))
ax3b.set_ylabel(y_analyte+' iFFT')
fmt_xaxes(ax3)
fmt_xaxes(ax3b)
ax3b.spines['right'].set_visible(True)
# ax3.text(0.82, 0.1, 'Lo-pass cut \nT > '+str(round(1/max_freq,2))+' hr', transform=ax3.transAxes, fontsize=10)
ax3.text(0.82, 0.1, 'Hi-pass cut \nT < '+str(round(1/max_freq,2))+' hr', transform=ax3.transAxes, fontsize=10)
ax3.text(-0.07, -0.3, 'Source: '+str(file), transform=ax3.transAxes, fontsize=6)

fig11.tight_layout(h_pad=0)
plt.show()

#%% PLOT: DTW

fig12 = plt.figure(12, facecolor='white', figsize=(10,7))
plt.clf()

d1 = df['Glucose_1'].interpolate().values
d2 = df['Glucose_2'].interpolate().values
d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(d1,d2, dist='euclidean')

plt.imshow(acc_cost_matrix.T, origin='lower', cmap='summer', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlabel('Glucose_1')
plt.ylabel('Glucose_2')
plt.title(f'DTW Minimum Path with minimum distance: {np.round(d,2)}')
plt.show()

fig12.tight_layout(h_pad=0)
plt.show()

#%% SAVE FIGURES AND OUTPUT FILES

# os.chdir(save_path)
if save=='y':
    fig1.savefig(f' {x_analyte} vs {y_analyte} fig1 raw.pdf',bbox_inches='tight')
    fig2.savefig(f' {x_analyte} vs {y_analyte} fig2 ratio.pdf',bbox_inches='tight')
    fig3.savefig(f' {x_analyte} vs {y_analyte} fig3 zscore.pdf',bbox_inches='tight')
    fig4.savefig(f' {x_analyte} vs {y_analyte} fig4 zrmean.pdf',bbox_inches='tight')
    fig5.savefig(f' {x_analyte} vs {y_analyte} fig5 diff.pdf',bbox_inches='tight')
    fig6.savefig(f' {x_analyte} vs {y_analyte} fig6 acorr.pdf',bbox_inches='tight')
    fig7.savefig(f' {x_analyte} vs {y_analyte} fig7 tlcc.pdf',bbox_inches='tight')
    fig8.savefig(f' {x_analyte} vs {y_analyte} fig8 wtlcc.pdf',bbox_inches='tight')
    fig9.savefig(f' {x_analyte} vs {y_analyte} fig9 rwtlcc.pdf',bbox_inches='tight')
    fig10.savefig(f' {x_analyte} vs {y_analyte} fig10 ips.pdf',bbox_inches='tight')
    fig11.savefig(f' {x_analyte} vs {y_analyte} fig11 fft.pdf',bbox_inches='tight')
    fig12.savefig(f' {x_analyte} vs {y_analyte} fig12 dtw.pdf',bbox_inches='tight')

    with open('Output_'+x_analyte+'_'+y_analyte+'.txt', 'a') as f:
        print('\n', file=f)
        print('----------------------------', file=f)
        print('Source: '+str(file), file=f)
        print('----------------------------', file=f)
        print('\n', file=f)
        print('AUC for: '+x_analyte+' vs '+y_analyte, file=f)
        print(f'AUC {x_analyte} = {x_auc}', file=f)
        print(f'AUC {y_analyte} = {y_auc}', file=f)
        print(f'AUC ratio {x_analyte}/{y_analyte} = {auc_ratio}', file=f)
        print('\n', file=f)
        print('Zscore ratios for: '+x_analyte+' vs '+y_analyte, file=f)
        print(df_zratio.describe(), file=f)
        print('\n', file=f)
        print('Spearman ranks', file=f)
        print(f'Spearman r: {r} and p-value: {p}', file=f)
        print('\n', file=f)
        print('Spearman correlation 1st-O diff:', file=f)
        print(f'{sc1d}', file=f)
        print('\n', file=f)

print()
print('DONE!')
