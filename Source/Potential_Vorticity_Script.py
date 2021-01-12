#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:14:58 2021

@author: jamiemathews
"""

import numpy as np
import matplotlib.pyplot as plt
from Potential_Vorticity_Subroutes import expnd
from Potential_Vorticity_Subroutes import cor
from Potential_Vorticity_Subroutes import calc_rho
from Potential_Vorticity_Subroutes import calc_sigma0
from Potential_Vorticity_Subroutes import calc_vortyz
from Potential_Vorticity_Subroutes import calc_vortyy
from Potential_Vorticity_Subroutes import calc_vortyx
from Potential_Vorticity_Subroutes import calc_PVt
from Potential_Vorticity_Subroutes import av_fw

# =============================================================================
# Importing Varriables
# =============================================================================

[glamt, gphit, glamu, gphiu, glamv, gphiv, glamf, gphif, e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f] = np.load('HGRnemo.npy',allow_pickle=True)
[gdept, gdepu, gdepv, gdepw, e3t, e3u, e3v, e3w, mbathy] = np.load('ZGRnemo.npy',allow_pickle=True)
[T, S] = np.load('TRAnemo.npy')
[U, V, W] = np.load('DYNnemo.npy')

#Turning e1,2 into type 3 tensors
e1u=expnd(e1u, len(e3t))
e1v=expnd(e1v, len(e3t))
e1f=expnd(e1f, len(e3t))
e1t=expnd(e1t, len(e3t))
e2u=expnd(e2u, len(e3t))
e2v=expnd(e2v, len(e3t))
e2f=expnd(e2f, len(e3t))
e2t=expnd(e2t, len(e3t))

#Assigning scale varriables at all points on the grid
e1fw=e1f
e2fw=e2f
e3fw=e3w
e1vw=e1v
e2vw=e2v
e3vw=e3w
e1uw=e1u
e2uw=e2u
e3uw=e3w
e3f=e3t

#Creating a mask for land points 
land=T/T
U=U*land
V=V*land
W=W*land

#Calculating pressure and density
RHO=calc_rho(S, T, gdept)
SIG=calc_sigma0(S, T)
SIG=SIG*land    #applying mask
RHO=RHO*land

#Calculating the coriolis parameter
[f,b]=cor(gphif)    #defined on fw points
f=expnd(f, len(e3t))    #turning into 3 tensors
b=expnd(b, len(e3t))

# =============================================================================
# Calculating the vorticities
# =============================================================================

gvori=calc_vortyx(V, np.zeros(np.shape(e3t)), e2vw, e3vw) 
gvorj=calc_vortyy(U, np.zeros(np.shape(e3t)), e1uw, e3uw) 
gvork=calc_vortyz(U, V, e1u, e2v, e1f, e2f)

# =============================================================================
# Calculaing the potential vorticity 
# =============================================================================

#Ertel's potentoal vorticity
[PV,PVi,PVj,PVk]=calc_PVt(gvori, gvorj+b, gvork+f, e1vw, e2uw, e3f, RHO, SIG)

#Planetary geostrophic potential vorticity
[PVf,PVfi,PVfj,PVfk]=calc_PVt(np.zeros(np.shape(e3t)),np.zeros(np.shape(e3t)), f, e1vw, e2uw, e3f, RHO, SIG)

# =============================================================================
# #Plotting the PV for the WBC
# =============================================================================

#Choosing level
ko=24   #depth
lo=50   #longitude
la=20   #latitude

#Choosing min/max + contour plot levels
minim=20
maxim=-1
thickness=0.3
levels = np.arange(1024.0, 1027, thickness)

#Lat-Long plot

plt.figure(1)
#Plotting isopycnals
sfield=np.ma.array(SIG[ko,:,:])
CTP=plt.contour(glamf,gphif,sfield,colors='white',levels=levels)#SIG contour plt
plt.clabel(CTP,inline=thickness, fontsize=10,fmt='%1.1f')
#Plotting colour mesh
field = np.ma.array(PV[ko,:,:])*1e10
field[np.isnan(field)] = np.ma.masked
C=plt.pcolormesh(glamt,gphit,field,cmap='jet',vmin=minim,vmax=maxim) #PV colourplt
cbar=plt.colorbar(C)
cbar.set_label('PV ($10^{-10}s^{-1}$)',rotation=270,labelpad=20, y=0.45)
#Labeling
plt.xlabel('Longitude $\lambda$')
plt.ylabel('Latitude $\phi$')
plt.title('PV for the Western Boundary Current at depth '+str(round(gdepw[ko,0,0],2))+'$m$')
plt.show()

#East to West Plots

#Creating 3 tensor grid
glamtp=expnd(glamt, len(e3t))
gphitp=expnd(gphit, len(e3t))
gdeptp=-np.flip(gdept,0)#flipping depth axis
PVp=np.flip(PV,0)


plt.figure(2)
#Plotting isopycnals
sfield=np.ma.array(np.flip(av_fw(SIG)[:,la,:],0))
CTP=plt.contour(glamtp[:,la,:],gdeptp[:,la,:],sfield,colors='white',levels=levels)
plt.clabel(CTP,inline=thickness, fontsize=10,fmt='%1.1f')
CTP.collections[0].set_label('Potential Density $kg m^{-3}$')
#Plotting colour mesh
field = np.ma.array(PVp[:,la,:])*1e10
field[np.isnan(field)] = np.ma.masked
C=plt.pcolormesh(glamtp[:,la,:],gdeptp[:,la,:],field,cmap='nipy_spectral',vmin=minim,vmax=maxim) 
cbar=plt.colorbar(C)
cbar.set_label('PV ($10^{-10}s^{-1}$) ',rotation=270,labelpad=20, y=0.45)
CUT=plt.plot([glamf[la,0], glamf[la,np.shape(glamf)[1]-2]], [-gdepw[ko,0,0], -gdepw[ko,0,0]], 'k-',color='red',label='Horizontal Cross Section', lw=2)      #drawing line accros
plt.ylim(-800,0)
#Labeling
leg=plt.legend(facecolor='black',loc='lower left',prop={'size': 13})
for text in leg.get_texts():
    text.set_color("white")
plt.xlabel('Longitude $\lambda$')
plt.ylabel('Depth $m$')
plt.title('Ertel PV at Latitude'+str(round(gphit[la,0],2))+ '\N{DEGREE SIGN}')
plt.show()



