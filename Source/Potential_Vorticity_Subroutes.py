#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:30:49 2021

@author: jamiemathews
"""

import numpy as np

# =============================================================================
# Utility functions
# =============================================================================

#Turning type 2 tensor into a type 3 tensor
def expnd(A,d):
    A=np.repeat(A[np.newaxis,:,:], d, axis=0)
    return A

#Replacing last index of tensor with null values
def null(e,d):
    E=e
    for i in range(1,d+1):
        E[np.shape(e)[0]-i,:,:]=None
        E[:,np.shape(e)[1]-i,:]=None
        E[:,:,np.shape(e)[2]-i]=None
    return E

# =============================================================================
# Density functions
# =============================================================================


def calc_rho(sa,ct,p0):
    #Computes in situ density using TEOS-10 (boussinesq) from Roquet et al. (2015)
    #Absolute salinity sa in kg/kg, conservative temp ct in deg C, pressure p0 in dbar
    SAu = 40*35.16504/35 
    CTu = 40 
    Zu = 1e4
    deltaS = 32
    R000 = 8.0189615746e+02 
    R100 = 8.6672408165e+02 
    R200 = -1.7864682637e+03
    R300 = 2.0375295546e+03 
    R400 = -1.2849161071e+03 
    R500 = 4.3227585684e+02
    R600 = -6.0579916612e+01 
    R010 = 2.6010145068e+01 
    R110 = -6.5281885265e+01
    R210 = 8.1770425108e+01
    R310 = -5.6888046321e+01 
    R410 = 1.7681814114e+01
    R510 = -1.9193502195e+00 
    R020 = -3.7074170417e+01 
    R120 = 6.1548258127e+01
    R220 = -6.0362551501e+01
    R320 = 2.9130021253e+01 
    R420 = -5.4723692739e+00
    R030 = 2.1661789529e+01 
    R130 = -3.3449108469e+01 
    R230 = 1.9717078466e+01
    R330 = -3.1742946532e+00 
    R040 = -8.3627885467e+00 
    R140 = 1.1311538584e+01
    R240 = -5.3563304045e+00 
    R050 = 5.4048723791e-01 
    R150 = 4.8169980163e-01
    R060 = -1.9083568888e-01 
    R001 = 1.9681925209e+01 
    R101 = -4.2549998214e+01
    R201 = 5.0774768218e+01 
    R301 = -3.0938076334e+01 
    R401 = 6.6051753097e+00
    R011 = -1.3336301113e+01 
    R111 = -4.4870114575e+00
    R211 = 5.0042598061e+00
    R311 = -6.5399043664e-01
    R021 = 6.7080479603e+00
    R121 = 3.5063081279e+00
    R221 = -1.8795372996e+00 
    R031 = -2.4649669534e+00 
    R131 = -5.5077101279e-01
    R041 = 5.5927935970e-01
    R002 = 2.0660924175e+00
    R102 = -4.9527603989e+00
    R202 = 2.5019633244e+00
    R012 = 2.0564311499e+00 
    R112 = -2.1311365518e-01
    R022 = -1.2419983026e+00 
    R003 = -2.3342758797e-02 
    R103 = -1.8507636718e-02
    R013 = 3.7969820455e-01

    SA = sa
    CT = ct
    Z = -p0

    ss = np.sqrt ( (SA+deltaS)/SAu );
    tt = CT / CTu;
    zz =  -Z / Zu;
    rz3 = R013 * tt + R103 * ss + R003
    rz2 = (R022 * tt+R112 * ss+R012) * tt+(R202 * ss+R102) * ss+R002
    rz1 = (((R041 * tt+R131 * ss+R031) * tt + (R221 * ss+R121) * ss+R021) * tt + ((R311 * ss+R211) * ss+R111) * ss+R011) * tt + (((R401 * ss+R301) * ss+R201) * ss+R101) * ss+R001
    rz0 = (((((R060 * tt+R150 * ss+R050) * tt + (R240 * ss+R140) * ss+R040) * tt + ((R330 * ss+R230) * ss+R130) * ss+R030) * tt + (((R420 * ss+R320) * ss+R220) * ss+R120) * ss+R020) * tt + ((((R510 * ss+R410) * ss+R310) * ss+R210) * ss+R110) * ss+R010) * tt +(((((R600 * ss+R500) * ss+R400) * ss+R300) * ss+R200) * ss+R100) * ss+R000
    r = ( ( rz3 * zz + rz2 ) * zz + rz1 ) * zz + rz0

    Zu = 1e4 
    zz = -Z / Zu
    R00 = 4.6494977072e+01 
    R01 = -5.2099962525e+00
    R02 = 2.2601900708e-01
    R03 = 6.4326772569e-02 
    R04 = 1.5616995503e-02
    R05 = -1.7243708991e-03
    r0 = (((((R05 * zz+R04) * zz+R03 ) * zz+R02 ) * zz+R01) * zz+R00) * zz
    rho = r0 + r

    return rho


def calc_sigma0(sp,ti):
        # Computes sigma0 using the code /home/users/atb299/CDFTOOLS_Nov18/src/eos.f90
        zt  = ti          # in situ temp (i.e. not ct)
        zs  = sp          # in situ salinity (i.e., not sa)
        zsr = np.sqrt(zs) # square root of interpolated S

        #Compute volumic mass pure water at atm pressure
        zr1 = ( ( ( ( 6.536332e-9*zt-1.120083e-6 )*zt+1.001685e-4)*zt -9.095290e-3 )*zt+6.793952e-2 )*zt+999.842594
        #Seawater volumic mass atm pressure
        zr2= ( ( ( 5.3875e-9*zt-8.2467e-7 )*zt+7.6438e-5 ) *zt-4.0899e-3 ) *zt+0.824493
        zr3= ( -1.6546e-6*zt+1.0227e-4 ) *zt-5.72466e-3
        zr4= 4.8314e-4
        zrau0 = 1000
        #Potential volumic mass (reference to the surface)
        sigma0 = ( zr4*zs + zr3*zsr + zr2 ) *zs + zr1 - zrau0

        return sigma0+1000
    
# =============================================================================
# Averaging functions
# =============================================================================

def av(S,d,p):
    #d=0,1,2 respectively corresponds to the x,y,z plane averaging
    #p=1 average forward +1/2, p=-1 average backward -1/2 
    S=null(S,1)
    if d==0: #move to vw
        m=0
        n=1
    elif d==1:  #move to uw
        m=0
        n=2
    elif d==2:  #move to f
        m=1
        n=2
    S1=np.roll(S,-p,n)
    S2=np.roll(S,-p,m)
    S3=np.roll(np.roll(S,-p,n),-p,m)
    S_av=(S+S1+S2+S3)/4
    S_av=np.roll(S_av,1-m,0)
    return S_av

def av_fw(S):
    #Averaging to the corner of the cube
    S1=np.roll(S,-1,0)  #w
    S2=np.roll(S,-1,1)  #v
    S3=np.roll(S,-1,2)  #u
    S4=np.roll(S1,-1,1) #vw
    S5=np.roll(S1,-1,2) #uw
    S6=np.roll(S2,-1,2) #f   
    S7=np.roll(S6,-1,0) #fw
    S_av=(S+S1+S2+S3+S4+S5+S6+S7)/8
    S_av=null(S_av,1)
    S_av=np.roll(S_av,1,0)
    return S_av

# =============================================================================
# vorticity Functions
# =============================================================================

#Calculating the coriolis parameter 
    
def cor(gphi):
    #given in deg
    omega=7.2921159e-5
    gphi=np.radians(gphi)
    return [2*omega*np.sin(gphi),2*omega*np.cos(gphi)]


#Calculating relative vorticities          
           
def calc_vortyz(U,V,e1u,e2v,e1f,e2f):
    dA=e1f*e2f
    eU=U*e1u
    eV=V*e2v
    du=np.roll(eU,-1,1)-eU
    dv=np.roll(eV,-1,2)-eV
    vort=(dv - du)/dA       #defined at f points 
    
    vort=null(vort,1)
    return vort

def calc_vortyy(U,W,e1uw,e3uw):
    du=-np.roll(U,-1,0)+U
    dw=np.roll(W,-1,2)-W
    vort=du/np.roll(e3uw,-1,0)-np.roll(dw/e1uw,-1,0)
    
    vort=null(vort,1)
    vort=np.roll(vort,1,0)   #bring back to uw(0) point
    return vort

    
def calc_vortyx(V,W,e2vw,e3vw):
    dv=-np.roll(V,-1,0)+V
    dw=np.roll(W,-1,1)-W
    vort=np.roll(dw/e2vw,-1,0)-dv/np.roll(e3vw,-1,0)
    
    vort=null(vort,1)
    vort=np.roll(vort,1,0)  #bring back to uw(0)
    return vort


# =============================================================================
# Potential vorticity function
# =============================================================================

def calc_PVt(gvori,gvorj,gvork,e1vw,e2uw,e3f,RHO,SIG):#,glamf,glamt,gphif,gphit):
    #Averaging potenital density to fw point
    SIGfw=av_fw(SIG)
    #Calculating the divergence
    dSi=-(np.roll(SIGfw,+1,2)-SIGfw)/e1vw
    dSj=-(np.roll(SIGfw,+1,1)-SIGfw)/e2uw
    dSk=(-np.roll(SIGfw,-1,0)+SIGfw)/e3f
    #PV formula
    PVi=-gvori*dSi
    PVj=-gvorj*dSj
    PVk=-gvork*dSk
    #Averaging over to t points
    PVi=av(PVi,0,-1)/RHO
    PVj=av(PVj,1,-1)/RHO
    PVk=av(PVk,2,-1)/RHO
        
    PV=PVk+PVj+PVi
    return [PV, PVi, PVj, PVk]
 