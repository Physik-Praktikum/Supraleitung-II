import os.path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import scipy.optimize as opt    # Funktionen für Anpassung


def Dampfdruck(p,dp):
	p_0 = -123.71036725 #mbar
	H = 34173.12824798 # J/mol
	dp_0 = 0.961912809262 #mbar
	dH = 1027.24010516 # J/mol
	T = H/(R*np.log(abs(p/p0)))
	dT = np.sqrt((dH/(R*np.log(abs(p/p0))))**2 + (H*dp_0/(p*R*(np.log(abs(p/p0)))**2))**2 + (H*dp/(R*p*(np.log(abs(p/p0)))**2))**2)
	return T,dT
	

def Germanium(R):
	R_0 = 223.779909858 # Ohm
	B = 7.38647452429 # K
	dR_0 = 12.6201340921 # Ohm
	dB = 0.127791826985 #dB
	T = B/(np.log(R/R_0))
	dT = np.sqrt((dB/(np.log(R/R_0)))**2 + ((B*dR_0)/(R*(np.log(R/R_0))**2))**2)
	return T, dT


Dateien = ["160120a.dat","160120c.dat","160120d.dat","160120e.dat","160120f.dat","160120g.dat","160120h.dat","160120i.dat","160120j.dat","160120k.dat",
"160120l.dat","160120m.dat","160120n.dat","160120o.dat","160120p.dat","160120q.dat","160120r.dat","160120s.dat","160120t.dat","160120u.dat","160120v.dat","160120w.dat",
"160120x.dat","160120y.dat","160120z.dat","160120A.dat","160120B.dat","160120C.dat"]

ps_fit = []
dps_fit = []
Ts_fit = []
dTs_fit = []
Ts_Germ = []
dTs_Germ = []

for i in range(len(Dateien)):
	p, T, Ge, I, U, iwas = np.loadtxt(Dateien[i],skiprows=5,unpack=True)
	p_mean = p.mean()
	dp = p.std()
	T_dampf, dT_dampf = Dampfdruck(p_mean,dp)
	ps_fit.append(p_mean)
	dps_fit.append(dp)
	Ts_fit.append(T_dampf)
	dTs_fit.append(dT_dampf)
	
	T_Germ, dT_Germ = Germanium(Ge)
	Ts_Germ.append(T_Germ)
	dTs_Germ.append(dT_Germ)
	
ps_fit = np.array(ps_fit)
dps_fit = np.array(dps_fit)
Ts_fit = np.array(Ts_fit)
dTs_fit = np.array(dTs_fit)
Ts_Germ = np.array(Ts_Germ)
dTs_Germ = np.array(dTs_Germ)
