import os.path
import numpy as np
from matplotlib import pyplot as plt
import sympy as sy
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import scipy.optimize as opt    # Funktionen für Anpassung




def Dampfdruck(p,dp):
	R = 8.314
	H = -123.71036725 #J/mol
	p0 = 34173.12824798  # mbar
	dH = 0.961912809262 #J/mol
	dp_0 = 1027.24010516 # mbar
	T = H/(R*np.log(abs(p/p0)))
	dT = np.sqrt((dH/(R*np.log(abs(p/p0))))**2 + (H*dp_0/(p*R*(np.log(abs(p/p0)))**2))**2 + (H*dp/(R*p*(np.log(abs(p/p0)))**2))**2)
	return T,dT
	

def Germanium(R):
	R_0 = 223.779909858 # Ohm
	B = 7.38647452429 # K
	dR_0 = 12.6201340921 # Ohm
	dB = 0.127791826985 # K
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
Ts_real = []
dTs_real = []

Torr = 101325/760    # 1Torr sind soviele Pascal

for i in range(len(Dateien)):
	p, T, Ge, I, U, iwas = np.loadtxt(Dateien[i],skiprows=5,unpack=True)
	p_mean = p.mean()
	dp = p.std()
	T_dampf, dT_dampf = Dampfdruck(p_mean,dp)
	ps_fit.append(p_mean)
	dps_fit.append(dp)
	Ts_fit.append(T_dampf)
	dTs_fit.append(dT_dampf)
	
	Ge = Ge*8230.317112#*8285.6
	U_Germ = Ge.mean()
	T_Germ, dT_Germ = Germanium(U_Germ)
	Ts_Germ.append(T_Germ)
	dTs_Germ.append(dT_Germ)
	T_val = T.mean()
	dT = T.std()
	Ts_real.append(T_val)
	dTs_real.append(dT)
	
ps_fit = np.array(ps_fit)
dps_fit = np.array(dps_fit)
Ts_fit = np.array(Ts_fit)
dTs_fit = np.array(dTs_fit)
Ts_Germ = np.array(Ts_Germ)
dTs_Germ = np.array(dTs_Germ)
Ts_real = np.array(Ts_real)
dTs_real = np.array(dTs_real)

#Datei = open("Thermometer_Vergleich.dat","a")

#print("\\\\\\hline",file=Datei)
#print("p/mbar & dp/mbar & T_dampf/K & dT_dampf/K & T_Ge/K & dT_Ge/K & T_real/K & T_real/K \\\\\\hline",file=Datei)
#for i in range(len(ps_fit)):
#	print("{0:.3f} & {1:3.3f} & {2:3.3f} & {3:3.3f} & {4:3.3f} & {5:3.3f} & {6:3.3f} & {7:3.3f} \\\\\\hline".format(ps_fit[i],dps_fit[i],Ts_fit[i],dTs_fit[i],Ts_Germ[i],dTs_Germ[i],Ts_real[i],dTs_real[i]),file=Datei)

#Datei.close()

def Geradenfit(x,m,n):
	return m*x + n

def Kurvenfit(x,a,b,c,d,e):
	return np.exp(a*x + b) + c + d*x**2 + e*x**4 


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Ts_real,Ts_fit,"o",markersize=4,color="blue",label="Dampfdruck")
ax.plot(Ts_real,Ts_Germ,"o",markersize=4,color="red",label="Germanium")
ax.plot(Ts_real,Ts_real,"o",markersize=4,color="green",label="tatsächliche Temperatur")

p_opt, kov = opt.curve_fit(Geradenfit,Ts_real[3:13],Ts_fit[3:13],sigma=dTs_fit[3:13])
fit_x = np.linspace(0,4.35,1000)
ax.plot(fit_x,p_opt[0]*fit_x + p_opt[1],"--",color="blue")

vals_x = np.hstack([Ts_real[0:2],Ts_real[13:]])
vals_y = np.hstack([Ts_fit[0:2],Ts_fit[13:]])
errors = np.hstack([dTs_fit[0:2],dTs_fit[13:]])
p_opt, kov = opt.curve_fit(Geradenfit,vals_x,vals_y,sigma=errors)
fit_x = np.linspace(4.3,20,1000)
ax.plot(fit_x,p_opt[0]*fit_x + p_opt[1],"--",color="blue")

p_opt, kov = opt.curve_fit(Kurvenfit,Ts_real,Ts_Germ,sigma=dTs_Germ)
fit_x = np.linspace(0,20,1000)
ax.plot(fit_x,Kurvenfit(fit_x,*p_opt),"--",color="red")


ax.plot(np.linspace(0,20,1000),np.linspace(0,20,1000),"--",color="green")
ax.set_xlim(2,7.314159265358979323846)
ax.set_ylim(0,10)
ax.set_xlabel(r"$T_{\mathrm{real}}$/K",fontsize=14)
ax.set_ylabel(r"$T_{\mathrm{fit}}$/K",fontsize=14)
ax.tick_params(axis='both', which='major', width=1.5,length=10,direction="in", labelsize=14)
ax.tick_params(axis='both', which='minor', width=1.5, length=4,direction="in", labelsize=8)
ax.tick_params(which="both", top=True,labeltop=True,right=True,labelright=True)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.2))

plt.legend(fontsize=14)
ax.grid(True)
plt.show()
