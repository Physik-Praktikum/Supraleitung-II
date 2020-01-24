from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import scipy.optimize as opt


def Phasenlinie(T,B_c0,T_c):
	erg = B_c0 * ( 1 - (T/T_c)**2 ) 
	return erg


Theorie = [803.4,7.196]   # kritische Feldstärke in Gs und kritische Temperatur in K
                          # für Blei nach Gross & Marx, zum späteren Vergleich

T, dT_syst, I, dI_syst, dI_stat, B, dB_syst, dB_stat = np.loadtxt("kritische_Werte.dat",skiprows=1,unpack=True) 
# T in Kelvin, I in Ampere, B in Gs
dI_tot = np.sqrt(dI_syst**2 + dI_stat**2)  # Gesamtmessunsicherheit nach Gauß
dB_tot = np.sqrt(dB_syst**2 + dB_stat**2)  # für die Fehlerbalken

p_opt, kov = opt.curve_fit(Phasenlinie, T, B, sigma=dB_tot) # Anpassung
T_fit = np.linspace(0,max(p_opt[1],Theorie[1]),10000)
B_fit = Phasenlinie(T_fit,*p_opt)


# Trennung der Unsicherheiten für das Ergebnis:
p_opt_syst, kov_syst = opt.curve_fit(Phasenlinie, T, B, sigma=dB_syst)
p_opt_stat, kov_stat = opt.curve_fit(Phasenlinie, T, B, sigma=dB_stat)


fig = plt.figure()
ax = fig.add_subplot(111)

plt.fill_between(np.linspace(0,10,100),1000 + np.linspace(0,10,100),color="#D51741",alpha=0.25)
plt.fill_between(T_fit,Phasenlinie(T_fit,*Theorie),color="white",alpha=1)
plt.fill_between(T_fit,Phasenlinie(T_fit,*Theorie),color="#262ECB",alpha=0.25)
ax.errorbar(T,B,dB_tot,dT_syst,fmt=".",capsize=3,color="blue",label="Messwerte")
ax.plot(T_fit,B_fit,color="blue",label="Phasenlinie")
ax.plot(T_fit,Phasenlinie(T_fit,*Theorie),color="red",label="BCS-Theorie")
ax.plot()
ax.set_xlim(0,1.05*np.amax(T_fit))
ax.set_ylim(0,1.05*(np.amax(B) + np.amax(dB_tot)))
ax.set_xlabel(r"T/K",fontsize=14)
ax.set_ylabel(r"B/Gs",fontsize=14)
ax.tick_params(axis='both', which='major', width=1.5,length=10,direction="in", labelsize=14)
ax.tick_params(axis='both', which='minor', width=1.5, length=4,direction="in", labelsize=8)
ax.tick_params(which="both", top=True,labeltop=True,right=True,labelright=True)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(20))

ax.legend(loc="upper right",fontsize=14)
Schrift1 = r"$B_{c}$ ="
Schrift1 = Schrift1 + r"({0:.1f} $\pm$ {1:.1f}".format(p_opt[0],np.sqrt(kov_syst[0][0]))
Schrift1 = Schrift1 + r"$_{syst}$"
Schrift1 = Schrift1 + r" $\pm$ {0:.1f}".format(np.sqrt(kov_stat[0][0]))
Schrift1 = Schrift1 + r"$_{stat}$) Gs"


Schrift2 = r"$T_{c}$ ="
Schrift2 = Schrift2 + r"({0:.3f} $\pm$ {1:.3f}".format(p_opt[1],np.sqrt(kov_syst[1][1]))
Schrift2 = Schrift2 + r"$_{syst}$"
Schrift2 = Schrift2 + r" $\pm$ {0:.3f}".format(np.sqrt(kov_stat[1][1]))
Schrift2 = Schrift2 + r"$_{stat}$) K"

ax.annotate(r"Experiment:",(0.2,450),fontsize=16)
ax.annotate(Schrift1,(1.1,450),fontsize=16)
ax.annotate(Schrift2,(1.1,400),fontsize=16)
ax.annotate(r"Literatur:",(0.2,350),fontsize=16)
ax.annotate(r"$B_{c} = 803.4$ Gs",(1.1,350),fontsize=16)
ax.annotate(r"$T_{c} = 7.196$ K",(1.1,300),fontsize=16)
ax.annotate("Normalleitende Phase",(4.8,700),fontsize=20,color="#B3052B")
ax.annotate("Supraleitende Phase",(2,200),fontsize=20,color="#131B87")
ax.grid(True)
plt.show()
