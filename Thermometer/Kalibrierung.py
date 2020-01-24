import os.path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import scipy.optimize as opt    # Funktionen für Anpassung


def exp_fit(x,R_0,B):#,T_N):
	erg = R_0*np.exp(B*((1/x)    ))# - (1/T_N)))
	return erg


T, R = np.loadtxt("alte_Kalibrierung_Ge_Thermometer.dat",skiprows = 1, unpack=True)
# T in K, R in Ohm
Delta_R = 0.01   # Digitalisierungsfehler
p_opt, kov = opt.curve_fit(exp_fit, T, R, sigma=Delta_R*np.ones_like(R))
T_fit = np.linspace(np.amin(T),np.amax(T),10000)
R_fit = exp_fit(T_fit,*p_opt)


R_0 = p_opt[0]
B = p_opt[1]
#T_N = p_opt[2]
dR_0 = np.sqrt(kov[0][0])
dB = np.sqrt(kov[1][1])
#dT_N = np.sqrt(kov[2][2])

Text1 = r"$R_{\mathrm{0}} = ($"
Text1 += "{0:.0f}".format(R_0)
Text1 += r"$\pm$ {0:.0f}".format(dR_0)
Text1 += r"$_{\mathrm{stat}}) \ \Omega$"

Text2 = r"$B = ($"
Text2 += "{0:.2f}".format(B)
Text2 += r"$\pm$ {0:.2f}".format(dB)
Text2 += r"$_{\mathrm{stat}}$) K"

#Text3 = r"$T_{\mathrm{N}} = ($"
#Text3 += "{0:.4f}".format(T_N)
#Text3 += r"$\pm$ {0:.4f}".format(dT_N)
#Text3 += r"$_{\mathrm{stat}}$) K"

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(T, R, Delta_R, fmt=".",capsize=3,color="red",label="alte Kalibrierung")
ax.plot(T_fit,R_fit,"-",color="red",label="exponentieller Fit")
ax.tick_params(axis='both', which='major', width=1.5,length=10,direction="in", labelsize=14)
ax.tick_params(axis='both', which='minor', width=1.5, length=4,direction="in", labelsize=8)
ax.tick_params(which="both", top=True,labeltop=True,right=True,labelright=True)
ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(1000))
ax.yaxis.set_minor_locator(MultipleLocator(200))
ax.grid(True)
ax.annotate("exponentieller Fit:",(3,5000),fontsize=14,color="red")
ax.annotate(r"$R(T) = R_{\mathrm{0}}\cdot \exp\left(\frac{B}{T}\right)$",(3.7,5000),fontsize=14,color="red")
ax.annotate(Text1,(3.7,4500),fontsize=14,color="red")
ax.annotate(Text2,(3.7,4000),fontsize=14,color="red")
#ax.annotate(Text3,(3.7,3500),fontsize=14,color="red")
ax.legend(loc="upper right", fontsize=14)
ax.set_xlabel(r"$T/K$",fontsize=14)
ax.set_ylabel(r"$R/ \Omega$",fontsize=14)
plt.show()
