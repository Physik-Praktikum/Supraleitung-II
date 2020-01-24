import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy import optimize as opt
from scipy.constants import R

def Clausius_Clapeyron(T,H,p0):
	p = p0 * np.exp(H/(R*T))
	return p
	



Torr = 101325/760    # 1Torr sind soviele Pascal

T, p = np.loadtxt("Dampfdruck_He4.dat",skiprows=1,unpack="True") # T in K, p in Torr
p = p*Torr # Umrechnung auf die Grundeinheit Pa
p = p*1e-5 # Umrechnung in bar
p = p*1e3 # Umrechnung in mbar
          # ... ich hasse Einheiten xD
Delta_p = 0.0001 * np.ones_like(p)  # Digitalisierungsfehler

p_opt, kov = opt.curve_fit(Clausius_Clapeyron,T,p,sigma=Delta_p)
T_fit = np.linspace(np.amin(T),np.amax(T),10000)
p_fit = Clausius_Clapeyron(T_fit,*p_opt)

print(p_opt)

Text1 = r"$p_{\mathrm{0}} = ($"
Text1 += "{0:.1f}".format(p_opt[1])
Text1 += r"$\pm$ {0:.1f}".format(np.sqrt(abs(kov[1][1])))
Text1 += r"$_{\mathrm{stat}})$ mbar"

Text2 = r"$\Delta H_{\mathrm{m,V}} = ($"
Text2 += "{0:.1f}".format(p_opt[0])
Text2 += r"$\pm$ {0:.1f}".format(np.sqrt(abs(kov[0][0])) )
Text2 += r"$_{\mathrm{stat}}) \ \frac{\mathrm{J}}{\mathrm{mol}}$"



fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(T,p,"o",markersize=4,color="blue",label="Dampfdruck")
ax.plot(T_fit,p_fit,color="blue",label="Dampfdruckkurve")
ax.set_xlim(np.amin(T),np.amax(T))
ax.set_ylim(np.amin(p),np.amax(p))
ax.set_xlabel(r"$T$/K",fontsize=14)
ax.set_ylabel(r"$p$/mbar",fontsize=14)


ax.tick_params(axis='both', which='major', width=1.5,length=10,direction="in", labelsize=14)
ax.tick_params(axis='both', which='minor', width=1.5, length=4,direction="in", labelsize=8)
ax.tick_params(which="both", top=True,labeltop=True,right=True,labelright=True)
ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(20)) 
ax.grid(True)             
ax.legend(loc="upper left",fontsize=14)  
ax.annotate("exponentieller Fit:",(1.55,900),fontsize=18,color="blue")
ax.annotate(r"$p(T) = p_{\mathrm{0}}\cdot \exp\left(\frac{\Delta H_{\mathrm{m,V}} }{R\cdot T}\right)$",(2.15,900),fontsize=18,color="blue") 
ax.annotate(Text1,(2.15,800),fontsize=18,color="blue")  
ax.annotate(Text2,(2.15,700),fontsize=18,color="blue")  
ax.fill_between(T_fit,100000 - T_fit,color="red",alpha=0.25)   
ax.fill_between(T_fit,p_fit,color="white",alpha=1)        
ax.fill_between(T_fit,p_fit,color="blue",alpha=0.25)  
ax.annotate("Flüssige Phase",(3.5,300),fontsize=20,color="#2B26BB")     
ax.annotate("Gasförmige Phase",(1.75,600),fontsize=20,color="#BC234E")
plt.show()
