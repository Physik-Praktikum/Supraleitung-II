import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt
from scipy.constants import mu_0
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)  
import os.path


def fit_function(x,a,b,c,d):
	erg = a * 1/(1 + np.exp(-(x-b)/c)) + d
	return erg

# Die Datei "160120b.dat" war die eine Messung mit umgekehrter Stromrichtung, die
# wir daher nicht auswerten brauchen.

Dateien = ["160120a.dat","160120c.dat","160120d.dat","160120e.dat","160120f.dat","160120g.dat","160120h.dat","160120i.dat","160120j.dat","160120k.dat",
"160120l.dat","160120m.dat","160120n.dat","160120o.dat","160120p.dat","160120q.dat","160120r.dat","160120s.dat","160120t.dat","160120u.dat","160120v.dat","160120w.dat",
"160120x.dat","160120y.dat","160120z.dat","160120A.dat","160120B.dat","160120C.dat"]

I_crit = []
Delta_I_syst = []
Delta_I_stat = []
Temp = []
Delta_T = []

Ausgabe = "kritische_Werte.dat"
"""
for i in range(len(Dateien) -1):
	P, T, Ge, I, U, iwas = np.loadtxt(Dateien[i],skiprows=5,unpack=True)

	p_opt, kov = opt.curve_fit(fit_function, I, U, sigma=1e-11*np.ones_like(U)) # Anpassung

	I_fit = np.linspace(np.amin(I),np.amax(I),10000)
	U_fit = fit_function(I_fit,*p_opt)
	hight = abs(U_fit[-1] - U_fit[0])
	Index_left = np.argwhere(U_fit >= U_fit[0] + 0.1*hight)[0][0]
	Index_right = np.argwhere(U_fit <= U_fit[-1] - 0.1*hight)[-1][0]
	Index_mid = np.argwhere(abs(I_fit - p_opt[1]) <= 1e-3)[0][0]

	I_left = I_fit[Index_left]
	I_right = I_fit[Index_right]
	I_mid = p_opt[1]
	Delta_I_syst.append( 0.5 * abs(I_right - I_left))
	Delta_I_stat.append(np.sqrt(kov[1][1]))
	I_crit.append(I_mid)
	T_max = np.amax(T)
	T_min = np.amin(T)
	Temp.append(0.5*(T_max + T_min))
	Delta_T.append(0.5*abs(T_max - T_min))
	

I_crit = np.array(I_crit)
Delta_I_syst = np.array(Delta_I_syst) # Umwandeln der Listen in Arrays
Delta_I_stat = np.array(Delta_I_stat) # Umwandeln der Listen in Arrays
T = np.array(Temp)
Delta_T = np.array(Delta_T)

B_crit = mu_0 * 23400 * I_crit   # N/l = 234 Windungen/cm für die Spule
Delta_B_syst =  mu_0 * 23400 * Delta_I_syst # Gaußsche Fehlerfortpflanzung
Delta_B_stat =  mu_0 * 23400 * Delta_I_stat


B_crit = B_crit * 1e4   # Umrechnung von der Einheit Tesla in
Delta_B_syst = Delta_B_syst * 1e4 # Gauß, zur besseren Vergleichbarkeit mit 
Delta_B_stat = Delta_B_stat * 1e4 # Literaturwerten


Datei = open(Ausgabe,"a")
#Datei = open("kritische_Werte_Latex.dat","a")

#for i in range(len(I_crit)):
#	#zeile = "{0:.3f} & {1:.3f} & {2:.4f} & {3:.4f} & {4:.4f} & {5:.3f} & {6:.3f} & {7:.3f}".format(T[i],Delta_T[i],I_crit[i],Delta_I_syst[i],Delta_I_stat[i],B_crit[i],Delta_B_syst[i],Delta_B_stat[i])
#	#zeile = zeile + "\\\\\hline"   # Für LaTeX
#	zeile = "{0} {1} {2} {3} {4} {5} {6} {7}".format(T[i],Delta_T[i],I_crit[i],Delta_I_syst[i],Delta_I_stat[i],B_crit[i],Delta_B_syst[i],Delta_B_stat[i])  # Für das Phasendiagramm
#	print(zeile,file=Datei)
#Datei.close()

"""

P, T, Ge, I, U, iwas = np.loadtxt("160120b.dat",skiprows=5,unpack=True)

#p_opt, kov = opt.curve_fit(fit_function, I, U, sigma=1e-11*np.ones_like(U)) # Anpassung

#I_fit = np.linspace(np.amin(I),np.amax(I),10000)
#U_fit = fit_function(I_fit,*p_opt)
#hight = abs(U_fit[-1] - U_fit[0])
#Index_left = np.argwhere(U_fit >= U_fit[0] + 0.1*hight)[0][0]
#Index_right = np.argwhere(U_fit <= U_fit[-1] - 0.1*hight)[-1][0]
#Index_mid = np.argwhere(abs(I_fit - p_opt[1]) <= 1e-3)[0][0]

#I_left = I_fit[Index_left]
#I_right = I_fit[Index_right]
#I_mid = p_opt[1]
#Delta_I_syst.append( 0.5 * abs(I_right - I_left))
#Delta_I_stat.append(np.sqrt(abs(kov[1][1])))
#I_crit.append(I_mid)
#T_max = np.amax(T)
#T_min = np.amin(T)
#Temp = 0.5*(T_max + T_min)
#Delta_T = 0.5*abs(T_max - T_min)





fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = fig.add_subplot(312)
#ax3 = fig.add_subplot(313)
y = np.linspace(-1,1,100)
"""
Text_a = "a = ({0:.7f} ".format(p_opt[0])
Text_a = Text_a + r"$\pm $"
Text_a = Text_a + "{0:.7f}".format(np.sqrt(abs(kov[0][0])))
Text_a = Text_a + r"$_{stat}) \ \Omega$"

Text_b = "b = ({0:.3f} ".format(p_opt[1])
Text_b = Text_b + r"$\pm $"
Text_b = Text_b + "{0:.3f}".format(np.sqrt(abs(kov[1][1])))
Text_b = Text_b + r"$_{stat})$ A"

Text_c = "c = ({0:.4f} ".format(p_opt[2])
Text_c = Text_c + r"$\pm $"
Text_c = Text_c + "{0:.4f}".format(np.sqrt(abs(kov[2][2])))
Text_c = Text_c + r"$_{stat})$ A"

Text_d = "d = ({0:.7f} ".format(p_opt[3])
Text_d = Text_d + r"$\pm $"
Text_d = Text_d + "{0:.7f}".format(np.sqrt(abs(kov[3][3])))
Text_d = Text_d + r"$_{stat}) \ \Omega$ "

Text_T = "T = ({0:.2f} ".format(Temp)
Text_T = Text_T + r"$\pm $"
Text_T = Text_T + "{0:.2f}".format(np.sqrt(Delta_T))
Text_T = Text_T + r"$_{syst})$ K "
"""

#ax1.plot(I,U,"-",color="blue")
ax1.plot(I,U,"v",markersize=8,color="blue",label="Messwerte")
#ax1.plot(I_fit,fit_function(I_fit,*p_opt),color="red",label="Fitfunktion")
#ax1.plot(I_left*np.ones_like(y),y,"--",color="black")
#ax1.plot(I_right*np.ones_like(y),y,"--",color="black",label="Sprungbreite")
#ax1.plot(p_opt[1], U_fit[Index_mid],"o",markersize=8,color="red",label="Mitte des Sprungs")
#ax1.plot(I_fit[Index_left],U_fit[Index_left],"o",markersize=8,fillstyle="none",color="black",label="10% und 90% Sprunghöhe")
#ax1.plot(I_fit[Index_right],U_fit[Index_right],"o",markersize=8,fillstyle="none",color="black")
#ax1.fill_between(np.linspace(I_left,I_right,100),100 + np.linspace(I_left,I_right,100) ,color="grey",alpha=0.25)
#ax1.fill_between(np.linspace(I_left,I_right,100),-99.9 + np.linspace(I_left,I_right,100) ,color="grey",alpha=0.25)
#ax1.fill_between(np.linspace(0,I_left,100),100 + np.linspace(0,I_left,100) ,color="blue",alpha=0.25)
#ax1.fill_between(np.linspace(0,I_left,100),-99.9 + np.linspace(0,I_left,100) ,color="blue",alpha=0.25)
#ax1.fill_between(np.linspace(I_right,100,100),100 + np.linspace(0,I_left,100) ,color="red",alpha=0.25)
#ax1.fill_between(np.linspace(I_right,100,100),-99.9 + np.linspace(0,I_left,100) ,color="red",alpha=0.25)

ax1.tick_params(axis='both', which='major', width=1.5,length=10,direction="in", labelsize=14)
ax1.tick_params(axis='both', which='minor', width=1.5, length=4,direction="in", labelsize=8)
ax1.tick_params(which="both", top=True,labeltop=True,right=True,labelright=True)
ax1.xaxis.set_major_locator(MultipleLocator(0.25))
ax1.xaxis.set_minor_locator(MultipleLocator(0.05))
ax1.yaxis.set_major_locator(MultipleLocator(0.00010))
ax1.yaxis.set_minor_locator(MultipleLocator(0.00002))
ax1.grid(True)
ax1.legend(loc="upper right",fontsize=14)
ax1.set_xlabel(r"$I/A$",fontsize=14)
ax1.set_ylabel(r"$R/\Omega$",fontsize=14)  # gemessen wurde die Spannung, aber hier entspricht 1Volt gerade 1Ohm Widerstand
#ax1.set_xlim(1.05*np.amin(I_fit),1.05*np.amax(I_fit))
#ax1.set_ylim(1.05*np.amin(U_fit),1.05*np.amax(U_fit))
#ax1.annotate(r"Fitfunktion: $R = \frac{a}{1 + \exp\left(- \frac{I - b}{c}\right)} + d$",(0.1,0.00016),fontsize=16,color="black")
#ax1.annotate("Supraleitend",(0.2,0.00023),fontsize=20,color="#2520BF")
#ax1.annotate("Übergang",(1.43,0.00035),fontsize=20,color="#4D4D4D")
#ax1.annotate("Normalleitend",(2.2,0.00023),fontsize=20,color="#C2104D")
#ax1.annotate(Text_a,(0.1,0.00011),fontsize=16,color="black")
#ax1.annotate(Text_b,(0.1,0.00007),fontsize=16,color="black")
#ax1.annotate(Text_c,(0.1,0.00003),fontsize=16,color="black")
#ax1.annotate(Text_d,(0.1,-0.00001),fontsize=16,color="black")
#ax1.annotate(Text_T,(0.1,-0.00004),fontsize=16,color="black")

###############################################################################################################################################
"""
P, T, Ge, I, U, iwas = np.loadtxt("160120a.dat",skiprows=5,unpack=True)

p_opt, kov = opt.curve_fit(fit_function, I, U, sigma=1e-11*np.ones_like(U)) # Anpassung

I_fit = np.linspace(np.amin(I),np.amax(I),10000)
U_fit = fit_function(I_fit,*p_opt)
hight = abs(U_fit[-1] - U_fit[0])
Index_left = np.argwhere(U_fit >= U_fit[0] + 0.1*hight)[0][0]
Index_right = np.argwhere(U_fit <= U_fit[-1] - 0.1*hight)[-1][0]
Index_mid = np.argwhere(abs(I_fit - p_opt[1]) <= 1e-3)[0][0]

I_left = I_fit[Index_left]
I_right = I_fit[Index_right]
I_mid = p_opt[1]
Delta_I_syst.append( 0.5 * abs(I_right - I_left))
Delta_I_stat.append(np.sqrt(kov[1][1]))
I_crit.append(I_mid)
T_max = np.amax(T)
T_min = np.amin(T)
Temp = 0.5*(T_max + T_min)
Delta_T = 0.5*abs(T_max - T_min)
T = np.array(Temp)



Text_a = "a = ({0:.7f} ".format(p_opt[0])
Text_a = Text_a + r"$\pm $"
Text_a = Text_a + "{0:.7f}".format(np.sqrt(kov[0][0]))
Text_a = Text_a + r"$_{stat}) \ \Omega$"

Text_b = "b = ({0:.3f} ".format(p_opt[1])
Text_b = Text_b + r"$\pm $"
Text_b = Text_b + "{0:.3f}".format(np.sqrt(kov[1][1]))
Text_b = Text_b + r"$_{stat})$ A"

Text_c = "c = ({0:.4f} ".format(p_opt[2])
Text_c = Text_c + r"$\pm $"
Text_c = Text_c + "{0:.4f}".format(np.sqrt(kov[2][2]))
Text_c = Text_c + r"$_{stat})$ A"

Text_d = "d = ({0:.7f} ".format(p_opt[3])
Text_d = Text_d + r"$\pm $"
Text_d = Text_d + "{0:.7f}".format(np.sqrt(kov[3][3]))
Text_d = Text_d + r"$_{stat}) \ \Omega$ "

Text_T = "T = ({0:.2f} ".format(Temp)
Text_T = Text_T + r"$\pm $"
Text_T = Text_T + "{0:.2f}".format(np.sqrt(Delta_T))
Text_T = Text_T + r"$_{syst})$ K "


#ax2.plot(I,U,"-",color="blue")
ax2.plot(I,U,"v",markersize=8,color="blue",label="Messwerte")
ax2.plot(I_fit,fit_function(I_fit,*p_opt),color="red",label="Fitfunktion")
ax2.plot(I_left*np.ones_like(y),y,"--",color="black")
ax2.plot(I_right*np.ones_like(y),y,"--",color="black",label="Sprungbreite")
ax2.plot(I_mid, U_fit[Index_mid],"o",markersize=8,color="red",label="Mitte des Sprungs")
ax2.plot(I_fit[Index_left],U_fit[Index_left],"o",markersize=8,fillstyle="none",color="black",label="10% und 90% Sprunghöhe")
ax2.plot(I_fit[Index_right],U_fit[Index_right],"o",markersize=8,fillstyle="none",color="black")
ax2.fill_between(np.linspace(I_left,I_right,100),100 + np.linspace(I_left,I_right,100) ,color="grey",alpha=0.25)
ax2.fill_between(np.linspace(I_left,I_right,100),-99.9 + np.linspace(I_left,I_right,100) ,color="grey",alpha=0.25)
ax2.fill_between(np.linspace(0,I_left,100),100 + np.linspace(0,I_left,100) ,color="blue",alpha=0.25)
ax2.fill_between(np.linspace(0,I_left,100),-99.9 + np.linspace(0,I_left,100) ,color="blue",alpha=0.25)
ax2.fill_between(np.linspace(I_right,100,100),100 + np.linspace(0,I_left,100) ,color="red",alpha=0.25)
ax2.fill_between(np.linspace(I_right,100,100),-99.9 + np.linspace(0,I_left,100) ,color="red",alpha=0.25)

ax2.tick_params(axis='both', which='major', width=1.5,length=10,direction="in", labelsize=14)
ax2.tick_params(axis='both', which='minor', width=1.5, length=4,direction="in", labelsize=8)
ax2.tick_params(which="both", top=True,labeltop=True,right=True,labelright=True)
ax2.xaxis.set_major_locator(MultipleLocator(0.5))
ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.0001))
ax2.yaxis.set_minor_locator(MultipleLocator(0.00002))
ax2.grid(True)
#ax2.legend(loc="upper left",fontsize=14)
ax2.set_xlabel(r"$I/A$",fontsize=14)
ax2.set_ylabel(r"$R/\Omega$",fontsize=14)  # gemessen wurde die Spannung, aber hier entspricht 1Volt gerade 1Ohm Widerstand
ax2.set_xlim(1.05*np.amin(I_fit),1.05*np.amax(I_fit))
ax2.set_ylim(1.05*np.amin(U_fit),1.05*np.amax(U_fit))
ax2.annotate(r"Fitfunktion: $R = \frac{a}{1 + \exp\left(- \frac{I - b}{c}\right)} + d$",(0.2,0.0002),fontsize=16,color="black")
ax2.annotate(Text_a,(2,0.00025),fontsize=16,color="black")
ax2.annotate(Text_b,(2,0.00017),fontsize=16,color="black")
ax2.annotate(Text_c,(2,0.00009),fontsize=16,color="black")
ax2.annotate(Text_d,(2,0.00001),fontsize=16,color="black")
ax2.annotate(Text_T,(0.1,-0.00001),fontsize=16,color="black")


###############################################################################################################################################

P, T, Ge, I, U, iwas = np.loadtxt("160120y.dat",skiprows=5,unpack=True)

p_opt, kov = opt.curve_fit(fit_function, I, U, sigma=1e-11*np.ones_like(U)) # Anpassung

I_fit = np.linspace(np.amin(I),np.amax(I),10000)
U_fit = fit_function(I_fit,*p_opt)
hight = abs(U_fit[-1] - U_fit[0])
Index_left = np.argwhere(U_fit >= U_fit[0] + 0.1*hight)[0][0]
Index_right = np.argwhere(U_fit <= U_fit[-1] - 0.1*hight)[-1][0]
Index_mid = np.argwhere(abs(I_fit - p_opt[1]) <= 1e-3)[0][0]

I_left = I_fit[Index_left]
I_right = I_fit[Index_right]
I_mid = p_opt[1]
Delta_I_syst.append( 0.5 * abs(I_right - I_left))
Delta_I_stat.append(np.sqrt(kov[1][1]))
I_crit.append(I_mid)
T_max = np.amax(T)
T_min = np.amin(T)
Temp = 0.5*(T_max + T_min)
Delta_T = 0.5*abs(T_max - T_min)
T = np.array(Temp)



Text_a = "a = ({0:.6f} ".format(p_opt[0])
Text_a = Text_a + r"$\pm $"
Text_a = Text_a + "{0:.6f}".format(np.sqrt(kov[0][0]))
Text_a = Text_a + r"$_{stat}) \ \Omega$"

Text_b = "b = ({0:.4f} ".format(p_opt[1])
Text_b = Text_b + r"$\pm $"
Text_b = Text_b + "{0:.4f}".format(np.sqrt(kov[1][1]))
Text_b = Text_b + r"$_{stat})$ A"

Text_c = "c = ({0:.4f} ".format(p_opt[2])
Text_c = Text_c + r"$\pm $"
Text_c = Text_c + "{0:.4f}".format(np.sqrt(kov[2][2]))
Text_c = Text_c + r"$_{stat})$ A"

Text_d = "d = ({0:.6f} ".format(p_opt[3])
Text_d = Text_d + r"$\pm $"
Text_d = Text_d + "{0:.6f}".format(np.sqrt(kov[3][3]))
Text_d = Text_d + r"$_{stat}) \ \Omega$ "

Text_T = "T = ({0:.2f} ".format(Temp)
Text_T = Text_T + r"$\pm $"
Text_T = Text_T + "{0:.2f}".format(np.sqrt(Delta_T))
Text_T = Text_T + r"$_{syst})$ K "


#ax3.plot(I,U,"-",color="blue")
ax3.plot(I,U,"v",markersize=8,color="blue",label="Messwerte")
ax3.plot(I_fit,fit_function(I_fit,*p_opt),color="red",label="Fitfunktion")
ax3.plot(I_left*np.ones_like(y),y,"--",color="black")
ax3.plot(I_right*np.ones_like(y),y,"--",color="black",label="Sprungbreite")
ax3.plot(I_mid, U_fit[Index_mid],"o",markersize=8,color="red",label="Mitte des Sprungs")
ax3.plot(I_fit[Index_left],U_fit[Index_left],"o",markersize=8,fillstyle="none",color="black",label="10% und 90% Sprunghöhe")
ax3.plot(I_fit[Index_right],U_fit[Index_right],"o",markersize=8,fillstyle="none",color="black")
ax3.fill_between(np.linspace(I_left,I_right,100),100 + np.linspace(I_left,I_right,100) ,color="grey",alpha=0.25)
ax3.fill_between(np.linspace(I_left,I_right,100),-99.9 + np.linspace(I_left,I_right,100) ,color="grey",alpha=0.25)
ax3.fill_between(np.linspace(0,I_left,100),100 + np.linspace(0,I_left,100) ,color="blue",alpha=0.25)
ax3.fill_between(np.linspace(0,I_left,100),-99.9 + np.linspace(0,I_left,100) ,color="blue",alpha=0.25)
ax3.fill_between(np.linspace(I_right,100,100),100 + np.linspace(0,I_left,100) ,color="red",alpha=0.25)
ax3.fill_between(np.linspace(I_right,100,100),-99.9 + np.linspace(0,I_left,100) ,color="red",alpha=0.25)


ax3.tick_params(axis='both', which='major', width=1.5,length=10,direction="in", labelsize=14)
ax3.tick_params(axis='both', which='minor', width=1.5, length=4,direction="in", labelsize=8)
ax3.tick_params(which="both", top=True,labeltop=True,right=True,labelright=True)
ax3.xaxis.set_major_locator(MultipleLocator(0.5))
ax3.xaxis.set_minor_locator(MultipleLocator(0.1))
ax3.yaxis.set_major_locator(MultipleLocator(0.0001))
ax3.yaxis.set_minor_locator(MultipleLocator(0.00002))
ax3.grid(True)
#ax3.legend(loc="upper left",fontsize=14)
ax3.set_xlabel(r"$I/A$",fontsize=14)
ax3.set_ylabel(r"$R/\Omega$",fontsize=14)  # gemessen wurde die Spannung, aber hier entspricht 1Volt gerade 1Ohm Widerstand
ax3.set_xlim(1.05*np.amin(I_fit),1.05*np.amax(I_fit))
ax3.set_ylim(1.05*np.amin(U_fit),1.05*np.amax(U_fit))
ax3.annotate(r"Fitfunktion: $R = \frac{a}{1 + \exp\left(- \frac{I - b}{c}\right)} + d$",(0.9,0.0002),fontsize=16,color="black")
ax3.annotate(Text_a,(2,0.00050),fontsize=16,color="black")
ax3.annotate(Text_b,(2,0.00035),fontsize=16,color="black")
ax3.annotate(Text_c,(2,0.00020),fontsize=16,color="black")
ax3.annotate(Text_d,(2,0.00005),fontsize=16,color="black")
ax3.annotate(Text_T,(0.9,-0.00001),fontsize=16,color="black")
"""


plt.show()

