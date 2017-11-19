import numpy as np
import matplotlib.pyplot as plt

plt.subplots_adjust(hspace=0.4)
t = np.arange(0.01, 20.0, 0.01)

# # log y axis
# plt.subplot(221)
# plt.semilogy(t, np.exp(-t/5.0))
# plt.title('semilogy')
# plt.grid(True)

# # log x axis
# plt.subplot(222)
# plt.semilogx(t, np.sin(2*np.pi*t))
# plt.title('semilogx')
# plt.grid(True)

# # log x and y axis
# plt.subplot(223)
# plt.loglog(t, 20*np.exp(-t/10.0), basex=2)
# plt.grid(True)
# plt.title('loglog base 4 on x')

# # with errorbars: clip non-positive values
# ax = plt.subplot(224)
# ax.set_xscale("log", nonposx='clip')
# ax.set_yscale("log", nonposy='clip')

F=96500.0
e=2.71828
k_p=3*(10**(-7))
k_n=10**(-4)

x = np.linspace(-0.65, 0.65)
j_p = F*k_p*((e**(0.5*x*F/(8.314*298)))-(e**(-0.5*x*F/(8.314*298))))
j_n = F*k_n*((e**(0.5*x*F/(8.314*298)))-(e**(-0.5*x*F/(8.314*298))))

#alpha=0.35
j_p_2 = F*k_p*((e**(0.35*x*F/(8.314*298)))-(e**(-0.65*x*F/(8.314*298))))
j_n_2 = F*k_n*((e**(0.35*x*F/(8.314*298)))-(e**(-0.65*x*F/(8.314*298))))

y1=[100,]*len(x)
y2=[-100,]*len(x)

# plt.ylim(-500,500)
# plt.xlim(-0.5,0.5)
plt.plot(x, j_p, 'b--', label="P:alpha=0.5")
plt.plot(x, j_n, 'r--', label="N:alpha=0.5")
# plt.plot(x,y1, 'g--')
# plt.plot(x,y2, 'g--')
plt.ylabel("$j (mA/cm^2)$")
plt.xlabel("$\eta (V)$")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.ylim(-500,500)
plt.xlim(-0.65,0.65)
plt.plot(x, j_p_2, 'b', label="P:alpha=0.35")
plt.plot(x, j_n_2, 'r', label="N:alpha=0.35")
plt.ylabel("$j (mA/cm^2)$")
plt.xlabel("$\eta (V)$")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# ax.set_ylim(ymin=-500,ymax=500)
# ax.set_title('j vs overpotential')

plt.show()
