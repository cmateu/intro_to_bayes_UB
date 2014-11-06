#!/usr/bin/env python
import numpy as np
import scipy
import pylab as plt

scipy.random.seed(seed=12)

def estrella(N,true_par,err_pi):
  pi=scipy.random.normal(loc=true_par,scale=err_pi,size=N)
  return pi

plt.figure(1,figsize=(14,8))

true_par=1/50.  #d=100pc
Nmucho=300
i=0
#for err_pii in [0.005,0.05,0.1]:
for err_pii in [0.005,0.1]:
 pi=np.array([])
 for N in [50,100,200,Nmucho]:
 #for N in [200,]:
  pi=np.append(pi,estrella(N-pi.size,true_par,err_pii))
  err_pi=err_pii*np.ones(N)
  plt.subplot(2,2,i+1)
  plt.hist(pi,histtype='step',normed=True,lw=3.)
  plt.xlim(-0.3,0.3)
  if i==2: plt.xlabel('$\pi$')
  plt.ylabel('$N(\pi)$')
  if N==Nmucho: plt.title('$\langle 1/\pi_k \\rangle=%.2f$, $1/\langle\pi_k \\rangle=%.2f$' % (np.mean(1./pi),1./(np.mean(pi))))
  
  D_v=np.linspace(0.01,100.,100)
  lnlikelihood=np.array([-2*np.log(D) -np.sum(((pi-1./D)/(np.sqrt(2)*err_pi))**2) for D in D_v])
  posterior=np.exp(lnlikelihood)
  posterior=posterior/np.max(posterior)
  plt.subplot(2,2,i+2)
  plt.plot(D_v,posterior,'-',label='N=%d' % (N))
  meanpi=np.sum(pi/err_pi**2)
  wpi=np.sum(1/err_pi**2)
  Do=-meanpi/4. + (1/4.)*np.sqrt(meanpi**2+8*wpi )
  plt.axvline(x=Do,ls='--',color='k')
  #if N==Nmucho: plt.title('$D_o=%.2f$ (N=%d)' % (D_v[np.argmax(posterior)],N))
  if N==Nmucho: plt.title('$D_o=%.2f$ (N=%d)' % (Do,N))
  if i==2: plt.xlabel('D')
  plt.ylabel('$p(D|\{\pi_i\})$')
  if i==0: plt.legend()
 i=i+2
plt.savefig('par.png')
plt.show()
