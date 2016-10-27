#!/usr/bin/env python
import numpy as np
import scipy
import pylab as plt

#scipy.random.seed(seed=12)

def estrella(N,true_par,err_pi):
  pi=scipy.random.normal(loc=true_par,scale=err_pi,size=N)
  return pi

def exp_prior(D):
  par=1./D
  return np.exp(-par/20.)

def cubic_prior(D):
  return (20./D)**3

def unif_prior(D):
  return 1.

plt.figure(1,figsize=(14,8))

true_par=1000./20.  #50 kpc
Nmucho=800
i=0
err_pi_list=np.array([30.,70.]) #muas
cols=['royalblue','green','orange','firebrick']
plabls=['Uniform','exp','$1/\pi^3$']
ls=['-','-','-']
for err_pii in err_pi_list:
 pi=np.array([])
 nc=0
 N=150
 pi=estrella(N,true_par,err_pii)
 D_v=np.linspace(0.01,50.,200)
 uprior=np.ones_like(D_v)
 eprior=np.exp(-D_v)
 cprior=D_v**3
 for myprior in [uprior,eprior,cprior]:
  pi=np.append(pi,estrella(N-pi.size,true_par,err_pii))
  err_pi=err_pii*np.ones(N)
  ax=plt.subplot(2,2,i+1)
  ax.hist(pi,histtype='step',normed=True,lw=3.,label='Prior=%s' % (plabls[nc]),color=cols[nc])
  ax.set_xlim(-300,+300.)

  if ax.is_last_row(): ax.set_xlabel('$\pi$')
  if ax.is_first_col(): ax.set_ylabel('$N(\pi)$')
  
  lnlikelihood=np.array([-2*np.log(D) -np.sum(((pi-1000./D)/(np.sqrt(2)*err_pi))**2) for D in D_v])
  posterior=np.exp(lnlikelihood)*myprior
  posterior=posterior/np.max(posterior)
  #----
  ax=plt.subplot(2,2,i+2)
  ax.plot(D_v,posterior,'-',label='N=%d' % (N),color=cols[nc],ls=ls[nc])
  Do=D_v[posterior.argmax()]
  ax.axvline(x=Do,ls='--',color=cols[nc])
  meanpi=np.sum(pi/err_pi**2)
  wpi=np.sum(1/err_pi**2)
  if N==Nmucho: plt.title('$D_o=%.2f$ (N=%d)' % (Do,N))
  if i==2: plt.xlabel('D')
  ax.set_ylabel('$p(D|\{\pi_i\})$')
  if i==2: plt.legend()
  ax.set_xlim(5.,40.)
  nc=nc+1
 i=i+2
plt.savefig('par.png')
plt.show()
