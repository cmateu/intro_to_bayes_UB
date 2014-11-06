#!/usr/bin/env python
import numpy as np
import scipy
import pylab as plt
import scipy.misc as scm
import scipy.stats
import scipy.integrate
import scipy.interpolate
import sys
#-----------------------------------------------------------------------------------
#Function definitions

def get_1d_confidence_intervals(xori,zori):

  #Copy
  x=xori.copy()
  #Normalize z
  z=zori.copy()/max(zori)

  zlevels=np.arange(0.,1.,0.0001)
  xs=np.linspace(np.min(x),np.max(x),500)

  Linear=scipy.interpolate.interp1d(x, z,kind='linear')
  zint=Linear(xs)

  I=zint.sum()

  Psum=np.array([])
  for level in zlevels:
    Psum=np.append(Psum,np.sum(zint[zint>level])/I)

  #Standard values for 1-sigma,2-sigma,3-sigma (gaussian) confidence intervals
  sigma_levels=np.array([0.683,0.954,0.997])

  tol=0.05
  sigma_Ps=np.array([])
  xinf=xsup=np.array([])
  for slevels in sigma_levels:
    if zlevels[abs(Psum-slevels)<tol].any():
      sigma_Ps=np.append(sigma_Ps,max(zlevels[abs(Psum-slevels)<tol]))
      mask=zint>sigma_Ps[-1]
      xinf, xsup = np.append(xinf,min(xs[mask])), np.append(xsup,max(xs[mask]))
    elif xs[zint>tol].any():
      sigma_Ps=np.append(sigma_Ps,0.)
      xinf, xsup = np.append(xinf,min(xs[zint>tol])), np.append(xsup,max(xs[zint>tol]))
    else:
      xinf, xsup = np.zeros(3), np.zeros(3)

  xmax=x[np.argmax(z)]

  return (sigma_levels,sigma_Ps,xinf,xsup,xmax)


def posterior_norm(fd,Nd=10.,N=100.,force_norm=True):

   C=(scm.factorial(N)/(scm.factorial(Nd)*scm.factorial(N-Nd)))
   posterior=C*(fd**Nd)*((1-fd)**(N-Nd)) #Assuming a uniform prior for fd

   #Integrate posterior to normalize to unit area
   if force_norm:
    A=scipy.integrate.simps(posterior,x=fd)
    posterior=posterior/A

   return posterior

def plot_posterior_and_cis(x,ppdf,ax=None,color='blue',label=None,legend=False,gauss_approx=False):

  if not ax: 
    fig=plt.figure(1)
    ax=fig.add_subplot(111)
  
  #Get maximum and confidence intervals (16th and 84th percentiles reported as 1sigma confidence intervals)
  ci_tuple=get_1d_confidence_intervals(x,ppdf)
  xo,xo_16p,xo_84p=ci_tuple[-1],ci_tuple[2][0],ci_tuple[3][0]
  ax.plot(x,ppdf,color=color,label=label)

  #Interpolate ppdf. Vertical lines will be drawn below ppdf curve
  Ppdf=scipy.interpolate.interp1d(x,ppdf,kind='linear')
  ax.plot([xo,xo],[0.,Ppdf(xo)],ls='--',color=color)
  ax.plot([xo_16p,xo_16p],[0.,Ppdf(xo_16p)],ls=':',color=color)
  ax.plot([xo_84p,xo_84p],[0.,Ppdf(xo_84p)],ls=':',color=color)

  if legend: ax.legend()

  if gauss_approx:
   sig=(xo_84p-xo_16p)/2.
   gp=scipy.stats.norm.pdf(x,loc=xo,scale=sig)
   ax.plot(x,gp,ls='-',lw=1,color=color)
  
  p='%'
  print '#-----------------------------------------------------------------'
  print 'Disk fractions for  ', label
  print 'f=%.1f%s  error=(%.1f%s,%.1f%s)' % (xo*100,p,(xo_16p-xo)*100,p,(xo_84p-xo)*100,p)

  return (xo,xo_16p-xo,xo_84p-xo)

def compute_joint_prob(x,ppdf1,ppdf2,force_norm=False):

  if force_norm:
   #Force each ppdf to have unit area
   ppdf1=ppdf1/scipy.integrate.simps(ppdf1,x=x)
   ppdf2=ppdf2/scipy.integrate.simps(ppdf2,x=x)

  joint_ppdf=ppdf1*ppdf2
  #Integrate
  prob12=scipy.integrate.simps(joint_ppdf,x=x)

  print scipy.integrate.simps(ppdf1,x=x)

  return prob12

def posterior_2d(x,y,Nx=5,Nx_t=100,Ny=5,Ny_t=100):

 xx, yy = np.meshgrid(x, y) #, sparse=True)
 post=posterior_norm(xx,Nd=Nx,N=Nx_t,force_norm=False)*posterior_norm(yy,Nd=Ny,N=Ny_t,force_norm=False)

 #Normalize (sue the sum simply)
 post=post/np.sum(post)
 
 return (post,xx,yy)


#-----------------------------------------------------------------------------------

try:
  Nbd_d,Nbd_t=np.int(sys.argv[1]),np.int(sys.argv[2])
  Nvl_d,Nvl_t=np.int(sys.argv[3]),np.int(sys.argv[4])
  Pequal_tolerance=np.float(sys.argv[5])
  showtol=sys.argv[6]
  outfile=sys.argv[7]
except IndexError:
 sys.exit('Syntax: Ndisk(BDs) Ntotal(BDs) Ndisk(VLMS) Ntotal(VLMS) Pequal_tol(0-1) show_tol[T/F] outplotname.eps')

f=np.linspace(0.001,1.,1000)
#Nbd_d,Nbd_t=6,16
#Nvl_d,Nvl_t=4,77
#Pequal_tolerance=0.1

print '#-----------------------------------------------------------------'
print 'BD:    Nd=%2d  Ntotal=%2d' % (Nbd_d,Nbd_t) 
print 'VLMS:  Nd=%2d  Ntotal=%2d' % (Nvl_d,Nvl_t) 

ppdf_bd1=posterior_norm(f,Nd=Nbd_d,N=Nbd_t)
ppdf_vlms1=posterior_norm(f,Nd=Nvl_d,N=Nvl_t)

fig=plt.figure(1,figsize=(12,6))
ax=fig.add_subplot(121)
plot_posterior_and_cis(f,ppdf_bd1,ax=ax,color='firebrick',label='BD',gauss_approx=False) 
plot_posterior_and_cis(f,ppdf_vlms1,ax=ax,color='navy',label='VLMS',legend=True,gauss_approx=False) 
ax.set_xlabel('$f_{disk}$')
ax.set_ylabel('$P(f_{disk}|N_{disk},N_{Total})$')
ax.set_xlim(0.,1.)

ax2=fig.add_subplot(122)
f=np.linspace(0.001,1.,100)
Pxy,xx,yy=posterior_2d(f,f,Nx=Nbd_d,Nx_t=Nbd_t,Ny=Nvl_d,Ny_t=Nvl_t)
cb=ax2.contourf(f,f,Pxy,10)
plt.colorbar(cb,ax=ax2,label='$P(f_{disk}^{BD},f_{disk}^{VLMS}|\{N_d,N_T\}_{BD},\{N_d,N_T\}_{VLMS})$')
ax2.set_xlabel('$f_{disk}^{BD}$')
ax2.set_ylabel('$f_{disk}^{VLMS}$')
#ax2.plot(f,f,color='w')

mask=(np.abs(xx-yy)/xx<Pequal_tolerance)
if 't' in showtol.lower(): plt.plot(xx[mask].flatten(),yy[mask].flatten(),'w.',mec='w',alpha=0.2)
print '#-----------------------------------------------------------------'
print 'Pequal_x = %.4f%s' % (np.sum(Pxy[mask])*100,'%')
mask=(np.abs(xx-yy)/yy<Pequal_tolerance)
#print 'Pequal_y = %.4f%s' % (np.sum(Pxy[mask])*100,'%')
print 'Pequal computed with tolerance ', Pequal_tolerance

fig.savefig(outfile)
plt.show()
