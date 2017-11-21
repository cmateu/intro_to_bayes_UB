#!/usr/bin/env python
import numpy as np
import scipy
import scipy.interpolate
import pylab as plt
import sys
import argparse

def select_prior(Nprior):

  if Nprior==1:  prior,col=1.,'royalblue'
  elif Nprior==2:
     prior=np.exp(-(H-0.5)**2/(2.*(0.1)**2))+0.3
     col='forestgreen'
  elif Nprior==3:
     prior=np.exp(-H/0.05) + np.exp((H-1.)/0.05) + 0.1
     col='orange'
  else:
     prior=1.
     col='firebrick'

  return (prior,col)
 
def Likelihood(H,N,Nh):
   return (H**Nh)*((1-H)**(N-Nh))

#-----------------------------------------------------------------------------------------------------
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
  #sigma_levels=np.array([0.5,0.75,0.25])

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



#===================================================================================================

__what__= sys.argv[0]+": This program illustrates the coin example for some user-defined inputs."
#
parser = argparse.ArgumentParser(description=__what__)

parser.add_argument('Nheads',metavar="Nheads", help="Number of coin tosses", nargs=1)
parser.add_argument('Ntot',metavar="Ntot", help="Observed number of heads", nargs=1)
parser.add_argument("-p","--priorkeys", help="Select Prior Key [1=Uniform,2=Gaussian,3=Weird]. Options: single or space-separated key list. Default is 1", nargs='+',default=[1,])
parser.add_argument("-o","--outplot", help="Output plot filename", nargs=1,default='moneda.png')
parser.add_argument("-f","--figsave", help="Save figure",action='store_true',default=False)
parser.add_argument("-m","--median", help="Show median",action='store_true',default=False)
parser.add_argument("-ci", help="Show confidence intervals",action='store_true',default=False)

args = parser.parse_args()
Nprior_l=args.priorkeys
Ntot=np.int(args.Ntot[0])
Nh=np.int(args.Nheads[0])

fig=plt.figure(1,figsize=(12,6))

H=np.linspace(0.,1.,101) #The coin bias
i=0

for N in [0,Ntot]:
  i=i+1
  #Toss coin and save the coin tosses appended to the previous ones
  #------------
  #Likelihood 
  if N==0: likelihood=Likelihood(H,N,0)
  else: likelihood=Likelihood(H,N,Nh)

  ax=fig.add_subplot(1,2,i)
  #Compute and plot posterior for different priors
  for Nprior in Nprior_l:


    prior,pcol=select_prior(np.int(Nprior))
    posterior=prior*likelihood
    ax.plot(H,posterior/np.max(posterior),'-',c=pcol)
    Hbest=H[posterior.argmax()]
    Psum=posterior.cumsum()/posterior.sum()
    Hmedian=H[Psum<=0.5][-1]
    if args.median and N>0: ax.axvline(x=Hmedian,ls='-',lw=2,c=pcol,label='median')
    Hmax=H[np.argmax(posterior)]

    #Confidence intervals
    if N>0:
     print '-----Prior %s-----' % (Nprior)
     print "Posterior Mode at=%.2f" % (Hmax)

     ci_tuple=get_1d_confidence_intervals(H,posterior)
     lss=['--',';',':']
     fs=[3.,2.,1.]
     posterior_norm=posterior/np.max(posterior)
     if args.ci:
      for ii in range(3):
        #ax.axvline(ci_tuple[2][ii],ls=lss[0],lw=fs[ii],c=pcol)
        #ax.axvline(ci_tuple[3][ii],ls=lss[0],lw=fs[ii],c=pcol)
        mask_ci=(H>=ci_tuple[2][ii]) & (H<=ci_tuple[3][ii])
        ax.fill_between(H[mask_ci],0*posterior_norm[mask_ci],posterior_norm[mask_ci],color=pcol,alpha=0.3)
        print "%d-sigma h=[%.2f,%.2f]" % (ii+1,ci_tuple[2][ii],ci_tuple[3][ii])

  #-------Format stuff-----------------------------------------------
  if i==1: print 'N=%-5d, Nh=%-d, Hmode=%5.3f, Hmedian=%5.3f' % (N,Nh,Hbest,Hmedian)
  ax.set_xlim(0,1.01)
  ax.set_ylim(0,1.1)
  if ax.is_last_row(): ax.set_xlabel('$h$')
  if ax.is_first_col(): ax.set_ylabel('$P(h|N_h,N)$')
  ax.set_title('T=%d   H=%d' % (N-Nh,Nh),fontsize='small')
  ax.text(0.06,1.06,'N=%d' % (N),fontsize=12,horizontalalignment='left')
  ax.grid(which='both')   
  if not ax.is_last_row(): ax.xaxis.set_ticklabels([])
  if not ax.is_first_col(): ax.yaxis.set_ticklabels([])


#-------------------------Save plot---------------------
if args.figsave: plt.savefig(args.outplot[0])
plt.show()
#-----
