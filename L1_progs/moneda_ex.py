#!/usr/bin/env python
import numpy as np
import scipy
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
 
#===================================================================================================

__what__= sys.argv[0]+": This program illustrates the coin example for some user-defined inputs."
#
parser = argparse.ArgumentParser(description=__what__)

parser.add_argument('Nheads',metavar="Nheads", help="Number of coin tosses", nargs=1)
parser.add_argument('Ntot',metavar="Ntot", help="Observed number of heads", nargs=1)
parser.add_argument("-p","--priorkeys", help="Select Prior Key [1=Uniform,2=Gaussian,3=Weird]. Options: single or space-separated key list. If set to ALL will use all keys in library. Default is 1", nargs='+',default=[1,])
parser.add_argument("-o","--outplot", help="Output plot filename", nargs=1,default='moneda.png')
parser.add_argument("-f","--figsave", help="Save figure",action='store_true',default=False)
parser.add_argument("-m","--median", help="Show median",action='store_true',default=False)

args = parser.parse_args()
Nprior_l=args.priorkeys
Ntot=np.int(args.Ntot[0])
Nh=np.int(args.Nheads[0])

fig=plt.figure(1,figsize=(14,8))

H=np.linspace(0.,1.,101) #The coin bias
i=0

for N in [0,Ntot]:
  i=i+1
  #Toss coin and save the coin tosses appended to the previous ones
  #------------
  #Likelihood 
  if N>0: likelihood=(H**Nh)*((1-H)**(N-Nh))
  else: likelihood=(H**0)*((1-H)**(N-0))
  ax=fig.add_subplot(1,2,i)
  #Compute and plot posterior for different priors
  for Nprior in Nprior_l:
    prior,pcol=select_prior(np.int(Nprior))
    posterior=prior*likelihood
    ax.plot(H,posterior/np.max(posterior),'-',c=pcol)
    Hbest=H[posterior.argmax()]
    Psum=posterior.cumsum()/posterior.sum()
    Hmedian=H[Psum<=0.5][-1]
    if args.median: ax.axvline(x=Hmedian,ls='--',lw=1,c=pcol)

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
