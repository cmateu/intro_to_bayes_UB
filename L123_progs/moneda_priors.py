#!/usr/bin/env python
import numpy as np
import scipy
import pylab as plt
import sys
import argparse

def lanzar_moneda(Nveces,H=0.73):
   unif=scipy.random.random(Nveces)
   moneda=np.zeros_like(unif)
   moneda[unif<=H]=1
   return moneda

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

#try: 
#  myseed=np.int(sys.argv[1])
#  Nprior=np.int(sys.argv[2])
#except IndexError:
#  sys.exit('Syntax:')

__what__= sys.argv[0]+": This program illustrates the coin example."
#
parser = argparse.ArgumentParser(description=__what__)

parser.add_argument("-p","--priorkeys", help="Select Prior Key [1=Uniform,2=Gaussian,3=Weird]. Options: single or space-separated key list. If set to ALL will use all keys in library. Default is 1", nargs='+',default=[1,])
parser.add_argument("-s","--seed", help="Set seed for random number generator", nargs=1)
parser.add_argument("-o","--outplot", help="Output plot filename", nargs=1,default='moneda.png')
parser.add_argument("-f","--figsave", help="Save figure",action='store_true',default=False)
parser.add_argument("-m","--median", help="Show median",action='store_true',default=False)
parser.add_argument("-hc","--hcoin", help="Coin Bias [0-1]",action='store',default=0.7)

args = parser.parse_args()
Nprior_l=args.priorkeys

#Set coint bias
Htrue=np.float(args.hcoin)

#Set seed
if args.seed:
  print 'Setting seed to:', np.int(args.seed[0]) 
  scipy.random.seed(seed=np.int(args.seed[0])) #1234  #98765
else:
  scipy.random.seed(seed=None)

fig=plt.figure(1,figsize=(14,8))

H=np.linspace(0.,1.,501)
i=0
moneda=np.array([])

for N in [0,1,2,3,4,6,8,12,16,20,32,64,128,256,512,1024]:
  i=i+1
  #Toss coin and save the coin tosses appended to the previous ones
  moneda=np.append(moneda,lanzar_moneda(int(N)-len(moneda),H=Htrue))
  #------------
  r=len(moneda[moneda==1])
  #Likelihood 
  likelihood=(H**r)*((1-H)**(N-r))
  ax=fig.add_subplot(4,4,i)
  #Compute and plot posterior for different priors
  for Nprior in Nprior_l:
    prior,pcol=select_prior(np.int(Nprior))
    posterior=prior*likelihood
    ax.plot(H,posterior/np.max(posterior),'-',c=pcol)
    Hbest=H[posterior.argmax()]
    Psum=posterior.cumsum()/posterior.sum()
    Hmedian=H[Psum<=0.5][-1]
    if args.median: ax.axvline(x=Hmedian,ls='--',lw=1,c=pcol)

  #Plot line around true value of the coin bias
  ax.axvline(x=Htrue,ls='-',lw=1,c='k')

  #-------Format stuff-----------------------------------------------
  #print 'Htrue=',Htrue,'N=',N,'Hmode=',Hbest,'Hmedian=',Hmedian
  print 'Htrue=%.3f, N=%-5d, Hmode=%5.3f, Hmedian=%5.3f' % (Htrue,N,Hbest,Hmedian)
  ax.set_xlim(0,1.01)
  ax.set_ylim(0,1.18)
  if ax.is_last_row(): ax.set_xlabel('$h$')
  if ax.is_first_col(): ax.set_ylabel('$P(h|N_h,N)$')
  ax.set_title('T=%d   H=%d' % (N-r,r),fontsize='small')
  ax.text(0.06,1.01,'N=%d' % (N),fontsize=12,horizontalalignment='left')
  ax.grid(which='both')   
  if not ax.is_last_row(): ax.xaxis.set_ticklabels([])
  if not ax.is_first_col(): ax.yaxis.set_ticklabels([])

#-------------------------Save plot---------------------
if args.figsave: plt.savefig(args.outplot[0])
plt.show()
#-----
