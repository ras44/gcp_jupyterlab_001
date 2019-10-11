import numpy as np
import pandas as pd
from scipy.special import ndtri,ndtr
from scipy.stats import multivariate_normal

corrs=['00','30','60','90']
#DIMS=[1,2,3,4,5,6,7,8,9,10]
DIMS=[1,2]

# kappas=[0.0,1.5,3.0,0.0,1.5,3.0,0.0,1.5,3.0]
# gammas=[0.5,0.5,0.5,1.0,1.0,1.0,3.0,3.0,3.0]
kappas=[0.0,1.5,3.0,0.0,1.5,3.0,0.0,1.5,3.0]
gammas=[0.5,0.5,0.5,1.0,1.0,1.0,3.0,3.0,3.0]


Np=100
pm=(np.linspace(1,Np,Np)-0.5)/Np
Zm=ndtri(pm)

for ccorr in range(0,len(corrs)):
    label=corrs[ccorr]
    TF=pd.DataFrame()
    #TF['Zm']=Zm
    TF['pm']=pm
    
    TPR=pd.DataFrame()
    TPR['pm']=pm
    TPRone=pd.DataFrame()
    TPRone['pm']=pm
    
    DP=pd.DataFrame()
    DPone=pd.DataFrame()

    for cdim in range(0,len(DIMS)):
        print(ccorr,cdim)
        
        dim=DIMS[cdim]
        corr=float(corrs[ccorr])/100.0
        
        mean = np.zeros(dim)
        covar = corr*np.ones([dim,dim],float)
        np.fill_diagonal(covar,1.0)

        dist = multivariate_normal(mean=mean, cov=covar)

        Zrep=np.tile(Zm,(dim,1)).T
        TF[(corrs[ccorr],dim)]=dist.cdf(Zrep)

        TPR[(corrs[ccorr],dim)]=1.0-TF[(corrs[ccorr],dim)]
        TPRone[(corrs[ccorr],dim)]=1.0-TF[(corrs[ccorr],dim)]
        omp=np.append(np.append([0.0],TF[(corrs[ccorr],dim)].to_numpy()),[1.0])
        for ccorrp in range(0,len(corrs)):
            corrp=float(corrs[ccorrp])/100.0
            covarp = corrp*np.ones([dim,dim],float)
            np.fill_diagonal(covarp,1.0)
            distp = multivariate_normal(mean=mean, cov=covarp)
            for cs in range(0,len(kappas)):
                kap=kappas[cs]
                gam=gammas[cs]
                TPR[(corrs[ccorr],dim,corrs[ccorrp],kap,gam)]=1.0-distp.cdf(-kap+gam*Zrep)
                ttpr=np.append([1.0],np.append(TPR[(corrs[ccorr],dim,corrs[ccorrp],kap,gam)].to_numpy(),[0.0]))
                gini=2.0 * np.trapz( ttpr, x=omp ) - 1.0
                DP[(corrs[ccorr],dim,corrs[ccorrp],kap,gam)]=[gini]
#                print(gini)
                Zrepcpy=Zrep.copy()
                Zrepcpy[:,0]=-kap+gam*Zrep[:,0]
                TPRone[(corrs[ccorr],dim,corrs[ccorrp],kap,gam)]=1.0-distp.cdf(Zrepcpy)
                ttpr=np.append([1.0],np.append(TPRone[(corrs[ccorr],dim,corrs[ccorrp],kap,gam)].to_numpy(),[0.0]))
                gini=2.0 * np.trapz( ttpr, x=omp ) - 1.0
                DPone[(corrs[ccorr],dim,corrs[ccorrp],kap,gam)]=[gini]

#     TF.to_csv('TF_'+label+'.csv')
#     TPR.to_csv('TPR_'+label+'.csv')
#     DP.transpose().to_csv('DP_'+label+'.csv')
#     TPRone.to_csv('TPRone_'+label+'.csv')
#     DPone.transpose().to_csv('DPone_'+label+'.csv')