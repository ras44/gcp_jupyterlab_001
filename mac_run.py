import numpy as np
import pandas as pd
from scipy.special import ndtri,ndtr
from scipy.stats import multivariate_normal

corrs=['00','90']
#DIMS=[1,2,3,4,5,6,7,8,9,10]
DIMS=[1,2]

# kappas=[0.0,1.5,3.0,0.0,1.5,3.0,0.0,1.5,3.0]
# gammas=[0.5,0.5,0.5,1.0,1.0,1.0,3.0,3.0,3.0]
# kappas=[0.0,1.5,3.0,0.0,1.5,3.0,0.0,1.5,3.0]
# gammas=[0.5,0.5,0.5,1.0,1.0,1.0,3.0,3.0,3.0]
kappas=[0.0, 1.5]
gammas=[1.0, 1.0]



Np=10
pm=(np.linspace(1,Np,Np)-0.5)/Np # equally spaced p-values between (0,1)
Zm=ndtri(pm) # Z-values for the p-values

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
        
        dim=DIMS[cdim]
        corr=float(corrs[ccorr])/100.0
        
        mean = np.zeros(dim)
        covar = corr*np.ones([dim,dim],float)
        np.fill_diagonal(covar,1.0)

        dist = multivariate_normal(mean=mean, cov=covar)  # define a mv-normal dist with mean=0 and cov=covar

        Zrep=np.tile(Zm,(dim,1)).T  # a matrix of repeated Zm values for each dim, as np->Inf, zrep->[-Inf,Inf]
        TF[(corrs[ccorr],dim)]=dist.cdf(Zrep)  # TF represents prob under the null that value is < each Zrep (1-FWER)

        TPR[(corrs[ccorr],dim)]=1.0-TF[(corrs[ccorr],dim)]  # TPR represents prob under null that value is > Zrep (FWER)
        TPRone[(corrs[ccorr],dim)]=1.0-TF[(corrs[ccorr],dim)] # A copy of TPR
        omp=np.append(np.append([0.0],TF[(corrs[ccorr],dim)].to_numpy()),[1.0]) # TF values bookended by 0,1
        for ccorrp in range(0,len(corrs)):
            corrp=float(corrs[ccorrp])/100.0
            covarp = corrp*np.ones([dim,dim],float)
            np.fill_diagonal(covarp,1.0)
            distp = multivariate_normal(mean=mean, cov=covarp)  # distp is another mv-normal with mean=0 and cov=covarp
            for cs in range(0,len(kappas)):
                kap=kappas[cs]
                gam=gammas[cs]
                TPR[(corrs[ccorr],dim,corrs[ccorrp],kap,gam)]=1.0-distp.cdf(-kap+gam*Zrep) # prob under null that value is > -kap+gam*Zrep ("rescaled" FWER)
                ttpr=np.append([1.0],np.append(TPR[(corrs[ccorr],dim,corrs[ccorrp],kap,gam)].to_numpy(),[0.0]))
                gini=2.0 * np.trapz( ttpr, x=omp ) - 1.0  # gini compares distribution of FWER with rescaled FWER
                DP[(corrs[ccorr],dim,corrs[ccorrp],kap,gam)]=[gini]
                print("corr: ", corr, " dim: ", dim, " corrp:", corrp, " kap: ", kap, " gam: ", gam, " DP gini: ", gini)
                Zrepcpy=Zrep.copy()
                Zrepcpy[:,0]=-kap+gam*Zrep[:,0]  # now we only replace the first column with the rescaled Zrep  
                TPRone[(corrs[ccorr],dim,corrs[ccorrp],kap,gam)]=1.0-distp.cdf(Zrepcpy)
                ttpr=np.append([1.0],np.append(TPRone[(corrs[ccorr],dim,corrs[ccorrp],kap,gam)].to_numpy(),[0.0]))
                gini=2.0 * np.trapz( ttpr, x=omp ) - 1.0
                DPone[(corrs[ccorr],dim,corrs[ccorrp],kap,gam)]=[gini]
        print("")

#     TF.to_csv('TF_'+label+'.csv')
#     TPR.to_csv('TPR_'+label+'.csv')
#     DP.transpose().to_csv('DP_'+label+'.csv')
#     TPRone.to_csv('TPRone_'+label+'.csv')
#     DPone.transpose().to_csv('DPone_'+label+'.csv')