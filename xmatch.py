from astropy.io import fits
import pandas as pd
pd.options.mode.chained_assignment = None 
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle
from astropy.modeling import models, fitting
from tqdm import tqdm
import pickle
import scipy.spatial





def trapezoidal_area(xyz):
    """Calculate volume under a surface defined by irregularly spaced points
    using delaunay triangulation. "x,y,z" is a <numpoints x 3> shaped ndarray."""
    d = scipy.spatial.Delaunay(xyz[:,:2])
    tri = xyz[d.vertices]

    a = tri[:,0,:2] - tri[:,1,:2]
    b = tri[:,0,:2] - tri[:,2,:2]
    proj_area = np.cross(a, b).sum(axis=-1)
    zavg = tri[:,:,2].sum(axis=1)
    vol = zavg * np.abs(proj_area) / 6.0
    return vol.sum()





def match(fname='C02_master_merged.fits',depth=5,outfile='results.p',targfile='xmatch.p'):

    '''
    Matches against an EPIC catalog, returns array of matches to a specified depth.
    Please pass a pickled pandas dataframe of the EPIC catalog you want to match against.
    '''
    h=fits.open(fname)
    mag=25-2.5*np.log10(h[1].data['Total_flux'])
    RA,Dec=h[1].data['RA']*u.radian,h[1].data['DEC']*u.radian
    RA,Dec=RA.to(u.deg),Dec.to(u.deg)

    #Ensure all targets are unique
    unq=np.unique(RA*Dec,return_index=True)[1]
    RA,Dec,mag=RA[unq],Dec[unq],mag[unq]
    
    targlist=pickle.load(open(targfile,'rb'))
    EPICcatalog=SkyCoord(targlist.k2_ra*u.deg,targlist.k2_dec*u.deg)
    INPUTcatalog = SkyCoord(ra=RA, dec=Dec)
    
    columns=['RA','Dec','InputMag']
    preffixes=['EPICKpMag_','EPICdKpMag_','EPICd2d_','EPICRA_','EPICDec_','EPICID_','KepFlag_']
    for i in np.arange(depth)+1:
        for p in preffixes:
            columns.append('{}{}'.format(p,i))
    [columns.append(i) for i in ['Nth_Neighbour','Xflag','EPICID','KpMag']]

    results=pd.DataFrame(columns=columns)

    results['RA']=RA
    results['Dec']=Dec
    results['InputMag']=mag
    results['Nth_Neighbour']=0
    results['Xflag']=-1
    
    for n in np.arange(depth)+1:
        idx, d2d, d3d = INPUTcatalog.match_to_catalog_sky(EPICcatalog,nthneighbor=n)  
        gri=np.where(np.asarray(targlist.kepflag)[idx]=='gri')[0]
        if n==1:
            zeropoint=np.median(np.asarray(targlist.kp)[idx][gri]-mag[gri])

        pos=np.arange(np.where(results.columns == '{}{}'.format(preffixes[0],n))[0][0],np.where(results.columns == '{}{}'.format(preffixes[-1],n))[0][0]+1)
        nkeys=results.keys()[pos]
        ar=np.transpose([targlist.kp[idx],
                         targlist.kp[idx]-(mag+zeropoint),
                         d2d.to(u.arcsecond).value,
                         targlist.k2_ra[idx],
                         targlist.k2_dec[idx],
                         targlist.id[idx],
                         targlist.kepflag[idx]])
        results[nkeys]=ar
    results.to_pickle(outfile)



def starmap(i,results,frame,depth=5):

    '''
    Plots a star map for a given source to check how close the nearest match is.
    '''
    def circle_scatter(axes, x_array, y_array, radius=0.5, **kwargs):
        for x, y in zip(x_array, y_array):
            circle = Circle((x,y), radius=radius, **kwargs)
            axes.add_patch(circle)
        return True
    

    cols,ras,decs=[],[],[]
    for n in np.arange(depth)+1:
        ra,dec,col,flag=results.loc[i,'EPICRA_{}'.format(n)],results.loc[i,'EPICDec_{}'.format(n)],results.loc[i,'EPICdKpMag_{}'.format(n)],results.loc[i,'KepFlag_{}'.format(n)]
        cols.append(np.abs(col))
        ras.append(ra)
        decs.append(dec)
        if flag=='gri':
            color='lime'
        if flag!='gri':
            color='black'
        
        #plt.text(ra+0.001,dec+0.001,'{}'.format(n),fontsize=15)
        circle_scatter(frame, [ra], [dec], radius=(4*u.arcsec).to(u.deg).value, alpha=0.3, color=color,zorder=-5)
    n=results.loc[i,'Nth_Neighbour']-1
    if n>-1:
        circle_scatter(frame, [ras[n]], [decs[n]], radius=(8*u.arcsec).to(u.deg).value, alpha=0.5, facecolor='white',edgecolor='C3',zorder=-1,ls='--')
    
    sc=frame.scatter(ras,decs,c=cols,vmin=0,vmax=5)
    circle_scatter(frame, [results.loc[i,'RA']], [results.loc[i,'Dec']], radius=(1.2*u.arcsec).to(u.deg).value, alpha=1, color='C3',zorder=1)
    frame.set_xlabel('Dec')
    frame.set_ylabel('RA')
    frame.set_title('Star {}'.format(i))
    return 



def nonsym_model(x,ppos,pneg,zeropoint):

    '''
    makes a non symmetric lorentzian model
    '''
    if len(np.shape(x))>1:
        xs=x.ravel()
    else:
        if isinstance(x,float)==True:
            xs=[x]
        else:
            xs=np.copy(x)
    result=[]
    for i in xs:
        if i-zeropoint<0.:
            result.append(pneg(i-zeropoint))
        else:
            result.append(ppos(i-zeropoint))

    if len(np.shape(x))>1:
        return np.reshape(result,(np.shape(x)[0],np.shape(x)[1]))
    else:
        if isinstance(x,float)==True:
            return result[0]
        else:
            return result




def prob(dist,dmag,plot=False):

    '''
    Calculates the probability model for distance and delta mag.
    Both are based on Lorentzian models.
    '''
    fit_p = fitting.LevMarLSQFitter()
    
    #Fit the change in magnitude
    h=np.histogram(dmag,200)
    x,y=(h[1][:-1]+h[1][1]-h[1][1]),h[0]
    zeropoint=fit_p(models.Gaussian1D(),x, y).mean.value
    x-=zeropoint


    p_init = models.Lorentz1D(x_0=0)
    p_init.x_0.fixed=True

    ok=np.where(x>=0)
    
    ppos = fit_p(p_init, x[ok], y[ok])

    ok=np.where(x<0)
    fit_p = fitting.LevMarLSQFitter()
    pneg = fit_p(p_init, x[ok], y[ok])


    #Fit the distance
    h=np.histogram(dist,200)
    x,y=(h[1][:-1]+h[1][1]-h[1][1]),h[0]


    p_init = models.Lorentz1D(x_0=0)
    p_init.x_0.fixed=True

    ok=np.where(x>=0.1)
    fit_p = fitting.LevMarLSQFitter()
    pdist = fit_p(p_init, x[ok], y[ok])

    X,Y=np.meshgrid(np.linspace(0,10,200),np.linspace(-5,5,200))
    surf=pdist(X)*nonsym_model(Y,ppos,pneg,zeropoint)
    norm=trapezoidal_area(np.asarray([X.ravel(),Y.ravel(),surf.ravel()]).T)
    surf/=norm
        
    if plot==True:
        #Plot up the surface and show it
        plt.figure()
        plt.scatter(dist,dmag,c='C0',s=0.1)
        plt.ylim(-2,2)
        plt.contour(X,Y,surf,colors='C1',levels=np.linspace(0,np.max(surf),20))
        plt.xlim(0.,5)
        plt.xlabel('Distance from Nearest Source (arcseconds)')
        plt.ylabel('Difference in Magnitude')
    return pdist,ppos,pneg,zeropoint,norm







def fit(infile='C02_master_merged.fits',catalog='xmatch.p',run_match=False,depth=5,contaminationlim=12,blendlim=2,accept=6):

    '''
    Fit an input catalog of RAs, Decs and Mags.

    contaminationlim :  the distance limit where a source is no longer considered contaminated.
                        Set to 12 arcseconds (3 pixels) by default. 

    blendedlim :        The tolerance for a source to be considered blended. Set to 2 arcseconds by
                        default. i.e. Two sources would have to be the same distance away from the target
                        with a tolerance of 2 arcseconds. Set this to a wider tolerance to find more blends.

    accept:             The distance up to which all sources should be accepted as the correct source. 
                        Set to 6 arcseconds (1.5 pixels) by default. 
    '''


    if run_match==True:
    	print 'Matching against {}'.format(catalog)
    	match(fname=infile,depth=depth,outfile='results.p',targfile=catalog)

    results=pd.read_pickle('results.p')
    results['PROB']=0


    print 'Modeling distance and magnitude distributions'
    pos=np.where(results.KepFlag_1=='gri')[0]
    gri_distmodel,gri_ppos,gri_pneg,gri_zeropoint,gri_norm=prob(results.EPICd2d_1[pos],results.EPICdKpMag_1[pos],plot=True)
    plt.title('gri model')
    plt.savefig('images/gri_model.png',dpi=150,bbox_inches='tight')
    plt.close()

    pos=np.where(results.KepFlag_1!='gri')[0]
    ngri_distmodel,ngri_ppos,ngri_pneg,ngri_zeropoint,ngri_norm=prob(results.EPICd2d_1[pos],results.EPICdKpMag_1[pos],plot=True)
    plt.title('ngri model')
    plt.savefig('images/ngri_model.png',dpi=150,bbox_inches='tight')
    plt.close()

    dists=np.zeros((len(results),depth))
    dmags=np.zeros((len(results),depth))
    flags=np.chararray((len(results),depth),itemsize=3)
    ids=np.zeros((len(results),depth),dtype=int)

    for n in xrange(depth):
        dists[:,n]=np.asarray(results['EPICd2d_{}'.format(n+1)])
        dmags[:,n]=np.asarray(results['EPICdKpMag_{}'.format(n+1)])
        flags[:,n]=np.asarray(results['KepFlag_{}'.format(n+1)])
        ids[:,n]=np.asarray(results['EPICID_{}'.format(n+1)])

    print 'Calculating probability'
    probs=np.copy(dists)*0.
    for i,ds,ms,fs in tqdm(zip(xrange(len(dists)),dists,dmags,flags)):
        for j,d,m,f in zip(xrange(depth),ds,ms,fs):
            if f=='gri':
                probs[i,j]=(gri_distmodel(d)*nonsym_model(m,gri_ppos,gri_pneg,gri_zeropoint))/gri_norm
            if f!='gri':
                probs[i,j]=(ngri_distmodel(d)*nonsym_model(m,ngri_ppos,ngri_pneg,ngri_zeropoint))/ngri_norm


    results.Nth_Neighbour=np.argmax(probs,axis=1)+1
    results.PROB=np.max(probs,axis=1)
    results.Xflag=[f[i] for f,i in zip(flags,np.argmax(probs,axis=1))]
    results.EPICID=[f[i] for f,i in zip(ids,np.argmax(probs,axis=1))]

    d=[]
    for n,i in zip(np.asarray(results.Nth_Neighbour),xrange(len(results))):
        d.append(results.loc[i,'EPICd2d_{}'.format(n)])
    results['EPICd2d']=np.asarray(d,dtype=float)

    d=[]
    for n,i in zip(np.asarray(results.Nth_Neighbour),xrange(len(results))):
        d.append(results.loc[i,'EPICKpMag_{}'.format(n)])
    results['EPICKpMag']=np.asarray(d,dtype=float)


    plt.scatter(np.log10(dists)[flags=='gri'],dmags[flags=='gri'],c=np.log10(probs)[flags=='gri'],s=-np.log10(probs)[flags=='gri'])
    plt.xlim(-3,1.5)
    plt.ylim(-10,10)
    plt.title('gri targets')
    cbar=plt.colorbar()
    cbar.set_label('Probability')
    plt.xlabel('Distance to Target (log(arcsecond))')
    plt.ylabel('Magnitude Difference')
    plt.savefig('images/gri.png',dpi=150,bbox_inches='tight')
    plt.close()

    plt.scatter(np.log10(dists)[flags!='gri'],dmags[flags!='gri'],c=np.log10(probs)[flags!='gri'],s=-np.log10(probs)[flags!='gri'])
    plt.xlim(-3,1.5)
    plt.ylim(-10,10)
    plt.title('!gri targets')
    cbar.set_label('log$_{10]$ Prob')
    cbar=plt.colorbar()
    cbar.set_label('Probability')
    plt.xlabel('Distance to Target (log(arcsecond))')
    plt.ylabel('Magnitude Difference')
    plt.savefig('images/ngri.png',dpi=150,bbox_inches='tight')
    plt.close()

    pos=results.Xflag=='gri'
    plt.scatter(np.log10(results.EPICd2d)[pos],np.log10(results.PROB[pos]),s=1,label='gri')
    pos=results.Xflag!='gri'
    plt.scatter(np.log10(results.EPICd2d)[pos],np.log10(results.PROB[pos]),s=1,label='ngri')
    plt.xlabel('log10(Distance')
    plt.ylabel('Probability')
    plt.axhline(-7,c='black',ls='--')
    plt.legend()
    plt.savefig('images/distprob.png',dpi=150,bbox_inches='tight')
    plt.close()

    print len(np.where(results.Nth_Neighbour>1)[0]),'targets where Nth Neighbour was suboptimal'
    #Remove duplicates
    mask=np.copy(probs)*0.
    for i,n in enumerate(np.asarray(results.Nth_Neighbour-1)):
        mask[i,n]=1

    print 'Reshuffling duplicates'
    dupes=np.where(np.asarray([len(np.where(results.EPICID==s)[0]) for s in results.EPICID])>1)[0]
    while len(dupes)!=0:
        for d in tqdm(dupes):
            i=np.asarray(results.loc[d,'EPICID'])

            pos=np.where(np.asarray(results.EPICID)==i)[0]    

            #Indexes that need to be refound:
            bad=np.asarray(list(set(np.arange(len(pos)))-set([np.argmax(np.max(probs[pos,:],axis=1))])))
            for b in bad:
                remain=np.where(mask[pos[b]]==0)[0]
                if len(remain)==0:
                    results.PROB[pos[b]]=0
                    results.loc[pos[b],'Nth_Neighbour']=0
                    results.loc[pos[b],'PROB']=0
                    results.loc[pos[b],'Xflag']='bad'
                    results.loc[pos[b],'EPICID']=0
                    results.loc[pos[b],'EPICd2d']=99
        
                nth=remain[np.argmax(probs[pos[b],remain])]+1
                mask[pos[b],nth-1]=1
                results.PROB[pos[b]]=np.max(probs[pos[b],remain])
                results.loc[pos[b],'Nth_Neighbour']=nth
                results.loc[pos[b],'PROB']=np.max(probs[pos[b],remain])
                results.loc[pos[b],'Xflag']=flags[pos[b],remain[np.argmax(probs[pos[b],remain])]]
                results.loc[pos[b],'EPICID']=results.loc[pos[b],'EPICID_{}'.format(nth)]
                results.loc[pos[b],'EPICd2d']=results.loc[pos[b],'EPICd2d_{}'.format(nth)]
        dupes=np.where(np.asarray([len(np.where(results.EPICID==s)[0]) for s in results.EPICID])>1)[0]


    contaminationlim=12
    contaminated=np.empty((len(results),depth),dtype=bool)
    for n in np.arange(depth)+1:
        try:
            check1=(results['EPICd2d_{}'.format(n-1)]<=contaminationlim)&(results.Nth_Neighbour==n)
        except:
            check1=[False]*len(results)
        try:
            check2=(results['EPICd2d_{}'.format(n+1)]<=contaminationlim)&(results.Nth_Neighbour==n)
        except:
            check2=[False]*len(results)
        contaminated[:,n-1]=np.any([check1,check2],axis=0)

    contaminated=np.any(contaminated,axis=1)

    blendlim=2
    blended=np.empty((len(results),depth),dtype=bool)
    for n in np.arange(depth)+1:

        now=np.where(results.Nth_Neighbour==n)[0]
        
        try:
            bl0=results['EPICd2d_{}'.format(n-1)]
        except:
            bl0=np.zeros(len(results))-99
        bl1=results['EPICd2d_{}'.format(n)]
        try:
            bl2=results['EPICd2d_{}'.format(n+1)]
        except:
            bl2=np.zeros(len(results))+99
        
        prv=np.asarray(bl1[now]-bl0[now])
        nxt=np.asarray(bl2[now]-bl1[now])
        
        bl=[False]*len(results)
        
        for i in now[np.any([nxt<blendlim,prv<blendlim],axis=0)]:
            bl[i]=True
        bl=np.all([bl,bl1<=accept],axis=0)
        blended[:,n-1]=bl

     
    blended=np.any(blended,axis=1)
    blended=np.all([contaminated,blended],axis=0)


    accept=6
    bad=np.where((np.min(dists,axis=1)>=accept)&(blended==False))[0]
    good=np.where((np.min(dists,axis=1)<=accept)&(blended==False))[0]


    results['contaminated']=contaminated
    results['blended']=blended
    results['xmatch']=0

    #good
    results.xmatch[good]=2

    #bad
    results.PROB[bad]=0
    results.loc[bad,'Nth_Neighbour']=0
    results.loc[bad,'PROB']=0
    results.loc[bad,'EPICID']=0
    results.loc[bad,'EPICd2d']=99


    print '------------------------'
    print len(np.where(contaminated==True)[0]),'contaminated sources'
    print len(np.where(blended==True)[0]),'blended sources'
    print len(bad),'unmatched sources'
    print len(good),'matched sources'
    print '------------------------'
    results.to_pickle(open('results_probabilities.p','wb'))
    print 'Saved to ','results_probabilities.p'
    return 