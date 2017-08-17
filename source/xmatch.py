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





def match(RA=None,Dec=None,mag=None,depth=5,outfile='results.p',targfile='epic_1_06jul17.txt'):

    '''
    Matches against an EPIC catalog, returns array of matches to a specified depth.
    Please pass a pickled pandas dataframe of the EPIC catalog you want to match against.
    '''

    targlist=pd.read_csv(targfile,delimiter=',',comment='#')

    targlist=targlist[np.all([targlist.k2_ra>np.min(RA),
            targlist.k2_ra<np.max(RA),
            targlist.k2_dec>np.min(Dec),
            targlist.k2_dec<np.max(Dec)],axis=0)]

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



def starmap(i,results,frame,depth=5,cbar=False):

    '''
    Plots a star map for a given source to check how close the nearest match is.
    '''
    def circle_scatter(axes, x_array, y_array, radius=0.5, **kwargs):
        for x, y in zip(x_array, y_array):
            circle = Circle((x,y), radius=radius, **kwargs)
            axes.add_patch(circle)
        return True
    
    frame.set_facecolor('black')
    cols,ras,decs,ids=[],[],[],[]
    for n in np.arange(depth)+1:
        ra,dec,col,flag,idx=results.loc[i,'EPICRA_{}'.format(n)],results.loc[i,'EPICDec_{}'.format(n)],results.loc[i,'EPICKpMag_{}'.format(n)],results.loc[i,'KepFlag_{}'.format(n)],results.loc[i,'EPICID_{}'.format(n)]
        cols.append(np.abs(col))
        ras.append(ra)
        decs.append(dec)
        ids.append(idx)
        if flag=='gri':
            color='lime'
        if flag!='gri':
            color='white'
        
        #plt.text(ra+0.001,dec+0.001,'{}'.format(n),fontsize=15)
        circle_scatter(frame, [ra], [dec], radius=(4*u.arcsec).to(u.deg).value, alpha=0.3, facecolor=None,edgecolor=color,zorder=20,lw=2)
    n=results.loc[i,'Nth_Neighbour']-1
    sc=plt.scatter(ras,decs,c=cols,cmap=plt.get_cmap('Greys'),vmin=5,vmax=20,s=50)
    if cbar==True:
       cbar=plt.colorbar()
       cbar.set_label('Magnitude')
    if n>-1:
        plt.scatter(ras[n],decs[n],marker='o',edgecolor='C3',s=3000,facecolor='black',zorder=-10,lw=2)
  

   
    circle_scatter(frame, [results.loc[i,'RA']], [results.loc[i,'Dec']], radius=(1.2*u.arcsec).to(u.deg).value, alpha=1, color='C3',zorder=1)
    
    #frame.set_xlabel('Dec')
    #frame.set_ylabel('RA')
    frame.set_xticks([])
    frame.set_yticks([])

    frame.set_title('Star {}'.format(i))

    for j,k,r,d in zip(xrange(depth),ids,ras,decs):
        if j==n:
            continue
        if len(np.where(results.EPICID==k)[0])==1:
            plt.scatter(r,d,marker='o',edgecolor='black',s=1000,facecolor='black',zorder=10,lw=2)
    return 





def calc_prob(pos,results,title=None,plot=True,outfile=None,cap=True):
    if plot==True:
        fig=plt.figure(figsize=(8,7))
        ax1=plt.subplot2grid((6,6),(0,1),colspan=5,rowspan=5)
        H=plt.hist2d(np.log10(results.EPICd2d_1[pos]),np.abs(results.EPICKpMag_1[pos]), bins=40,cmap=plt.get_cmap('Blues'),vmin=10)
        X,Y=np.meshgrid(H[1][:-1]+np.median(H[1][1:]-H[1][0:-1]),H[2][:-1]+np.median(H[2][1:]-H[2][0:-1]))

        cbar=plt.colorbar()
        cbar.set_label('Frequency')
        plt.ylim(6,19)
        plt.xlim(-1.5,1.5)
        plt.xticks([])
        plt.yticks([])
        if title!=None:
            plt.title(title)
    else:   
        H=np.histogram2d(np.log10(results.EPICd2d_1[pos]),np.abs(results.EPICKpMag_1[pos]), bins=40)
        X,Y=np.meshgrid(H[1][:-1]+np.median(H[1][1:]-H[1][0:-1]),H[2][:-1]+np.median(H[2][1:]-H[2][0:-1]))

    p_init = models.Gaussian2D(x_mean=-0.5,y_mean=12)

    fit_p = fitting.LevMarLSQFitter()
    p=fit_p(p_init,X,Y,H[0].T)
    
    h=np.histogram(np.log10(results.EPICd2d_1[pos]),40,normed=True)
    x=(np.arange(len(h[0])))*(np.median(h[1][1:]-h[1][0:-1]))+h[1][0]
    
    if plot==True:
        ax1.contour(X,Y,p(X,Y),colors='C3',alpha=0.5)
        ax2=plt.subplot2grid((6,6),(5,1),colspan=4)
        plt.plot(x,h[0])
        model=np.nansum(p(X,Y),axis=0)
        plt.plot(X[0],np.mean(h[0])*model/np.mean(model))
        plt.xlim(-1.5,1.5)
        plt.xlabel('Distance to Nearest Source (arcseconds)')

    h=np.histogram(results.EPICKpMag_1[pos],200,normed=True)
    x=(np.arange(len(h[0])))*(np.median(h[1][1:]-h[1][0:-1]))+h[1][0]    

    if plot==True:
        ax3=plt.subplot2grid((6,6),(0,0),rowspan=5)
        plt.ylim(6,19)
        plt.plot(h[0],x)
        model=np.nansum(p(X,Y),axis=1)
        plt.plot(np.mean(h[0])*model/np.mean(model),Y[:,0])
        plt.ylabel('Magnitude')

    if cap==True:
        magcap=x[np.argmax(h[0])]

    
    
    mask=np.asarray(np.zeros(np.shape(X)),dtype=bool)
    mask[:,:-1]=np.asarray([(m[1:]-m[0:-1])>=0 for m in p(X,Y)])
    x,y=[],[]
    for i,m in zip(xrange(len(mask)),mask):
        x.append(X[0][np.where(m!=0)[0][-1]])
        y.append(Y[:,0][i])

  
    l=np.polyfit(y,x,1)

    


    dists,mags=np.meshgrid(np.linspace(-3,2,40),np.linspace(5,20))
    prob=[]
    for d,m in zip(dists.ravel(),mags.ravel()):
        if cap==True:
            if m<magcap:
                m=magcap
        if d<=m*l[0]+l[1]:
            prob.append(p(m*l[0]+l[1],m))
        else:
            prob.append(p(d,m))

       
    norm=trapezoidal_area(np.transpose([dists.ravel(),mags.ravel(),prob]))
    if plot==True:
        ax1.contour(dists,mags,np.reshape(prob,np.shape(dists)),colors='C1',alpha=0.5)
        if outfile!=None:
            plt.savefig(outfile,dpi=150,bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    return fig,p,norm,l,magcap





def fit(infile=infile,catalog='epic_1_06jul17.txt',run_match=False,depth=5,contaminationlim=12,blendlim=2,goodthresh=-6.,badthresh=-7.5):

    '''
    Fit an input catalog of RAs, Decs and Mags.

    run_match :         Whether to run the matching algorithm. Takes about 5 minutes and there is no need to repeat it.

    depth :             How many neighbors to match up to

    contaminationlim :  the distance limit where a source is no longer considered contaminated.
                        Set to 12 arcseconds (3 pixels) by default. 

    blendedlim :        The tolerance for a source to be considered blended. Set to 2 arcseconds by
                        default. i.e. Two sources would have to be the same distance away from the target
                        with a tolerance of 2 arcseconds. Set this to a wider tolerance to find more blends.

    accept:             The distance up to which all sources should be accepted as the correct source. 
                        Set to 6 arcseconds (1.5 pixels) by default. 

    goodthresh:         The probability cut off above which to accept all matchers

    badthresh:          The probability cut off below which all matches should be rejected
    '''


    if run_match==True:
        h=fits.open(infile)
        mag=25-2.5*np.log10(h[1].data['Total_flux'])
        RA,Dec=h[1].data['RA']*u.radian,h[1].data['DEC']*u.radian
        RA,Dec=RA.to(u.deg),Dec.to(u.deg)

        #Ensure all targets are unique
        unq=np.unique(RA*Dec,return_index=True)[1]
        RA,Dec,mag=RA[unq],Dec[unq],mag[unq]
        
    	print 'Matching against {}'.format(catalog)
    	match(RA,Dec,mag,depth=depth,outfile='results.p',targfile=catalog)

    results=pd.read_pickle('results.p')
    results['PROB']=0

    ax,gri_model,gri_norm,gri_l,gri_cap=calc_prob(results.KepFlag_1!='none',results,cap=True,title='gri targets',outfile='images/gri_model.png')
    ax,ngri_model,ngri_norm,ngri_l,ngri_cap=calc_prob((results.KepFlag_1!='gri')&(results.EPICd2d_1<4)&(results.EPICKpMag_1<=18),results,cap=True,title='gri targets',outfile='images/ngri_model.png')

    
    def assign_prob(dists,mags,flags,justgri=False):
        probs=[]
        if justgri==False:
            for d,m,f in tqdm(zip(dists.ravel(),mags.ravel(),flags.ravel())):
                if f=='gri':
                    if m<gri_cap:
                        m=gri_cap
                    if d<=m*gri_l[0]+gri_l[1]:
                        probs.append(gri_model(m*gri_l[0]+gri_l[1],m)/gri_norm)
                    else:
                        probs.append(gri_model(d,m)/gri_norm)
                if f!='gri':
                    if m<ngri_cap:
                        m=ngri_cap
                    if d<=m*ngri_l[0]+ngri_l[1]:
                        probs.append(ngri_model(m*ngri_l[0]+ngri_l[1],m)/ngri_norm)
                    else:
                        probs.append(ngri_model(d,m)/ngri_norm)   
            return np.reshape(probs,(np.shape(dists)))
        else:
            for d,m,f in tqdm(zip(dists.ravel(),mags.ravel(),flags.ravel())):
                if m<gri_cap:
                    m=gri_cap
                if d<=m*gri_l[0]+gri_l[1]:
                    probs.append(gri_model(m*gri_l[0]+gri_l[1],m)/gri_norm)
                else:
                    probs.append(gri_model(d,m)/gri_norm) 
            return np.reshape(probs,(np.shape(dists)))



    dists=np.zeros((len(results),depth))
    mags=np.zeros((len(results),depth))
    flags=np.chararray((len(results),depth),itemsize=3)
    ids=np.zeros((len(results),depth),dtype=int)

    for n in xrange(depth):
        dists[:,n]=np.asarray(results['EPICd2d_{}'.format(n+1)])
        mags[:,n]=np.asarray(results['EPICKpMag_{}'.format(n+1)])
        flags[:,n]=np.asarray(results['KepFlag_{}'.format(n+1)])
        ids[:,n]=np.asarray(results['EPICID_{}'.format(n+1)])
    probs=assign_prob(np.log10(dists),mags,flags,justgri=True)


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

    plt.scatter(np.log10(dists),mags,c=np.log10(probs),s=-np.log10(probs),vmin=badthresh,vmax=goodthresh)
    plt.xlim(-3,2)
    plt.ylim(5,22)
    cbar=plt.colorbar()
    cbar.set_label('Probability')
    plt.xlabel('Distance to Target (log(arcsecond))')
    plt.ylabel('Magnitude Difference')
    plt.savefig('images/prob.png',dpi=150,bbox_inches='tight')
    plt.close()



    pos=results.Xflag=='gri'
    plt.scatter(np.log10(results.EPICd2d)[pos],np.log10(results.PROB[pos]),s=0.1,label='gri')
    pos=results.Xflag!='gri'
    plt.scatter(np.log10(results.EPICd2d)[pos],np.log10(results.PROB[pos]),s=0.1,label='ngri')
    plt.xlabel('log10(Distance')
    plt.ylim(-9,-2)
    plt.ylabel('Probability')
    plt.axhline(goodthresh,c='black',ls='--')
    plt.legend()
    plt.savefig('images/distprob.png',dpi=150,bbox_inches='tight')
    plt.close()

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
        bl=np.all([bl,bl1<=contaminationlim],axis=0)
        blended[:,n-1]=bl

     
    blended=np.any(blended,axis=1)

    results['contaminated']=contaminated
    results['blended']=blended


    good=np.any([(np.log10(results.PROB)>goodthresh)],axis=0)
    bad=np.any([(np.log10(results.PROB)<badthresh)],axis=0)
   

    bad=np.where(bad==True)[0]
    good=np.where(good==True)[0]


    results['xmatch']=0.5

    #good
    results.xmatch[good]=1
    results.xmatch[results.blended]=0.5

    #bad
    results.xmatch[bad]=0
    results.EPICID[bad]=-99

    #remaining n neighbours must be blends
#    results.loc[np.where(results.Nth_Neighbour>1)[0],'blended']=True


    print '------------------------'
    print len(good),'matched sources (',np.round(100.*float(len(good))/float(len(results)),2),'%)'
    print len(np.where(results.xmatch==0.5)[0]),'soft matches'
    print '\t',len(np.where(blended==True)[0]),' of which are blended sources'
    print len(bad),'missing sources'

    print len(np.where(contaminated==True)[0]),'contaminated sources'
    print '------------------------'
    results.to_pickle(open('results_probabilities.p','wb'))
    pickle.dump(probs,open('probs.p','wb'))
    pickle.dump(mags,open('mags.p','wb'))
    print 'Saved to ','results_probabilities.p'
    return 