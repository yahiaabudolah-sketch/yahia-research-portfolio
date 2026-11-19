"""GHOSTHUNTER v10.1 — post-hoc source-identification audit (Spyder: F5).

No detections are recomputed; NO detector changes, NO internet requests.
Uses cached v10 DR11 FITS images + candidate CSV files in Downloads.
Inspects ALL 12 predeclared SMUDGes candidates, not just four hand-picked cases.

Measures source profiles at the PUBLISHED CATALOGUE position and the nearest
v10 detected peak, alongside local compact, star-like sources. Reports
5/10/20-arcsec matching sensitivity and warns about potential blends.

All results are diagnostic only. Comparators are NOT calibrated instrumental
PSFs; normalized profiles are NOT photometric fits. v10 success figures remain
unchanged; this audit does not create new independent validation.

Requirements: GHOSTHUNTER_V10_CATALOGUE_VALIDATION.py in same directory,
with original GHOSTHUNTER_V10_CATALOGUE_VALIDATION_RESULTS intact. Existing
NumPy, SciPy, Matplotlib only; no Astropy or additional installations.
"""
import csv, importlib.util, json, math, traceback, hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parent
V10=ROOT/'GHOSTHUNTER_V10_CATALOGUE_VALIDATION.py'
BASE=ROOT/'GHOSTHUNTER_V10_CATALOGUE_VALIDATION_RESULTS'
OUT=ROOT/'GHOSTHUNTER_V10_1_SOURCE_AUDIT_RESULTS'
EDGES=np.array([0,3,6,9,12,16,20,26,35,40,55],dtype=float) # pixel radii
MIN_STAR_COMPARATORS=4
MAX_STAR_COMPARATORS=24
REF_STAR_TARGET_EXCLUSION=70


def load_v10():
    if not V10.exists(): raise FileNotFoundError('Missing '+str(V10))
    sp=importlib.util.spec_from_file_location('ghosthunter_v10_readonly',str(V10))
    gh=importlib.util.module_from_spec(sp)
    sp.loader.exec_module(gh)
    return gh


def finite(v):
    try:
        a=float(v)
        return round(a,7) if np.isfinite(a) else None
    except (TypeError,ValueError):return None


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def annular_profile(img, centre, neighbours=()):
    """Annular medians around a fixed centre. Remove circles around other
    compact peaks, then subtract 40-55px sky median. No PSF correction.
    """
    x,y=map(float,centre)
    H,W=img.shape
    xmin=max(0,int(x)-57);xmax=min(W,int(x)+58)
    ymin=max(0,int(y)-57);ymax=min(H,int(y)+58)
    Y,X=np.ogrid[ymin:ymax,xmin:xmax]
    rr=np.hypot(X-x,Y-y)
    data=np.asarray(img[ymin:ymax,xmin:xmax],float)
    good=np.isfinite(data)
    for xx,yy in neighbours:
        d=math.hypot(x-xx,y-yy)
        if 8 < d < 62: good &= (X-xx)**2+(Y-yy)**2 >=36
    ann_sky=(rr>=40)&(rr<55)&good
    if ann_sky.sum()<140:return None
    sky=float(np.median(data[ann_sky]))
    meds=[];coverage=[]
    for lo,hi in zip(EDGES[:-1],EDGES[1:]):
        ring=(rr>=lo)&(rr<hi)&np.isfinite(data)
        kept=ring&good
        coverage.append(finite(kept.sum()/max(1,ring.sum())))
        meds.append(float(np.median(data[kept])-sky) if kept.sum()>=max(6,.3*ring.sum()) else np.nan)
    meds=np.array(meds)
    if not np.isfinite(meds[0]) or meds[0]<=0:return {'usable':False,'sky':finite(sky),'coverage':coverage,'reason':'nonpositive/invalid core flux'}
    norm=meds/meds[0]
    return {'usable':True,'sky':finite(sky),'core_flux':finite(meds[0]),
            'normalised':[finite(z) for z in norm], 'coverage':coverage,
            'outer_to_core':finite(np.nanmedian(norm[3:6])),
            'outer_raw_median':finite(np.nanmedian(meds[3:6]))}


def star_like_peaks(img, truth, gh):
    """PSF proxies only: intentionally excludes very bright/saturated objects
    and crowded positions; they are NOT guaranteed to be stars.
    """
    sky,_=gh.fit_sky_plane(img)
    residual=np.nan_to_num(img-sky,nan=0.)
    sig=gh.robust_std(residual)
    hp=gaussian_filter(residual,1)-gaussian_filter(residual,4)
    score=hp/max(sig,1e-12)
    ispeak=(score>=7)&(score==maximum_filter(score,size=9))
    py,px=np.nonzero(ispeak)
    if len(px)==0:return [],[],{'n_candidates':0,'n_comparators':0}
    points=np.column_stack((px,py)).astype(float)
    if len(points)>1:
        dist,_=cKDTree(points).query(points,k=2)
        nearest=dist[:,1]
    else:nearest=np.array([1e9])
    order=np.argsort(score[py,px])[::-1]
    stars=[]
    for i in order:
        x,y=int(px[i]),int(py[i])
        if min(x,y,img.shape[1]-1-x,img.shape[0]-1-y)<60:continue
        if math.hypot(x-truth[0],y-truth[1])<REF_STAR_TARGET_EXCLUSION:continue
        if nearest[i]<24:continue
        if score[y,x]>500:continue
        pr=annular_profile(img,(x,y))
        if pr is None or not pr.get('usable') or pr['core_flux']<5*sig:continue
        stars.append((x,y))
        if len(stars)>=MAX_STAR_COMPARATORS:break
    allpeaks=list(map(tuple,points))
    return stars,allpeaks,{'n_candidates':len(points),'n_comparators':len(stars),'noise_robust':finite(sig)}


def read_candidates(folder):
    file=folder/'POSITIVE_PEAKS.csv'
    if not file.exists():raise FileNotFoundError('Missing '+str(file))
    with file.open('r',encoding='utf-8',newline='') as f:
        rs=list(csv.DictReader(f))
    if not rs:raise ValueError('No candidate rows in '+str(file))
    for p in rs:
        for k in ('rank','x','y','ra_deg','dec_deg','score','scale_arcsec'):
            p[k]=float(p[k])
    return rs


def matched_candidate(cands,truth,gh):
    dlist=[(gh.sep_arcsec(p['ra_deg'],p['dec_deg'],*truth),p) for p in cands]
    return sorted(dlist,key=lambda z:z[0])[0] if dlist else (float('inf'),None)


def compact_profile_stats(img, centre, allpeaks, stars):
    exclude=[(x,y) for x,y in allpeaks if math.hypot(x-centre[0],y-centre[1])>8]
    target=annular_profile(img,centre,exclude)
    sig=1.4826*np.median(np.abs(img[np.isfinite(img)]-np.median(img[np.isfinite(img)])))
    starr=[]
    for s in stars:
        prox=[p for p in allpeaks if math.hypot(p[0]-s[0],p[1]-s[1])>8]
        profile=annular_profile(img,s,prox)
        if (profile and profile.get('usable') and profile['core_flux']>3*sig
                and all(k is not None for k in profile['normalised'][:7])):
            starr.append(profile['normalised'])
    out={'target_profile':target,'star_n':len(starr)}
    if len(starr)>=MIN_STAR_COMPARATORS:
        stack=np.asarray(starr,dtype=float)
        star_ratio=np.nanmedian(stack[:,3:6],axis=1)
        out.update(star_median_profile=[finite(v) for v in np.nanmedian(stack,axis=0)],
                   star_p10_profile=[finite(v) for v in np.nanpercentile(stack,10,axis=0)],
                   star_p90_profile=[finite(v) for v in np.nanpercentile(stack,90,axis=0)],
                   star_90th_outer_ratio=finite(np.nanpercentile(star_ratio,90)))
    return out


def show_panel(a,dat,truth,candidate,px,field):
    tx,ty=truth
    h,w=dat.shape
    x1=max(0,int(tx)-74);x2=min(w,int(tx)+75)
    y1=max(0,int(ty)-74);y2=min(h,int(ty)+75)
    patch=dat[y1:y2,x1:x2]
    q=np.percentile(patch[np.isfinite(patch)],[2,99.5])
    vmin,vmax=float(q[0]),float(q[1]);vmax=max(vmax,vmin+1e-9)
    a.imshow(dat,origin='lower',cmap='gray',vmin=vmin,vmax=vmax,interpolation='nearest')
    a.set_xlim(x1,x2);a.set_ylim(y1,y2)
    a.scatter([tx],[ty],marker='+',c='cyan',s=130,linewidth=2,label='catalogue centre')
    a.add_patch(Circle((tx,ty),5/px,fill=False,ec='cyan',lw=1.1,ls='--'))
    a.add_patch(Circle((tx,ty),20/px,fill=False,ec='gold',lw=1,ls=':'))
    if candidate:
        a.scatter([candidate['x']],[candidate['y']],s=90,facecolors='none',edgecolors='#fc8d28',lw=1.8,label=f"nearest rank {int(candidate['rank'])}")
    a.set_title(field+' | r-band | 5″/20″ circles',fontsize=10)
    a.legend(loc='upper right',fontsize=8)


def plot_field(filepath,field,bands,truth,match,px,truth_prof,match_prof):
    xs=(EDGES[:-1]+EDGES[1:])/2*px
    fig=plt.figure(figsize=(15,9.0),constrained_layout=True)
    spec=fig.add_gridspec(2,3,height_ratios=[1.1,1])
    ax_image=fig.add_subplot(spec[0,:2])
    ax_text=fig.add_subplot(spec[0,2])
    show_panel(ax_image,bands['r'],truth,match,px,field)
    meta=[]
    if match:
        d=math.hypot(match['x']-truth[0],match['y']-truth[1])*px
        meta.append(f"Nearest rank {int(match['rank'])}: {d:.2f} arcsec")
    else:
        meta.append('No detected peak within 20 arcsec')
    ax_text.axis('off')
    ax_text.text(.03,.92,'\n'.join(meta),fontsize=13,va='top',fontweight='bold')
    ax_text.text(.03,.73,'Blue = catalogue centre\nOrange = nearest detected peak\nGrey = local star-like profile proxies\n\nDiagnostic only: no calibrated PSF,\nno verified star identities,\nno completeness/purity estimate.',fontsize=10,va='top')
    for i,b in enumerate('grz'):
        a=fig.add_subplot(spec[1,i])
        dat=truth_prof[b]
        if dat.get('star_median_profile') is not None:
            m=np.array(dat['star_median_profile'],float)
            p10=np.array(dat['star_p10_profile'],float)
            p90=np.array(dat['star_p90_profile'],float)
            a.plot(xs,m,color='gray',label='star-like median')
            a.fill_between(xs,p10,p90,color='gray',alpha=.2,label='star-like 10–90%')
        p=dat.get('target_profile')
        if p and p.get('usable'):
            a.plot(xs,np.array([np.nan if v is None else v for v in p['normalised']]),'o-',label='catalogue centre')
        mp=match_prof.get(b)
        if mp and mp.get('usable'):
            a.plot(xs,np.array([np.nan if v is None else v for v in mp['normalised']]),'--',color='#ea7d1a',label='detected peak')
        a.axhline(0,c='gray',linewidth=.7)
        a.set_title(b+'-band profile')
        a.set_xlabel('Radius (arcseconds)')
        if i==0:a.set_ylabel('Median / central 0–3px brightness')
        a.set_ylim(bottom=-.3,top=1.3)
        a.legend(loc='upper right',fontsize=8)
    fig.suptitle('GHOSTHUNTER v10.1 | source-association audit | POST-HOC ONLY',fontsize=13)
    fig.savefig(filepath,dpi=125);plt.close(fig)


def run_one(gh,row,original_manifest):
    field=row['field'];folder=BASE/field
    fits=folder/'DR11_grz.fits'
    result={'field':field,'status':'FAILED','catalogue_mu0g':row['mu0g'],'catalogue_re_arcsec':row['re_arcsec'],
            'original_evaluation':row.get('evaluation'),'detector_frozen':True}
    if not fits.exists():return {**result,'error':'Missing cached FITS: '+str(fits)}
    bands,h=gh.read_fits(fits)
    if not all(b in bands for b in 'grz'):return {**result,'error':'Missing g/r/z bands'}
    tx,ty=gh.world_to_pixel(h,*row['truth_coordinates_evaluation_only'])
    truth_xy=(tx,ty);px=gh.pixel_scale(h)
    cands=read_candidates(folder)
    nearest,p=matched_candidate(cands,row['truth_coordinates_evaluation_only'],gh)
    source_match=p if nearest<=20 else None
    stars,allpeaks,diag=star_like_peaks(bands['r'],truth_xy,gh)
    truth_profiles={};matched_profiles={}
    for band in 'grz':
        truth_profiles[band]=compact_profile_stats(bands[band],truth_xy,allpeaks,stars)
        if source_match:
            nbrs=[q for q in allpeaks if math.hypot(q[0]-p['x'],q[1]-p['y'])>8]
            matched_profiles[band]=annular_profile(bands[band],(p['x'],p['y']),nbrs)
    profiles_ratios={}
    for b in 'grz':
        ts=truth_profiles[b]
        pt=ts.get('target_profile') or {}
        pc=matched_profiles.get(b) or {}
        profiles_ratios[b]={'catalogue_centre':pt.get('outer_to_core'),
                            'detected_centre':pc.get('outer_to_core'),
                            'star_90th':ts.get('star_90th_outer_ratio'),
                            'n_star_like':ts.get('star_n'),
                            'catalogue_core_valid':bool(pt.get('usable')),
                            'detected_core_valid':bool(pc.get('usable'))}
    outfolder=OUT/field;outfolder.mkdir(exist_ok=True,parents=True)
    fig=outfolder/'SOURCE_ASSOCIATION.png'
    plot_field(fig,field,bands,truth_xy,source_match,px,truth_profiles,matched_profiles)
    # 5" is a second, post-hoc, more conservative match criterion, NOT v10 scoring.
    result.update(status='OK',v10_positive_count=len(cands),
        cached_fits_sha256=sha(fits),pixel_scale_arcsec=finite(px),
        truth_xy=[finite(tx),finite(ty)],
        nearest_peak_distance_arcsec=finite(nearest),
        nearest_peak_rank=int(p['rank']) if p else None,
        nearest_peak_flags=p.get('review_flags') if p else None,
        nearest_peak_score=finite(p['score']) if p else None,
        match_5arcsec=bool(nearest<=5),
        top10_5arcsec=bool(nearest<=5 and p and p['rank']<=10),
        match_10arcsec=bool(nearest<=10),
        top10_10arcsec=bool(nearest<=10 and p and p['rank']<=10),
        match_20arcsec=bool(nearest<=20),
        top10_20arcsec=bool(nearest<=20 and p and p['rank']<=10),
        comparison_stars=diag,profile_ratios=profiles_ratios,
        plot=str(fig),limitations='POSTHOC source profiles; not proof of galaxy association')
    (outfolder/'MEASUREMENTS.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def run_self_test():
    y,x=np.indices((130,130))
    star=np.exp(-((x-65.)**2+(y-65.)**2)/(2*1.5**2))
    gal=np.exp(-np.hypot(x-65.,y-65.)/12.)
    a=annular_profile(star,(65,65));b=annular_profile(gal,(65,65))
    assert a['usable'] and b['usable']
    assert b['outer_to_core']>a['outer_to_core']+.10
    assert len(b['normalised'])==len(EDGES)-1
    return 'PASS — symmetric synthetic compact/diffuse profile test'


def main():
    print('='*75)
    print('GHOSTHUNTER v10.1 | SOURCE-ASSOCIATION AUDIT | FROZEN v10 NOT RERUN')
    print('No downloads; no changes to any existing source files or detection ranks')
    print('Output:',OUT)
    print('='*75)
    print('SELF-TEST:',run_self_test())
    gh=load_v10()
    manifestfile=BASE/'VALIDATION_MANIFEST.json'
    if not manifestfile.exists():raise FileNotFoundError('Missing v10 manifest: '+str(manifestfile))
    manifest=json.loads(manifestfile.read_text('utf-8'))
    controls=[r for r in manifest['fields'] if r['kind']=='catalogue_control']
    if len(controls)!=12:raise ValueError('Expected the original 12 predeclared controls; found '+str(len(controls)))
    OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    for i,row in enumerate(controls,1):
        print('\n',i,'/ 12:',row['field'],flush=True)
        try:res=run_one(gh,row,manifest)
        except Exception as e:
            res={'field':row['field'],'status':'FAILED','error':type(e).__name__+': '+str(e)}
        results.append(res)
        print('  ',res['status'],'nearest=',res.get('nearest_peak_distance_arcsec'),
              'rank=',res.get('nearest_peak_rank'),'5arcsec=',res.get('match_5arcsec'))
        if res['status']!='OK':print('  Reason:',res['error'])
    good=[r for r in results if r['status']=='OK']
    summary={k:sum(bool(r.get(k)) for r in good) for k in
             ('match_5arcsec','top10_5arcsec','match_10arcsec','top10_10arcsec',
              'match_20arcsec','top10_20arcsec')}
    summary.update(evaluated=len(good),planned=len(controls),
                   original_v10_summary_unchanged=manifest['summary'],
                   status='POSTHOC_DIAGNOSTIC_NOT_INDEPENDENT_VALIDATION')
    data={'version':'10.1 post-hoc source association','original_manifest_sha256':sha(manifestfile),
          'v10_source_sha256':sha(V10),'summary':summary,'results':results,
          'limitations':[
              'Do not interpret normalized annular source profiles as a galaxy classification.',
              'Selected compact comparators have no Gaia-based star verification and may be galaxies or blends.',
              'Bright neighbour wings and sky subtraction may bias catalogue-centred and detected profiles.',
              'New 5/10 arcsec thresholds are sensitivity diagnostics, not replacements for frozen v10 evaluation.',
              'This analysis uses published catalogue coordinates and is not blind or independent.',
              'Source association at <5 arcsec does not confirm a published catalogue entry physically.'
          ]}
    output=OUT/'SOURCE_ASSOCIATION_REPORT.json'
    output.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    cols=('field','status','catalogue_mu0g','catalogue_re_arcsec','nearest_peak_rank',
          'nearest_peak_distance_arcsec','nearest_peak_flags','match_5arcsec','top10_5arcsec','match_20arcsec','top10_20arcsec')
    with (OUT/'MATCH_SENSITIVITY.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(results)
    print('\nFINISHED: ',len(good),' / 12 audited')
    print('Posthoc positional sensitivity:',summary)
    print('REPORT:',output)
    print('Upload SOURCE_ASSOCIATION_REPORT.json and the SOURCE_ASSOCIATION.png figures')
    print('from CONTROL_07, CONTROL_09, CONTROL_10, and CONTROL_12.')
    print('v10 original scores and validation manifest are unchanged.')

if __name__=='__main__':main()
