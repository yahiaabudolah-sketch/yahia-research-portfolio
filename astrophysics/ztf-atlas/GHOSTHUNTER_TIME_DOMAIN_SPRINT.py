# -*- coding: utf-8 -*-
"""GHOSTHUNTER TIME-DOMAIN DISCOVERY SPRINT (2026-10-31)

A new research track: find *periodic-variable candidates* from ZTF time series.
NO detection here is a discovery until VSX, SIMBAD, Gaia, literature, and raw
images have been checked. This code is a first-stage candidate search only.

Run in Spyder with F5. Uses Python standard library, numpy, scipy, matplotlib.
No Astropy and no installation steps. Cached responses allow resumption.
Docs: https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import lombscargle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VERSION = 'time-domain-sprint-1.0'
BASE_URL = 'https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves'
OUT = Path.home() / 'Downloads' / 'GHOSTHUNTER_TIME_DOMAIN_SPRINT_RESULTS'
RAW = OUT / 'raw_ztf_csv'
# Predeclared fixed positions in Monoceros/Gemini, not selected after viewing data.
# Small local searches prevent gigantic downloads. Every position is documented.
POINTINGS = [
    ('M01', 99.2,  4.5), ('M02',101.3,  7.1), ('M03',103.1,  9.8),
    ('M04',105.0,  6.0), ('M05',107.3,  3.6), ('M06',109.4, 10.7),
    ('M07',100.9, 12.9), ('M08',103.9, 15.0), ('M09',106.7, 13.4),
    ('M10',110.8,  6.7), ('M11', 97.9, 10.3), ('M12',112.7, 14.1),
]
RADIUS_DEG = 0.055 # 3.3 arcminutes per cone
MIN_ZTF_OBS_REQUEST = 55
MAG_RANGE = '14.5 19.0'
BAD_CATFLAGS_MASK = 32768
MAX_CANDIDATE_G_CHECKS = 12
MIN_VALID_POINTS = 55
MIN_DISTINCT_NIGHTS = 30
MIN_BASELINE_DAYS = 180
MIN_SEASON_RANGE = 0.12 # robust 95%-5% r magnitude range, not proof of variability
PERIOD_MIN_DAYS = 0.17
PERIOD_MAX_DAYS = 10.0
N_FREQ_MIN = 6000
N_FREQ_MAX = 70000
OVERSAMPLE = 4.0
MAX_SOURCES_PER_FIELD = 700 # never silently discard beyond cap; report it
NETWORK_TIMEOUT_SECONDS = 50


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch_csv(field_id, ra, dec, band, radius=RADIUS_DEG):
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / f'{field_id}_{band}.csv'
    if target.exists() and target.stat().st_size > 40:
        return target.read_text(encoding='utf-8-sig', errors='replace'), 'cached'
    params = {
        'POS': f'CIRCLE {ra:.8f} {dec:.8f} {radius:.7f}',
        'BANDNAME': band,
        'NOBS_MIN': str(MIN_ZTF_OBS_REQUEST if radius==RADIUS_DEG else 10),
        'MAG': MAG_RANGE if radius==RADIUS_DEG else '13.0 20.0',
        'BAD_CATFLAGS_MASK': str(BAD_CATFLAGS_MASK),
        'FORMAT': 'CSV'
    }
    url = BASE_URL + '?' + urllib.parse.urlencode(params)
    last = None
    for retry in range(3):
        try:
            req=urllib.request.Request(url, headers={'User-Agent':'GHOSTHUNTER-research/1.0 (academic, bounded queries)'})
            with urllib.request.urlopen(req,timeout=NETWORK_TIMEOUT_SECONDS) as resp:
                b = resp.read(60*1024*1024+1)
            if len(b)>60*1024*1024: raise ValueError('Response >60 MB; skip this field safely')
            s=b.decode('utf-8-sig',errors='replace')
            if '<html' in s[:400].lower() or 'error' in s[:100].lower():
                raise ValueError('Server returned HTML/error instead of table: '+s[:130])
            if not s.strip(): raise ValueError('Blank response')
            target.write_bytes(b)
            return s, 'downloaded'
        except Exception as exc:
            last=repr(exc)
            if retry<2: time.sleep(2+retry*2)
    raise RuntimeError(f'ZTF request failed for {field_id} {band}: {last}')


def parse_ztf_csv(s):
    """Parse IRSA CSV and gracefully expose bad/empty responses."""
    lines=s.replace('\ufeff','').splitlines()
    clean=[line for line in lines if line.strip() and not line.lstrip().startswith(('\\','#'))]
    # Some IRSA services include metadata preceding the table.
    header_at=next((i for i,line in enumerate(clean) if 'oid' in line.lower().split(',') and ('mjd' in line.lower() or 'hjd' in line.lower())),None)
    if header_at is None:
        # Real no-data replies may be simple notices, not valid CSV.
        if any(('no data' in line.lower() or 'no records' in line.lower()) for line in clean): return []
        raise ValueError('Could not identify IRSA lightcurve CSV headers; first 250 chars: '+repr(s[:250]))
    rows=list(csv.DictReader(io.StringIO('\n'.join(clean[header_at:]))))
    out=[]
    for r in rows:
        d={str(k).strip().lower():v for k,v in r.items() if k is not None}
        try:
            oid=str(d.get('oid','')).strip()
            mag=float(d.get('mag','nan'))
            err=float(d.get('magerr', 'nan'))
            mjd=float(d.get('mjd', d.get('hjd','nan')))
            ra=float(d.get('ra','nan'));dec=float(d.get('dec','nan'))
            flags=int(float(d.get('catflags','0') or '0'))
            if not oid or not all(math.isfinite(x) for x in (mag,err,mjd,ra,dec)):
                continue
            if err<=0 or err>0.23 or mag<10 or mag>23 or flags & BAD_CATFLAGS_MASK:
                continue
            out.append({'oid':oid,'mjd':mjd,'mag':mag,'magerr':err,'ra':ra,'dec':dec})
        except (ValueError,TypeError): continue
    return out


def robust_amp(m):
    return float(np.percentile(m,95)-np.percentile(m,5))


def weighted_harmonic(t,y,err,period):
    omega=2*np.pi/period
    a=np.column_stack([np.ones(len(t)),np.sin(omega*t),np.cos(omega*t)])
    w=1/np.maximum(err,0.005)
    try:
        c=np.linalg.lstsq(a*w[:,None],y*w,rcond=None)[0]
        fit=a@c
        var=np.sum(((y-fit)*w)**2)/max(1,len(t)-3)
        amp=2*np.hypot(c[1],c[2])
        return float(amp),float(var)
    except Exception: return float('nan'),float('nan')


def periodic_fit(t,mag,err):
    t=np.asarray(t,float);mag=np.asarray(mag,float);err=np.asarray(err,float)
    sel=(np.isfinite(t)&np.isfinite(mag)&np.isfinite(err)&(err>0)&(err<0.23))
    t,mag,err=t[sel],mag[sel],err[sel]
    if len(t)<MIN_VALID_POINTS:return None
    span=float(t.max()-t.min())
    if span<MIN_BASELINE_DAYS or len(np.unique(np.floor(t)))<MIN_DISTINCT_NIGHTS:return None
    med=float(np.median(mag))
    # Avoid aggressive clipping: eclipses are real; reject only >3 mag isolated failures.
    keep=np.abs(mag-med)<3
    t,mag,err=t[keep],mag[keep],err[keep]
    if len(t)<MIN_VALID_POINTS:return None
    amplitude=robust_amp(mag)
    if amplitude<max(MIN_SEASON_RANGE, 3.5*float(np.median(err))):return None
    # Weighted mean centering; scipy's normalized periodogram is diagnostic only.
    w=1/np.maximum(err,0.01)**2
    avg=float(np.average(mag,weights=w))
    y=mag-avg
    n_freq=min(N_FREQ_MAX,max(N_FREQ_MIN,int(OVERSAMPLE*span*(1/PERIOD_MIN_DAYS-1/PERIOD_MAX_DAYS))))
    freq=np.linspace(1/PERIOD_MAX_DAYS, 1/PERIOD_MIN_DAYS,n_freq)
    power=lombscargle(t-t.min(),y,2*np.pi*freq,normalize=True)
    if not len(power) or not np.isfinite(power).any():return None
    ix=int(np.nanargmax(power))
    best_period=float(1/freq[ix])
    maxpower=float(power[ix])
    # Refine peak at resolution set by observational baseline.
    width=max(0.012, 5/span)
    f0=freq[ix]
    refined=np.linspace(max(1/PERIOD_MAX_DAYS,f0-width),min(1/PERIOD_MIN_DAYS,f0+width),900)
    p2=lombscargle(t-t.min(),y,2*np.pi*refined,normalize=True)
    ix2=int(np.argmax(p2))
    best_period=float(1/refined[ix2]);maxpower=float(p2[ix2])
    harmonic,redchi=weighted_harmonic(t-t.min(),mag,err,best_period)
    # 1-day sampling aliases: flag, never claim clean period automatically.
    freqbest=1/best_period
    near_alias=min(abs(freqbest-round(freqbest)),abs(freqbest-round(freqbest/0.5)*0.5))<0.035
    return {'period_days':best_period,'power':maxpower,'nobs':int(len(t)),'nights':int(len(np.unique(np.floor(t)))),'baseline_days':span,'p95_p5_mag':amplitude,'harmonic_pp_mag':harmonic,'near_day_alias':bool(near_alias),'t':t,'mag':mag,'err':err}


def field_analysis(tag, ra,dec):
    raw,how=fetch_csv(tag,ra,dec,'r')
    rows=parse_ztf_csv(raw)
    groups=defaultdict(list)
    for r in rows:groups[r['oid']].append(r)
    n_groups=len(groups)
    if n_groups>MAX_SOURCES_PER_FIELD:
        # Avoid a biased silent truncation; mark field unusable for claim-making.
        raise ValueError(f'Field has {n_groups} objects above cap {MAX_SOURCES_PER_FIELD}; reduce query radius for unbiased analysis')
    candidates=[]
    for oid,rr in groups.items():
        dat=periodic_fit([v['mjd'] for v in rr],[v['mag'] for v in rr],[v['magerr'] for v in rr])
        if dat is None: continue
        result={k:v for k,v in dat.items() if k not in ('t','mag','err')}
        result.update({'field':tag,'oid':oid,'ra':float(np.median([v['ra'] for v in rr])),'dec':float(np.median([v['dec'] for v in rr])),'band':'r','review_status':'VARIABLE_CANDIDATE_NOT_NOVELTY_CHECKED'})
        # Coarse triage ranking, not statistical significance.
        result['triage_score']=round(dat['power']*np.log1p(dat['nobs'])*min(2.0,dat['p95_p5_mag']/0.20)*(0.70 if dat['near_day_alias'] else 1),5)
        candidates.append(result)
    return candidates,{'field':tag,'ra':ra,'dec':dec,'download_status':how,'lines':len(rows),'ztf_objects':n_groups,'periodic_candidates':len(candidates),'sha256':sha256(raw.encode())}


def plot_one(candidate, index):
    tag=candidate['field'];ra=candidate['ra'];dec=candidate['dec']
    # independent-ish photometric band, not independent observations: same telescope.
    title=f"{tag}_RANK{index:02d}_{candidate['oid']}"
    rows=parse_ztf_csv((RAW/f'{tag}_r.csv').read_text(encoding='utf8',errors='replace'))
    r=[x for x in rows if x['oid']==candidate['oid']]
    period=candidate['period_days']
    fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    t=np.array([x['mjd'] for x in r]);mag=np.array([x['mag'] for x in r])
    ax[0,0].scatter(t,mag,s=8,alpha=.7,label=f'r; n={len(t)}')
    ax[0,0].set(xlabel='MJD',ylabel='r magnitude',title='Public ZTF time series');ax[0,0].invert_yaxis()
    ph=((t-t.min())/period)%1
    ax[0,1].scatter(np.r_[ph,ph+1],np.r_[mag,mag],s=8,alpha=.7)
    ax[0,1].set(xlabel=f'Phase, P={period:.6f} d',ylabel='r magnitude',title='Phase-folded r band');ax[0,1].invert_yaxis()
    gmeta='g band: not available'
    try:
        raw,how=fetch_csv(f'{tag}_G_{index:02d}',ra,dec,'g',radius=0.0015)
        groot=parse_ztf_csv(raw)
        g=[x for x in groot if abs(x['ra']-ra)*math.cos(math.radians(dec))<0.001 and abs(x['dec']-dec)<0.001]
        if g:
            tg=np.array([x['mjd'] for x in g]);mg=np.array([x['mag'] for x in g]);pg=((tg-t.min())/period)%1
            ax[1,0].scatter(tg,mg,s=9,alpha=.75,color='C2')
            ax[1,0].set(xlabel='MJD',ylabel='g magnitude',title=f'g-band time series; n={len(g)}');ax[1,0].invert_yaxis()
            ax[1,1].scatter(np.r_[pg,pg+1],np.r_[mg,mg],s=9,color='C2',alpha=.75)
            ax[1,1].set(xlabel='Phase',ylabel='g magnitude',title='Independent filter, same survey');ax[1,1].invert_yaxis()
            gmeta=f'g band n={len(g)} amplitude p95-p5={robust_amp(mg):.3f}'
        else:
            ax[1,0].text(.1,.5,'No nearby g lightcurve',transform=ax[1,0].transAxes)
            ax[1,1].axis('off')
    except Exception as exc:
        gmeta='g query failed: '+str(exc)[:160]
        ax[1,0].text(.03,.4,gmeta[:60],transform=ax[1,0].transAxes);ax[1,1].axis('off')
    fig.suptitle(f'PERIODIC CANDIDATE, NOT DISCOVERY  |  {title}\nRA={ra:.6f}, Dec={dec:+.6f}; {gmeta}',fontsize=11)
    path=OUT/(title+'.png');fig.savefig(path,dpi=135);plt.close(fig)
    return {'title':title,'plot':str(path),'g_check':gmeta}


def write_csv(path,rows):
    if not rows:
        path.write_text('field,oid,ra,dec,band,period_days,power,nobs,nights,baseline_days,p95_p5_mag,triage_score,review_status\n',encoding='utf8')
        return
    cols=list(rows[0])
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)


def offline_selftest():
    sample='oid,mjd,mag,magerr,ra,dec,catflags\n123,60000,16.00,0.04,100.0,12.0,0\n123,60001,16.3,0.03,100.0,12.0,32768\n'
    rr=parse_ztf_csv(sample)
    assert len(rr)==1 and rr[0]['oid']=='123'
    rng=np.random.default_rng(67893);t=np.sort(rng.uniform(0,1300,125));p=0.612345
    y=17+0.22*np.sin(2*np.pi*t/p)+rng.normal(0,.03,len(t))
    fit=periodic_fit(t+59000,y,np.full(len(t),.03))
    assert fit is not None and fit['power']>.20
    # 1-day aliases can outrank true period depending on sampling; synthetic random sample should resolve it.
    assert abs(fit['period_days']-p)<.002,(fit['period_days'],p)
    assert periodic_fit(t[:25]+59000,y[:25],np.full(25,.03)) is None
    return True


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print('='*78)
    print('GHOSTHUNTER | TIME-DOMAIN DISCOVERY SPRINT | NEW RESEARCH TRACK')
    print('ZTF archival periodic-variable search; NEVER an automatic novelty claim')
    print('Output:',OUT)
    print('='*78)
    print('OFFLINE TEST:',offline_selftest())
    settings={
        'region':'12 fixed cone positions in Monoceros/Gemini (not asserted unsearched)',
        'fields':POINTINGS,'radius_deg':RADIUS_DEG,'min_obs':MIN_VALID_POINTS,
        'mag_range':MAG_RANGE,'n_freq':'adaptive', 'oversample':OVERSAMPLE,'min_max_freq_samples':[N_FREQ_MIN,N_FREQ_MAX], 'frequency_search_warning':'Candidate periodograms only; aliases, harmonics, and cap-limited period resolution may still produce errors','period_days_range':[PERIOD_MIN_DAYS,PERIOD_MAX_DAYS],
        'references':['https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html',
            'https://vsx.aavso.org/index.php?view=about.notice'],
        'scientific_limitation':'No control sample, no calibrated FAP, no catalogue novelty checks, source blending and cadence aliases possible'
    }
    progress=[];candidates=[]
    for i,(tag,ra,dec) in enumerate(POINTINGS,1):
        print(f'FIELD {i:02d}/{len(POINTINGS)} {tag} (RA={ra}, Dec={dec})')
        try:
            cc,meta=field_analysis(tag,ra,dec)
            progress.append(dict(meta,status='OK'));candidates.extend(cc)
            print('  ZTF objects:',meta['ztf_objects'],'candidate variables:',len(cc))
        except Exception as e:
            progress.append(dict(field=tag,status='ERROR',error=repr(e)))
            print('  FAILED:',repr(e))
        (OUT/'FIELD_PROGRESS.json').write_text(json.dumps(progress,indent=2),encoding='utf8')
    candidates.sort(key=lambda x:x['triage_score'],reverse=True)
    write_csv(OUT/'ALL_PERIODIC_CANDIDATES.csv',candidates)
    print('\nCANDIDATES:',len(candidates))
    if not any(x.get('status')=='OK' for x in progress):
        print('NO SUCCESSFUL LIVE FIELD QUERIES. This is a data-access failure, NOT a non-discovery.')
    photos=[]
    for i,c in enumerate(candidates[:MAX_CANDIDATE_G_CHECKS],1):
        print(f"  REVIEW {i}/{min(len(candidates),MAX_CANDIDATE_G_CHECKS)} oid={c['oid']}, P={c['period_days']:.6f}d, power={c['power']:.3f}")
        try:photos.append(plot_one(c,i))
        except Exception as exc:photos.append({'oid':c['oid'],'status':'PLOT_FAILED','error':repr(exc)})
    summary={
        'version':VERSION,'utc':datetime.now(timezone.utc).isoformat(),
        'settings':settings,'fields':progress,'candidate_count':len(candidates),'plots':photos,
        'top_candidates':candidates[:25],
        'outcome':'PERIODIC_VARIABLE_CANDIDATES_ONLY - NO NEW OBJECT CONFIRMED',
        'to_claim_novelty':'Confirm period in multiple datasets; crossmatch VSX, Gaia DR3 variable, SIMBAD and 2026 literature; verify ZTF image/blends; peer review',
    }
    (OUT/'DISCOVERY_SPRINT_REPORT.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8')
    print('\nDONE:',OUT)
    print('Send DISCOVERY_SPRINT_REPORT.json, ALL_PERIODIC_CANDIDATES.csv, and top candidate plots.')
    print('NO discovery claimed. Check aliases/known catalogues before declaring novelty.')

if __name__=='__main__':main()
