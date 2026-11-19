"""GHOSTHUNTER v10.0 — FROZEN, CATALOGUE-BASED VALIDATION PILOT.

One file for Spyder, press F5. No astropy. Requirements: numpy scipy matplotlib.
Downloads the public SMUDGes III (Zaritsky+ 2022, J/ApJS/261/11/table2)
position catalogue, predeclares 12 untouched candidates across 3 g-band
surface-brightness bins. Then requests public Legacy Surveys DR11 grz cutouts,
runs the unchanged v8.3 detector and evaluates positions only AFTER ranking.

This is a TARGETED recovery pilot, not a blind sky survey. Original SMUDGes
sources are photometric CANDIDATES, not uniformly confirmed satellites.
No depth maps, calibrated PSF, or labelling of unrelated peaks exists.
Recovery on this sample is NOT complete/pure survey performance, much less
proof of novelty; use it to detect failures and plan a statistically adequate
validation, not to tweak parameters on these targets.

Citation: Zaritsky et al., ApJS 261, 11 (2022), catalogue DOI:
https://doi.org/10.26093/cds/vizier.22610011
See https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJS/261/11
DR11 cutout API: https://www.legacysurvey.org/dr11/description/
"""
# Frozen pipeline constants inherited verbatim from v9/v8.3
DR_RELEASE='ls-dr11'
PIXELS=512
ARCSEC_PER_PIXEL=0.85
DETECTION_SCALES_ARCSEC=(5,9,15,24)
MATCHED_FILTER_THRESHOLD=4.8
MAX_CANDIDATES_PER_FIELD=15
NETWORK_TIMEOUT=40

from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, html, io, json, math, sys, time
import urllib.parse, urllib.request
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import (gaussian_filter, maximum_filter, label,
                           binary_dilation, distance_transform_edt,
                           median_filter)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
OUT = HERE / "GHOSTHUNTER_V8_RESULTS"
SURVEY_BASE = "https://www.legacysurvey.org/viewer/fits-cutout"
TAP = "https://datalab.noirlab.edu/tap/sync"
VERSION = "8.0"
PIXEL_FMT = "Legacy Survey coadd pixels; flux units assumed nanomaggy/pixel for provisional injections"


def now(): return datetime.now(timezone.utc).isoformat()

def finite_num(v, default=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (ValueError, TypeError):
        return default


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def fits_val(card):
    value = card[10:].strip()
    if value.startswith("'"):
        return value[1:].split("'")[0].strip()
    value = value.split("/")[0].strip()
    if value == "T": return True
    if value == "F": return False
    try:
        return float(value.replace("D", "E")) if any(c in value for c in ".eED") else int(value)
    except ValueError:
        return value


def read_fits(path):
    """Uncompressed standard FITS primary/extension image, not tables or .fz.
    Assumes FITS axis NAXIS3 enumerates filter planes and BANDS keyword labels.
    Rejects unsupported WCS later rather than fabricating celestial coordinates.
    """
    types = {8: "u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}
    with open(path, "rb") as f:
        for _ in range(12):
            header = {}
            while True:
                b = f.read(2880)
                if len(b) != 2880: raise ValueError("Truncated FITS header")
                end = False
                for j in range(0, 2880, 80):
                    c = b[j:j+80].decode("ascii", "replace")
                    k = c[:8].strip()
                    if k == "END": end = True; break
                    if c[8:10] == "= ": header[k] = fits_val(c)
                if end: break
            bit = int(header.get("BITPIX", 8))
            dims = [int(header.get("NAXIS" + str(i), 0)) for i in range(1, int(header.get("NAXIS", 0))+1)]
            n = math.prod(dims) if dims else 0
            if bit not in types: raise ValueError("Unsupported FITS BITPIX " + str(bit))
            total_bytes = (n + int(header.get("PCOUNT", 0))) * int(header.get("GCOUNT", 1)) * (abs(bit)//8)
            if len(dims) in (2, 3) and 0 < n < 6000000 and str(header.get("XTENSION", "IMAGE")).strip() == "IMAGE":
                raw = f.read(n*abs(bit)//8)
                if len(raw) != n*abs(bit)//8: raise ValueError("FITS pixels truncated")
                arr = np.frombuffer(raw, dtype=np.dtype(types[bit])).reshape(tuple(reversed(dims))).astype(np.float64)
                arr = arr*float(header.get("BSCALE", 1)) + float(header.get("BZERO", 0))
                if arr.ndim == 2: return {"r": arr}, header
                names = str(header.get("BANDS", "grz")).replace(" ", "").lower()
                if len(names) != arr.shape[0] or len(set(names)) != len(names):
                    raise ValueError("Unknown FITS band mapping; cannot trust multi-band order")
                return {names[i]: arr[i] for i in range(arr.shape[0])}, header
            f.seek((total_bytes+2879)//2880*2880, 1)
    raise ValueError("No standard image HDU")


def cd_matrix(h):
    if "CD1_1" in h and "CD2_2" in h:
        return np.array([[float(h["CD1_1"]), float(h.get("CD1_2", 0))],
                         [float(h.get("CD2_1", 0)), float(h["CD2_2"])]])
    if "CDELT1" in h and "CDELT2" in h:
        # FITS convention: CD_i_j = CDELT_i * PC_i_j
        return np.array([[float(h["CDELT1"])*float(h.get("PC1_1", 1)),float(h["CDELT1"])*float(h.get("PC1_2", 0))],
                         [float(h["CDELT2"])*float(h.get("PC2_1", 0)),float(h["CDELT2"])*float(h.get("PC2_2", 1))]])
    raise ValueError("Unsupported FITS WCS transformation (need CD or PC+CDELT)")


def pixel_to_world(h, x, y):
    if not all(k in h for k in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2")):
        raise ValueError("WCS centre missing")
    if "TAN" not in str(h.get("CTYPE1", "")) or "TAN" not in str(h.get("CTYPE2", "")):
        raise ValueError("WCS is not TAN projection")
    dx = np.array([x+1.0-float(h["CRPIX1"]), y+1.0-float(h["CRPIX2"])])
    xi_deg, eta_deg = cd_matrix(h) @ dx
    xi, eta = math.radians(xi_deg), math.radians(eta_deg)
    a0, d0 = math.radians(float(h["CRVAL1"])), math.radians(float(h["CRVAL2"]))
    ra = math.degrees(a0+math.atan2(xi, math.cos(d0)-eta*math.sin(d0))) % 360
    dec = math.degrees(math.atan2(math.sin(d0)+eta*math.cos(d0),
                                  math.hypot(xi, math.cos(d0)-eta*math.sin(d0))))
    return ra,dec


def pixel_scale(h):
    cd = cd_matrix(h)
    return 3600*math.sqrt(abs(float(np.linalg.det(cd))))


def sep_arcsec(ra1, dec1, ra2, dec2):
    x = math.radians((ra1-ra2+180)%360-180)
    d1, d2 = math.radians(dec1), math.radians(dec2)
    aa = math.sin((d1-d2)/2)**2 + math.cos(d1)*math.cos(d2)*math.sin(x/2)**2
    return 3600*math.degrees(2*math.asin(min(1.0, math.sqrt(max(0, aa)))))


def robust_std(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    if a.size < 50: return float("nan")
    m = np.median(a)
    return max(1.4826*float(np.median(abs(a-m))), 1e-12)


_GAUSS_KERNELS = {}
def fast_gaussian(a, sigma):
    """Zero-padded FFT Gaussian smoothing for broad matched filters.
    O(N log N), unlike broad direct separable convolutions that scale as O(sigma*N).
    Mask support is explicitly normalized in the caller.
    """
    sigma=float(sigma)
    if sigma<5:
        return gaussian_filter(a,sigma,mode="constant",cval=0)
    key=round(sigma,4)
    if key not in _GAUSS_KERNELS:
        rad=int(min(4*sigma, 300))
        gy,gx=np.mgrid[-rad:rad+1,-rad:rad+1]
        k=np.exp(-(gx**2+gy**2)/(2*sigma**2))
        k/=k.sum()
        _GAUSS_KERNELS[key]=k
    return fftconvolve(a,_GAUSS_KERNELS[key],mode="same")


def valid_fits(p):
    try:
        with open(p, "rb") as f: return p.stat().st_size > 3000 and f.read(8)==b"SIMPLE  "
    except (OSError, ValueError): return False


def obtain_fits(folder, name, ra, dec):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "dr11_grz.fits"
    q = dict(ra=f"{ra:.8f}", dec=f"{dec:.8f}", size=PIXELS,
             pixscale=ARCSEC_PER_PIXEL, layer=DR_RELEASE, bands="grz")
    url = SURVEY_BASE + "?" + urllib.parse.urlencode(q)
    # Reuse ONLY exact same requested centre/scale/layer from local folder.
    meta_path = folder/"source_info.json"
    if valid_fits(path) and meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            if m.get("url")==url:
                data, head = read_fits(path)
                if data and all(v.shape==(PIXELS, PIXELS) for v in data.values()):
                    return path, url, "cache"
        except Exception:
            pass
    req = urllib.request.Request(url, headers={"User-Agent":"GhostHunterResearchPilot/8.0 (public educational project)"})
    with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT) as r:
        raw = r.read(18_000_001)
    if len(raw)>18_000_000 or not raw.startswith(b"SIMPLE  "):
        raise ValueError("No valid FITS from survey cutout service (check footprint/service)")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(raw)
    try:
        data, head = read_fits(tmp)
        if not all(v.shape==(PIXELS,PIXELS) for v in data.values()):
            raise ValueError("Unexpected FITS shape")
        _ = pixel_to_world(head, PIXELS/2, PIXELS/2)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)
    meta_path.write_text(json.dumps({"url":url,"sha256":sha256(path),"download_utc":now()},indent=2),encoding="utf-8")
    return path, url, "download"


def fit_sky_plane(a):
    """Robust 2D constant+linear fit; avoid overly local sky subtraction.
    Initial bright compact sources are clipped; not a replacement for raw-CCD
    sky modelling. Fitting over only one small cutout is still dangerous.
    """
    y,x = np.indices(a.shape)
    valid = np.isfinite(a)
    med = float(np.median(a[valid]))
    rs = robust_std(a[valid])
    valid &= (a < med + 2.5*rs) & (a > med-3*rs)
    # subsample to keep FITS tile fits fast
    sel = valid[::5,::5]
    xx = (x[::5,::5][sel]-(a.shape[1]-1)/2)/a.shape[1]
    yy = (y[::5,::5][sel]-(a.shape[0]-1)/2)/a.shape[0]
    zz = a[::5,::5][sel]
    if zz.size<200: return np.full_like(a,med), "constant_no_fit"
    M = np.stack([np.ones(len(xx)),xx,yy],axis=1)
    coeff = np.linalg.lstsq(M,zz,rcond=None)[0]
    for _ in range(3):
        err = zz - M@coeff
        mad = robust_std(err)
        mask = abs(err-np.median(err)) < 2.5*mad
        if mask.sum()<150: break
        coeff = np.linalg.lstsq(M[mask],zz[mask],rcond=None)[0]
    plane = coeff[0]+coeff[1]*(x-(a.shape[1]-1)/2)/a.shape[1]+coeff[2]*(y-(a.shape[0]-1)/2)/a.shape[0]
    return plane, "robust_plane_possible_LSB_self_subtraction"


def prep_bands(bands):
    # Never assume a band exists; DR11 cutout with grz explicitly requested.
    if not all(b in bands for b in "grz"):
        raise ValueError("g/r/z not all provided; do not fabricate band coincidence")
    shape = bands["r"].shape
    if any(v.shape!=shape for v in bands.values()):
        raise ValueError("Nonidentical band shapes/WCS; do not coadd unregistered bands")
    work={};sig={}; skytypes={}; valid=np.ones(shape,bool)
    for b in "grz":
        arr = bands[b].astype(float)
        valid &= np.isfinite(arr)
        sky, typ = fit_sky_plane(arr)
        residual = arr - sky
        s = robust_std(residual)
        if not math.isfinite(s) or s<=0: raise ValueError("Degenerate noise in "+b)
        work[b] = residual
        sig[b] = s
        skytypes[b] = typ
    return work,sig,valid,skytypes


def compact_mask(work, sig, valid, pxscale):
    """Conservative compact-object *proxy* mask, NOT a PSF-wing fit.
    Distinguish sharp compact peaks from broad extended smooth structure using
    central 1.0 pixel minus a 4 pixel smoothing, then cap growth near edge.
    """
    r = work["r"] / sig["r"]
    sharp = gaussian_filter(r, 1.0) - gaussian_filter(r, 4.0)
    # 'sharp' high threshold to avoid masking diffuse structures.
    peaks = (sharp > 7) & (sharp == maximum_filter(sharp, 11)) & valid
    # Remove regions very close to the cutout edge, which are unreliable.
    peaks[:8,:]=False;peaks[-8:,:]=False;peaks[:,:8]=False;peaks[:,-8:]=False
    big = (sharp > 22) & (sharp == maximum_filter(sharp, 17)) & valid
    mask = binary_dilation(peaks, iterations=max(2, int(round(2.0/pxscale))))
    mask |= binary_dilation(big, iterations=max(3,int(round(4.5/pxscale))))
    mask |= ~valid
    edge = max(12,int(round(9/pxscale)))
    mask[:edge,:]=True;mask[-edge:,:]=True;mask[:,:edge]=True;mask[:,-edge:]=True
    return mask, {"sharp_peaks":int(peaks.sum()),"strong_sharp_peaks":int(big.sum()),
                  "excluded_fraction":round(float(mask.mean()),4),"edge_pixels":edge}


def score_maps(work, sig, invalid, pxscale):
    """Compensated matched filters, with proper mask-support checks.
    Difference of narrow and broad Gaussians mitigates residual local sky
    gradients; it *does* attenuate very extended light, so upper limits need
    injections and cannot be interpreted as completeness. 'Score' empirical.
    """
    weight=(~invalid).astype(float)
    scores=[]; auxiliary={}
    for size in DETECTION_SCALES_ARCSEC:
        s = size / pxscale
        band_maps={}
        for b in "grz":
            a = work[b] / sig[b]
            den_s = fast_gaussian(weight,s)
            den_b = fast_gaussian(weight,2.5*s)
            sum_s = fast_gaussian(a*weight,s)
            sum_b = fast_gaussian(a*weight,2.5*s)
            close = np.divide(sum_s,den_s,out=np.zeros_like(sum_s),where=den_s>.87)
            far = np.divide(sum_b,den_b,out=np.zeros_like(sum_b),where=den_b>.65)
            dog = close-far
            dog[(den_s<.87)|(den_b<.65)] = np.nan
            band_maps[b]=dog
        g,r,z = (band_maps[b] for b in "grz")
        # Only pixels with positive *independent-band* broad excess are kept.
        combined=(g+r)/math.sqrt(2)
        centre=(slice(24,-24),slice(24,-24))
        med = float(np.nanmedian(combined[centre]))
        emp = robust_std(combined[centre])
        empirical=(combined-med)/max(emp,1e-8)
        good=np.isfinite(empirical) & (g>0) & (r>0)
        empirical[~good]=np.nan
        scores.append(empirical)
        auxiliary[size]=band_maps
    cube=np.stack(scores)
    count=np.sum(np.isfinite(cube),axis=0)
    safe=np.where(np.isfinite(cube),cube,-1e15)
    best=np.argmax(safe,axis=0)
    peak=np.max(safe,axis=0)
    peak[count==0]=np.nan
    return peak,best,auxiliary


def candidates_from_image(work, sig, mask, h, pxscale, maxnum=MAX_CANDIDATES_PER_FIELD):
    peak,which,aux=score_maps(work,sig,mask,pxscale)
    vv=np.where(np.isfinite(peak),peak,-1e14)
    maxima=(vv>=MATCHED_FILTER_THRESHOLD)&(vv==maximum_filter(vv,size=15))
    maxima &= ~mask
    yy,xx=np.nonzero(maxima)
    order=sorted(zip(yy,xx),key=lambda p: float(vv[p]),reverse=True)
    picks=[]
    clearance = distance_transform_edt(~mask)
    for y,x in order:
        size=DETECTION_SCALES_ARCSEC[int(which[y,x])]
        # Ensure the convolution did not manufacture an object near star masks
        # or the edge. This creates lost survey area, recorded separately.
        if clearance[y,x] < max(9,0.9*size/pxscale): continue
        if any(math.hypot(x-t["x"],y-t["y"]) < max(11,size/pxscale*0.85) for t in picks):
            continue
        ra,dec=pixel_to_world(h,float(x),float(y))
        picks.append({"x":int(x),"y":int(y),"ra_deg":round(ra,8),"dec_deg":round(dec,8),
                      "score_empirical":round(float(vv[y,x]),3),
                      "scale_arcsec":size,
                      "viewer_url":f"https://www.legacysurvey.org/viewer?ra={ra:.8f}&dec={dec:.8f}&layer=ls-dr11-south&zoom=16",
                      "catalogue_status":"NOT_CHECKED", "quality":"IMAGE_ONLY_UNVERIFIED"})
        if len(picks)>=maxnum: break
    return picks, {"max_score":round(float(np.nanmax(peak)),3) if np.any(np.isfinite(peak)) else None,
                   "above_threshold_localmax":len(order),"saved":len(picks),
                   "clearance_rule":"mask distance >= max(9px, 0.9*filter scale px)"}



def world_to_pixel(h, ra, dec):
    """Exact inverse TAN projection for a small local field, in zero-based pixels."""
    if not all(k in h for k in ("CRPIX1","CRPIX2","CRVAL1","CRVAL2")):
        raise ValueError("Missing WCS reference")
    a0,d0 = map(math.radians, (float(h['CRVAL1']),float(h['CRVAL2'])))
    a,d = map(math.radians, (ra,dec))
    dra = (a-a0 + math.pi) % (2*math.pi)-math.pi
    cosc = math.sin(d0)*math.sin(d)+math.cos(d0)*math.cos(d)*math.cos(dra)
    if cosc <= 0: raise ValueError("Target outside tangent projection")
    xi = math.cos(d)*math.sin(dra)/cosc
    eta = (math.cos(d0)*math.sin(d)-math.sin(d0)*math.cos(d)*math.cos(dra))/cosc
    delta = np.linalg.solve(cd_matrix(h), np.array([math.degrees(xi),math.degrees(eta)]))
    return float(delta[0]+float(h['CRPIX1'])-1),float(delta[1]+float(h['CRPIX2'])-1)









def local_maxima_all(score, which, mask, pxscale):
    """All position-blind local maxima above unchanged v8 score threshold.
    No min-distance-to-mask veto; the mask itself is still a hard exclusion.
    """
    safe = np.where(np.isfinite(score), score, -1e14)
    maxima = ((safe >= MATCHED_FILTER_THRESHOLD) &
              (safe == maximum_filter(safe, size=15)) & ~mask)
    yy, xx = np.nonzero(maxima)
    points = []
    clear = distance_transform_edt(~mask)
    for y,x in zip(yy,xx):
        size = DETECTION_SCALES_ARCSEC[int(which[y,x])]
        points.append(dict(x=int(x),y=int(y),score=float(safe[y,x]),
                           scale_arcsec=int(size),
                           clearance_px=float(clear[y,x]),
                           original_clearance_px=max(9, 0.9*size/pxscale)))
    points.sort(key=lambda q:q['score'], reverse=True)
    return points


def morphology_metrics(p, mask, aux, work, sig, pxscale):
    """Calculate compact/halo/visible-band diagnostics without calling these
    quantities galaxy probabilities or calibrated measurements.
    """
    x,y=p['x'],p['y']
    scale=p['scale_arcsec']
    rr = max(3., scale/pxscale)
    R = max(4, int(math.ceil(1.4*rr)))
    ya, yb = max(0,y-R), min(mask.shape[0],y+R+1)
    xa, xb = max(0,x-R), min(mask.shape[1],x+R+1)
    yy,xx = np.ogrid[ya:yb,xa:xb]
    radius = np.hypot(xx-x, yy-y)
    disk = radius <=rr
    ring = (radius >0.6*rr)&(radius <=1.4*rr)
    local_mask = mask[ya:yb,xa:xb]
    p['masked_fraction_in_scale'] = float(np.mean(local_mask[disk])) if disk.any() else 1.
    p['masked_fraction_in_ring'] = float(np.mean(local_mask[ring])) if ring.any() else 1.
    for b in 'grz':
        val = float(aux[scale][b][y,x])
        p['broad_'+b] = val if np.isfinite(val) else None
    # Note compactness is a diagnostic; a real galaxy can have point-like knots.
    arr = work['r']/max(float(sig['r']),1e-9)
    central = arr[ya:yb,xa:xb]
    inner = (radius<=max(2., min(4.,rr/3))) & ~local_mask
    outer = (radius>max(2.,min(4.,rr/3))) & (radius<=rr) & ~local_mask
    p['r_inner_med'] = float(np.median(central[inner])) if np.any(inner) else None
    p['r_outer_med'] = float(np.median(central[outer])) if np.any(outer) else None
    # Warning-only tags: mask support is already handled in compensated scores.
    flags=[]
    if p['clearance_px'] < 2: flags.append('NEAR_HARD_MASK')
    if p['masked_fraction_in_scale'] >0.1: flags.append('HIGH_MASK_FRACTION')
    if p['masked_fraction_in_ring'] >0.2: flags.append('MASKED_NEIGHBOURS')
    if p['broad_z'] is None or p['broad_z'] <=0: flags.append('Z_NOT_POSITIVE')
    p['review_flags'] = ';'.join(flags) or 'NO_SIMPLE_MASK_FLAG'
    return p


def suppress_duplicates(points, pxscale):
    """Spatial non-maximum suppression; no scientific classification."""
    taken=[]
    for p in points:
        # 1.25x chosen scale or 12px minimum to merge nearly identical peaks;
        # does not assume an extended source is a dwarf galaxy.
        limit = max(12., 1.25*p['scale_arcsec']/pxscale)
        if any(math.hypot(p['x']-q['x'],p['y']-q['y']) <
               max(limit,1.25*q['scale_arcsec']/pxscale) for q in taken):
            continue
        taken.append(p)
    return taken





# ----------------------- v10 frozen validation specification ------------
OUT = HERE / 'GHOSTHUNTER_V10_CATALOGUE_VALIDATION_RESULTS'
DR_RELEASE = 'ls-dr11'
MATCH_RADIUS_ARCSEC = 20.0
PASS_TOP_N = 10
SHOW_TOP = 15
EXPECTED_PX = ARCSEC_PER_PIXEL
POINTING_EAST_ARCSEC = 52.0
POINTING_NORTH_ARCSEC = 36.0
USE_NETWORK = True
SURVEY_BASE = 'https://www.legacysurvey.org/viewer/fits-cutout'

# Do NOT edit these after running. New experiment -> new version & targets.
CATALOGUE_ID = 'J/ApJS/261/11/table2'
CATALOGUE_DOI = 'https://doi.org/10.26093/cds/vizier.22610011'
SAMPLE_SEED = 'GHOSTHUNTER-V10-2026-10-31-FROZEN'
MU_BINS = ((24.3,24.9),(24.9,25.5),(25.5,26.2))
PER_BIN = 4
TARGET_COUNT = PER_BIN * len(MU_BINS)
CONTEXT_COUNT = 3
CATALOGUE_MIN_ROWS = 1500
CATALOGUE_SERVERS = (
    'https://vizier.cds.unistra.fr',
    'https://vizier.cfa.harvard.edu',
    'https://vizier.idia.ac.za',
)
CATALOGUE_COLUMNS=('SMDG','RAJ2000','DEJ2000','Re','mu0g','VC','f_Re')


def pointing_for(ra,dec):
    return ((ra + POINTING_EAST_ARCSEC/(3600*math.cos(math.radians(dec)))) % 360,
            dec + POINTING_NORTH_ARCSEC/3600)


def parse_vizier_catalogue(raw):
    """Reject incomplete/error pages, parse VizieR asu-tsv columns by name.
    VizieR asu-tsv includes # metadata, a header, units, dashed separator.
    """
    textdata=raw.decode('utf-8-sig','replace')
    if ('#ERROR' in textdata.upper() or '<html' in textdata.lower()
          or 'upstream' in textdata.lower() and 'error' in textdata.lower()):
        raise ValueError('VizieR error/HTML instead of TSV')
    header=None; rows=[]
    for line in textdata.splitlines():
        if not line.strip() or line.startswith('#'):continue
        values=[v.strip() for v in line.split('\t')]
        if header is None:
            if set(CATALOGUE_COLUMNS).issubset(set(values)):
                header=values
            continue
        if len(values)!=len(header):continue
        r=dict(zip(header,values))
        if (not r.get('SMDG') or not r.get('RAJ2000')):continue
        try:
            ra=float(r['RAJ2000']);dec=float(r['DEJ2000'])
            re=float(r['Re']);mu=float(r['mu0g'])
            vc=int(r['VC']);f_re=int(r['f_Re'])
        except (ValueError,TypeError):continue
        if all(map(math.isfinite,(ra,dec,re,mu))) and 0<=ra<360 and -90<=dec<=90:
            rows.append(dict(name=r['SMDG'],ra=ra,dec=dec,re_arcsec=re,
                             mu0g=mu,VC=vc,f_Re=f_re))
    if header is None:raise ValueError('VizieR TSV header fields not found')
    return rows


def fetch_catalogue():
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/'SMUDGES_SOUTH_SOURCE.tsv'
    cache_info=OUT/'SMUDGES_SOURCE_INFO.json'
    if cache.exists() and cache_info.exists():
        try:
            m=json.loads(cache_info.read_text('utf-8'))
            raw=cache.read_bytes()
            if hashlib.sha256(raw).hexdigest()==m.get('sha256'):
                rows=parse_vizier_catalogue(raw)
                if len(rows)>=CATALOGUE_MIN_ROWS:return rows,m
        except Exception:pass
    if not USE_NETWORK:
        raise RuntimeError('Cannot download external catalogue in offline test; no valid cached TSV')
    failures=[]
    for host in CATALOGUE_SERVERS:
        params={'-source':CATALOGUE_ID,'-out':','.join(CATALOGUE_COLUMNS),
                '-out.max':'unlimited'}
        url=host+'/viz-bin/asu-tsv?'+urllib.parse.urlencode(params)
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'GhostHunter10 research-validation (academic catalogue retrieval)'})
            with urllib.request.urlopen(req,timeout=65) as response:
                raw=response.read(5_000_001)
            if len(raw)>5_000_000:raise ValueError('Catalogue response exceeds safe size')
            rows=parse_vizier_catalogue(raw)
            if len(rows)<CATALOGUE_MIN_ROWS:
                raise ValueError(f'Too few valid catalogue rows ({len(rows)}), refusing partial catalogue')
            sha=hashlib.sha256(raw).hexdigest()
            cache.write_bytes(raw)
            m={'url':url,'sha256':sha,'rows':len(rows),'download_utc':now(),
               'catalogue_doi':CATALOGUE_DOI}
            cache_info.write_text(json.dumps(m,indent=2),'utf-8')
            return rows,m
        except Exception as e:
            failures.append(host+': '+type(e).__name__+' '+str(e)[:150])
    raise RuntimeError('All public VizieR mirrors failed. No observational tests were made.\n'+'\n'.join(failures))


def select_controls(rows):
    eligible=[r for r in rows
              if r['VC']==0 and r['f_Re']==0
              and -30<=r['dec']<=25 and 6<=r['re_arcsec']<=24
              and any(a<=r['mu0g']<b for a,b in MU_BINS)]
    # Stable hash of catalogue ID and fixed seed, not based on a score,
    # previous detector successes, imaging previews, or image availability.
    def key(r):return hashlib.sha256((SAMPLE_SEED+'|'+r['name']).encode()).hexdigest()
    controls=[]
    for bidx,(lo,hi) in enumerate(MU_BINS):
        eligible_bin=sorted((r for r in eligible if lo<=r['mu0g']<hi),key=key)
        count=0
        for r in eligible_bin:
            if any(sep_arcsec(r['ra'],r['dec'],q['ra'],q['dec']) < 1.5*3600
                   for q in controls):continue
            p=dict(r,bin=f'{lo:.1f}-{hi:.1f}',bin_index=bidx)
            controls.append(p);count+=1
            if count>=PER_BIN:break
        if count<PER_BIN:
            raise ValueError(f'Only {count} non-overlapping sources in mu bin {lo}-{hi}. No cherry-picking.')
    return controls


def target_specs(controls):
    specs=[]
    for i,r in enumerate(controls,1):
        c_ra,c_dec=pointing_for(r['ra'],r['dec'])
        specs.append(dict(name=f'CONTROL_{i:02d}_'+''.join(x for x in r['name'] if x.isalnum() or x in '-_'),
                          kind='catalogue_control', ra=c_ra,dec=c_dec,
                          truth_ra=r['ra'],truth_dec=r['dec'],
                          reference='Zaritsky et al. 2022 SMUDGes III; photometric candidate',
                          reference_url=CATALOGUE_DOI,mu0g=r['mu0g'],
                          re_arcsec=r['re_arcsec'],surface_brightness_bin=r['bin']))
    for i,r in enumerate(controls[:CONTEXT_COUNT],1):
        # Not blank fields; no false-positive labels.
        specs.append(dict(name='CONTEXT_'+str(i),kind='unlabelled_sky_context',
                          ra=(r['ra']+0.38/max(.35,math.cos(math.radians(r['dec']))))%360,
                          dec=r['dec']-0.22,truth_ra=None,truth_dec=None,
                          reference='fixed spatial offset from selected control',
                          reference_url=''))
    return specs


def load_or_freeze_protocol(rows,catalogue_info):
    path=OUT/'FROZEN_PROTOCOL.json'
    selected=select_controls(rows)
    specs=target_specs(selected)
    settings=dict(algorithm='v8.3 raw-peak triage unchanged',threshold=MATCHED_FILTER_THRESHOLD,
                  scales_arcsec=list(DETECTION_SCALES_ARCSEC),
                  candidate_merge='max(12px,1.25*scale_arcsec/pixel_scale)',
                  zero_clearance_veto=True,match_radius_arcsec=MATCH_RADIUS_ARCSEC,
                  pass_top_n=PASS_TOP_N,layer=DR_RELEASE,pixscale=ARCSEC_PER_PIXEL,
                  pixels=PIXELS,catalogue=CATALOGUE_ID,selection_seed=SAMPLE_SEED,
                  mu_bins=MU_BINS,per_bin=PER_BIN,dec_limits=[-30,25],
                  angular_re_limits=[6,24],require_vc=0,require_f_Re=0)
    expected={'frozen_before_images_utc':now(),'settings':settings,
              'catalogue_source':catalogue_info,'selected_catalogue_sources':selected,
              'field_specifications':specs,
              'note':'Photometric candidates, not individually spectroscopic truths. Evaluation only after detection.'}
    if path.exists():
        previous=json.loads(path.read_text('utf-8'))
        if (previous.get('settings')!=settings or
             previous.get('selected_catalogue_sources')!=selected or
             previous.get('field_specifications')!=specs):
            raise RuntimeError('Existing frozen protocol does not match this version/catalogue. '
                               'Original files were preserved. Do not silently modify a holdout.')
        return previous
    # Freeze before any cutout download or analysis.
    path.write_text(json.dumps(expected,indent=2,allow_nan=False),'utf-8')
    return expected


def download_validated_fits(spec):
    dest=OUT/spec['name'];dest.mkdir(parents=True,exist_ok=True)
    path=dest/'DR11_grz.fits';meta=dest/'DOWNLOAD_METADATA.json'
    query={'ra':f"{spec['ra']:.9f}",'dec':f"{spec['dec']:.9f}",
           'layer':DR_RELEASE,'size':PIXELS,'pixscale':f'{ARCSEC_PER_PIXEL:.2f}',
           'bands':'grz'}
    url=SURVEY_BASE+'?'+urllib.parse.urlencode(query)
    if path.exists() and meta.exists():
        try:
            old=json.loads(meta.read_text('utf-8'))
            if old.get('url')==url and sha256(path)==old.get('sha256'):
                bands,hd=read_fits(path)
                if set(bands)==set('grz') and all(a.shape==(PIXELS,PIXELS) for a in bands.values()):
                    return path,url,'cached_verified'
        except Exception:pass
    if not USE_NETWORK:raise RuntimeError('Offline test: no cached image')
    req=urllib.request.Request(url,headers={'User-Agent':'GhostHunter/10 catalogue pilot (academic research)'})
    with urllib.request.urlopen(req,timeout=55) as resp:data=resp.read(16_000_001)
    if len(data)>16_000_000 or not data.startswith(b'SIMPLE  '):
        raise ValueError('Cutout server did not return a supported FITS image')
    tmp=dest/'DR11_grz.part';tmp.write_bytes(data)
    try:
        bands,head=read_fits(tmp)
        if set(bands)!=set('grz') or any(a.shape!=(PIXELS,PIXELS) for a in bands.values()):
            raise ValueError('Missing bands or bad image shape')
        if abs(pixel_scale(head)-EXPECTED_PX)>.015:
            raise ValueError('Unexpected pixel scale')
        cr,cd=pixel_to_world(head,(PIXELS-1)/2,(PIXELS-1)/2)
        if sep_arcsec(cr,cd,spec['ra'],spec['dec'])>4:
            raise ValueError('Wrong WCS pointing')
    except Exception:
        tmp.unlink(missing_ok=True);raise
    tmp.replace(path)
    meta.write_text(json.dumps({'url':url,'sha256':sha256(path),'time_utc':now()},indent=2),'utf-8')
    return path,url,'downloaded'


def is_good_coverage(bands,hd,spec):
    fractions={b:round(float(np.isfinite(bands[b]).mean()),5) for b in 'grz'}
    scatter={b:robust_std(bands[b]) for b in 'grz'}
    px=pixel_scale(hd)
    cr,cd=pixel_to_world(hd,(PIXELS-1)/2,(PIXELS-1)/2)
    err=sep_arcsec(cr,cd,spec['ra'],spec['dec'])
    checks=dict(finite_fractions=fractions,robust_scatter=scatter,
                pixel_scale_arcsec=px,pointing_wcs_error_arcsec=err,
                warning='Only mechanical coverage checks; no calibrated PSF/depth or exposure map')
    return (bool(all(v>=.98 for v in fractions.values()) and
                 all(math.isfinite(v) and v>1e-8 for v in scatter.values()) and
                 abs(px-ARCSEC_PER_PIXEL)<.015 and err<4),checks)


def ranked_peaks(work,sig,mask,px,h,invert=False):
    w={b:-work[b] for b in 'grz'} if invert else work
    score,which,aux=score_maps(w,sig,mask,px)
    pts=suppress_duplicates(local_maxima_all(score,which,mask,px),px)
    for i,p in enumerate(pts,1):
        morphology_metrics(p,mask,aux,w,sig,px)
        p['rank']=i
        ra,dec=pixel_to_world(h,p['x'],p['y'])
        p['ra_deg']=round(ra,9);p['dec_deg']=round(dec,9)
        p['distance_to_boundary_px']=min(p['x'],p['y'],PIXELS-1-p['x'],PIXELS-1-p['y'])
        p['boundary_relative_to_filter_px']=round(p['distance_to_boundary_px']/(p['scale_arcsec']/px),3)
    return pts


def evaluate_control(pos,spec):
    if spec['truth_ra'] is None:return None
    matching=[]
    for p in pos:
        d=sep_arcsec(p['ra_deg'],p['dec_deg'],spec['truth_ra'],spec['truth_dec'])
        if d<=MATCH_RADIUS_ARCSEC:matching.append((d,p))
    if not matching:return dict(recovered_within_20arcsec=False,rank=None,
                                nearest_arcsec=None,pass_top10=False)
    d,p=min(matching,key=lambda t:t[0])
    return dict(recovered_within_20arcsec=True,rank=p['rank'],
                nearest_arcsec=round(d,3),pass_top10=bool(p['rank']<=PASS_TOP_N),
                flags=p['review_flags'],raw_score=round(p['score'],3),
                filter_scale_arcsec=p['scale_arcsec'])


def render_field_png(path,bands,mask,pos,neg,control_pixel=None):
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    r=bands['r'];q=r[np.isfinite(r)];low,hi=np.percentile(q,[10,99.6])
    for a in axes:
        a.imshow(r,origin='lower',cmap='gray',vmin=low,vmax=hi)
        a.set(xlim=(0,PIXELS-1),ylim=(0,PIXELS-1),xlabel='pixel x',ylabel='pixel y')
        if control_pixel is not None:
            a.plot(*control_pixel,marker='+',color='cyan',ms=10,mew=1.8)
            a.add_patch(Circle(control_pixel,MATCH_RADIUS_ARCSEC/ARCSEC_PER_PIXEL,
                               fill=False,edgecolor='yellow',lw=1))
    axes[0].set_title('DR11 r band | truth marker for evaluation')
    axes[1].set_title('Top 15 positive | unchanged raw ranks')
    for p in pos[:SHOW_TOP]:
        axes[1].plot(p['x'],p['y'],'o',ms=7,mfc='none',mec='orange')
        axes[1].annotate(str(p['rank']),(p['x']+3,p['y']+3),color='yellow',fontsize=8)
    axes[2].set_title('Top 15 inverted | diagnostic only')
    for p in neg[:SHOW_TOP]:axes[2].plot(p['x'],p['y'],'o',ms=7,mfc='none',mec='magenta')
    fig.savefig(path,dpi=115);plt.close(fig)


def run_field(spec):
    print('\nFIELD',spec['name'],'|',spec['kind'],flush=True)
    folder=OUT/spec['name'];folder.mkdir(parents=True,exist_ok=True)
    rec=dict(field=spec['name'],kind=spec['kind'],status='FAILED',
             source_reference=spec['reference'],reference_url=spec['reference_url'],
             centre_ra=spec['ra'],centre_dec=spec['dec'],
             mu0g=spec.get('mu0g'),re_arcsec=spec.get('re_arcsec'),
             surface_brightness_bin=spec.get('surface_brightness_bin'),
             truth_coordinates_evaluation_only=([spec['truth_ra'],spec['truth_dec']]
                          if spec['truth_ra'] is not None else None))
    try:
        path,url,source=download_validated_fits(spec)
        bands,h=read_fits(path)
        good,diag=is_good_coverage(bands,h,spec)
        rec.update(image_url=url,source_image=str(path),image_source=source,
                   sha256=sha256(path),mechanical_coverage=diag,mechanically_valid_image=good)
        if not good:raise ValueError('Image missing usable grz data/WCS; NOT a missed galaxy')
        px=pixel_scale(h)
        work,sig,valid,skytypes=prep_bands(bands)
        mask,maskdiag=compact_mask(work,sig,valid,px)
        pos=ranked_peaks(work,sig,mask,px,h,False)
        neg=ranked_peaks(work,sig,mask,px,h,True)
        tx=ty=None
        if spec['truth_ra'] is not None:
            tx,ty=world_to_pixel(h,spec['truth_ra'],spec['truth_dec'])
        rec['evaluation']=evaluate_control(pos,spec)
        rec.update(status='OK',pixel_scale_arcsec=px,
                   positions_generated_before_truth_evaluation=True,
                   positive_count=len(pos),inverted_count=len(neg),
                   mask_fraction=round(float(mask.mean()),4),mask_diagnostics=maskdiag,
                   sky_subtraction='v8.3 robust-plane unchanged',
                   raw_positive_top_scores=[round(p['score'],2) for p in pos[:5]])
        cols=('rank','x','y','ra_deg','dec_deg','score','scale_arcsec',
              'distance_to_boundary_px','boundary_relative_to_filter_px',
              'clearance_px','masked_fraction_in_scale','masked_fraction_in_ring',
              'broad_g','broad_r','broad_z','r_inner_med','r_outer_med','review_flags')
        write_csv(folder/'POSITIVE_PEAKS.csv',pos,cols)
        write_csv(folder/'INVERTED_PEAKS.csv',neg,cols)
        render_field_png(folder/'FIELD_REVIEW.png',bands,mask,pos,neg,
                         (tx,ty) if tx is not None else None)
        print('  source:',source,' positive=',len(pos),' inverted=',len(neg),
              ' mask_fraction=',round(float(mask.mean()),3))
        if rec['evaluation'] is not None:print('  CONTROL:',rec['evaluation'])
    except Exception as exc:
        rec['error']=type(exc).__name__+': '+str(exc)[:400]
        print('  FIELD FAILED:',rec['error'])
    return rec


def wilson_interval(k,n):
    if n<=0:return None
    z=1.959963984540054
    ph=k/n;d=1+z*z/n
    mid=(ph+z*z/(2*n))/d
    err=z*math.sqrt((ph*(1-ph)+z*z/(4*n))/n)/d
    return [round(max(0,mid-err),4),round(min(1,mid+err),4)]


def report_html(manifest):
    esc=html.escape
    s=manifest['summary']
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>GhostHunter v10</title>',
           '<style>body{font:15px Arial,sans-serif;max-width:1040px;margin:35px auto;line-height:1.5;color:#243043}img{max-width:100%}th,td{border:1px solid #ddd;padding:8px;text-align:left}table{border-collapse:collapse} .note{background:#f9f0d5;padding:15px}</style>',
           '</head><body><h1>GhostHunter v10 — Catalogue validation pilot</h1>',
           '<p class="note"><b>Not a discovery survey or calibrated completeness estimate.</b> '
           'The sources are published photometric candidates selected without detector feedback. '
           'The unchanged v8.3 pipeline reports positional recovery; unrelated peaks have no truth labels. '
           'The imaging source partly overlaps the DR9 catalogue input. No fitted PSF/depth maps.</p>',
           '<p><b>Evaluated:</b> '+str(s['controls_analyzed'])+' / '+str(s['controls_planned'])+
           '; <b>top ten positional matches:</b> '+str(s['controls_recovered_top10'])+
           '; <b>failures of coverage:</b> '+str(s['failed_control_downloads_or_checks'])+'</p>',
           '<h2>Predeclared brightness strata</h2><table><tr><th>mu0g bin</th><th>Matched / evaluable</th><th>image failures</th></tr>']
    for name,v in s['by_surface_brightness'].items():
        parts.append('<tr><td>'+esc(name)+'</td><td>'+str(v['top10'])+'/'+str(v['evaluated'])+
                     '</td><td>'+str(v['coverage_failed'])+'</td></tr>')
    parts.append('</table>')
    for f in manifest['fields']:
        parts.append('<h2>'+esc(f['field'])+'</h2>')
        if f['status']!='OK':
            parts.append('<p>Not evaluated: '+esc(f.get('error','unknown'))+'</p>');continue
        parts.append('<p>positive='+str(f['positive_count'])+', inverted='+str(f['inverted_count'])+
                     ', evaluation='+esc(str(f.get('evaluation')))+'</p>')
        parts.append('<img alt="detector overlay" src="'+esc(f['field'])+'/FIELD_REVIEW.png">')
    parts.append('</body></html>')
    (OUT/'OPEN_ME_VALIDATION.html').write_text('\n'.join(parts),encoding='utf-8')


def self_test():
    header='\t'.join(CATALOGUE_COLUMNS)
    sample='SMUDGES-A\t101.23456\t-20.12345\t12.5\t25.12\t0\t0'
    assert len(parse_vizier_catalogue(('#meta\n'+header+'\n'+sample+'\n').encode()))==1
    s=select_controls([dict(name=f'ID{i:03d}',ra=float(i*9),dec=float(-20+i*.3),
                  re_arcsec=10,mu0g=24.4 if i<6 else 25.1 if i<12 else 25.8,
                  VC=0,f_Re=0) for i in range(18)])
    assert len(s)==TARGET_COUNT
    assert sum(i['bin_index']==0 for i in s)==PER_BIN
    m=np.zeros((81,81),bool);m[45,40]=True
    aux={9:{k:np.ones_like(m,dtype=float) for k in 'grz'}}
    p=dict(x=40,y=40,score=6,scale_arcsec=9,clearance_px=5,original_clearance_px=9)
    morphology_metrics(p,m,aux,{b:np.ones(m.shape) for b in 'grz'},
                       {b:1. for b in 'grz'},.85)
    assert 'MASK' in p['review_flags']
    return 'PASS — TSV parse, 12-source deterministic selection, frozen triage'


def main_v10():
    OUT.mkdir(parents=True,exist_ok=True)
    print('='*72)
    print('GHOSTHUNTER v10 | 12 PREDECLARED SMUDGes CONTROLS | DETECTOR UNCHANGED')
    print('Output:',OUT)
    print('='*72)
    print('OFFLINE SELF TEST:',self_test())
    print('Retrieving SMUDGes published table; failures stop before any survey testing...')
    rows,info=fetch_catalogue()
    print('Catalogued entries:',len(rows),'| SHA256:',info['sha256'])
    protocol=load_or_freeze_protocol(rows,info)
    specs=protocol['field_specifications']
    print('FROZEN:',len(specs),'pointings; 12 catalogue controls + 3 unlabelled contexts')
    print('No detector parameters will be changed after seeing results.')
    # Write partial report after every field to preserve work even on cancellation.
    fields=[]
    for idx,s in enumerate(specs,1):
        print('\nPROCESS',idx,'/',len(specs),flush=True)
        fields.append(run_field(s))
        (OUT/'PARTIAL_FIELD_RESULTS.json').write_text(json.dumps(fields,indent=2,allow_nan=False),'utf-8')
    controls=[f for f in fields if f['kind']=='catalogue_control']
    good=[f for f in controls if f['status']=='OK']
    k=sum(bool(f.get('evaluation',{}).get('pass_top10')) for f in good)
    sb={}
    for lo,hi in MU_BINS:
        label=f'{lo:.1f}-{hi:.1f}'
        z=[f for f in controls if f['surface_brightness_bin']==label]
        evaluated=[f for f in z if f['status']=='OK']
        sb[label]=dict(planned=len(z),evaluated=len(evaluated),
                       top10=sum(bool(f.get('evaluation',{}).get('pass_top10')) for f in evaluated),
                       coverage_failed=len(z)-len(evaluated))
    summary=dict(controls_planned=len(controls),controls_analyzed=len(good),
                 controls_recovered_top10=k,controls_not_recovered_or_not_top10=len(good)-k,
                 failed_control_downloads_or_checks=len(controls)-len(good),
                 raw_positional_recovery_fraction=round(k/len(good),4) if good else None,
                 wilson_95pct_conditional_on_usable_images=wilson_interval(k,len(good)),
                 by_surface_brightness=sb,
                 unlabelled_context_fields_analyzed=sum(f['status']=='OK' and f['kind']=='unlabelled_sky_context' for f in fields),
                 status='PILOT_ONLY_NOT_COMPLETE_SCIENTIFIC_VALIDATION')
    manifest=dict(version='10.0',utc=now(),software_file_sha256=sha256(Path(__file__)),
                  protocol_sha256=sha256(OUT/'FROZEN_PROTOCOL.json'),
                  frozen_protocol=protocol,summary=summary,fields=fields,
                  limitations=[
                   'Published SMUDGes source entries are photometric candidates, not all spectroscopic confirmations.',
                   'This sample is selected by position/catalogue properties, not detector output, but is targeted not blind-sky.',
                   'DR11 overlaps the parent DR9 survey imaging; not independent instrument observations.',
                   'Top-ten matches within 20 arcseconds can be coincident stars; images must be reviewed.',
                   'The Wilson interval reflects only positional recovery among usable selected images; not survey completeness.',
                   'No calibrated PSF/depth/variance or truth-labeled negative fields.',
                   'Inverted peaks are NOT necessarily contamination/false positives.',
                   'Do not retune this version on the selected evaluation set.'
                  ])
    (OUT/'VALIDATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),'utf-8')
    report_html(manifest)
    print('\nFINISHED:',k,'/',len(good),'usable catalogue controls recovered top10;',
          len(controls)-len(good),'control cutouts failed checks')
    print('Surface-brightness strata:',sb)
    print('Open',OUT/'OPEN_ME_VALIDATION.html')
    print('Send VALIDATION_MANIFEST.json and console output. Not a galaxy discovery.')


if __name__=='__main__':
    main_v10()
