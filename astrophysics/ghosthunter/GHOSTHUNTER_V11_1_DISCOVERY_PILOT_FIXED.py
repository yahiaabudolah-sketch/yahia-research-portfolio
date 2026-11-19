"""GHOSTHUNTER v11.1 — DISCOVERY PILOT (integrity-check compatibility patch).

The frozen science detector is byte-identical to v10 across seven routines.
The original v11 preflight used ast.dump(), which can vary by Python version.
v11.1 checks canonical source text instead; sky search and detector are unchanged.

V10 DETECTION FUNCTIONS FROZEN.

One-file 49-cutout contiguous blind reconnaissance around the nearby NGC 7713
group; no astropy needed. numpy scipy matplotlib. Produces a complete
candidate table, duplicate merged list, review shortlist and thumbnails.
A previously uncatalogued galaxy cannot be declared from this one survey.

No calibrated PSF/depth/completeness. Masked area and image-model artifacts
are concerns. The target host and surrounding fields have prior literature.
Every candidate remains UNVERIFIED until multi-catalogue review.
Settings are predeclared and never automatically tuned to results.

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






# ---------------------------------------------------------------------------
# v11 SCIENTIFIC SURVEY PROTOCOL. Detector above is copied from v10 verbatim.
# ---------------------------------------------------------------------------
VERSION='11.0.1'
# Version-independent canonical SOURCE digest of 7 frozen v10 detector routines.
# Normalises CRLF vs LF but retains every token, comment and whitespace inside functions.
V10_FROZEN_DETECTOR_SHA256='23e56a921f7a3f4546e959e8e3288be9ba0e2efcc3fb80cd7f8e8e7976d43c68'
OUT = HERE / 'GHOSTHUNTER_V11_DISCOVERY_RESULTS'
DR_RELEASE='ls-dr11'
PIXELS=512
ARCSEC_PER_PIXEL=.85
GRID_SIDE=7
STEP_ARCSEC=360.0   # 6 arcmin centres; cutout FOV is ~7.25 arcmin.
HOST_NAME='NGC 7713'
HOST_RA_DEG=354.0625       # J2000, ESO Messenger 193 (2024)
HOST_DEC_DEG=-37.938889
MIN_CENTRE_HOST_WARNING_ARCMIN=10.0
CATALOGUE_SEARCH_ARCSEC=60.0
DUPLICATE_MERGE_ARCSEC=12.0
MAX_REVIEW=24
SHOW_PER_TILE=25
USER_AGENT='GHOSTHUNTER-v11-student-public-image-research'
SURVEY_SOURCE='https://www.legacysurvey.org/viewer/fits-cutout'
SMUDGES_CATALOGUE='J/ApJS/261/11/table2'
SMUDGES_COLS=('SMDG','RAJ2000','DEJ2000','Re','mu0g','VC','f_Re')

# This is a discovery-PILOT only; not claimed to be representative or untouched.
# For any interrupted run, press F5 again: completed fields and verified images
# are reused, missing fields are processed, and outputs are updated.

def detector_fingerprint():
    """Version-independent text fingerprint of the seven frozen v10 routines.

    ast.dump changes representation across Python interpreters (Spyder may use
    a different version). We use AST only for stable function line boundaries,
    then hash the literal source lines, normalizing Windows CRLF via splitlines.
    This fails CLOSED if a frozen routine was edited, instead of bypassing the
    integrity check. Changes elsewhere (e.g., UI or survey management) are OK.
    """
    import ast
    names=('prep_bands','compact_mask','score_maps','local_maxima_all',
           'morphology_metrics','suppress_duplicates','ranked_peaks')
    src=Path(__file__).read_text(encoding='utf-8')
    lines=src.splitlines()
    tree=ast.parse(src)
    blocks=[]
    for n in names:
        matches=[q for q in tree.body if isinstance(q,ast.FunctionDef) and q.name==n]
        if len(matches)!=1: raise RuntimeError('Frozen detector function missing/ambiguous: '+n)
        f=matches[0]
        if f.end_lineno is None: raise RuntimeError('Cannot verify frozen source boundaries for: '+n)
        blocks.append('\n'.join(lines[f.lineno-1:f.end_lineno]))
    return hashlib.sha256('\n---GH10-FUNC---\n'.join(blocks).encode('utf-8')).hexdigest()

def make_grid():
    specs=[]
    for row in range(GRID_SIDE):
        for col in range(GRID_SIDE):
            # RA grid is corrected by cos(dec), so angular E/W separation ~360".
            north=(row-(GRID_SIDE-1)/2)*STEP_ARCSEC
            east=(col-(GRID_SIDE-1)/2)*STEP_ARCSEC
            dec=HOST_DEC_DEG+north/3600
            ra=(HOST_RA_DEG+east/(3600*math.cos(math.radians(dec))))%360
            specs.append({'id':f'TILE_{row+1:02d}_{col+1:02d}',
                          'row':row,'col':col,'ra':round(ra,9),'dec':round(dec,9)})
    return specs

def freeze_protocol():
    specs=make_grid()
    actual=detector_fingerprint()
    if actual!=V10_FROZEN_DETECTOR_SHA256:
        raise RuntimeError('Frozen detector source differs from v10; no sky data processed. Expected '+V10_FROZEN_DETECTOR_SHA256+' but got '+actual+'. Download an unedited v11.1 script.')
    source=Path(__file__)
    expected={'version':VERSION,'host':HOST_NAME,'host_ra_deg':HOST_RA_DEG,
        'host_dec_deg':HOST_DEC_DEG,
        'settings':{'grid_side':GRID_SIDE,'step_arcsec':STEP_ARCSEC,'pixel_scale':ARCSEC_PER_PIXEL,
                    'pixel_size':PIXELS,'layer':DR_RELEASE,
                    'threshold':MATCHED_FILTER_THRESHOLD,
                    'filters_arcsec':list(DETECTION_SCALES_ARCSEC),
                    'duplicate_merge_arcsec':DUPLICATE_MERGE_ARCSEC,
                    'catalogue_match_arcsec':CATALOGUE_SEARCH_ARCSEC,
                    'frozen_detection':'v10 copied without changes, with no hard mask clearance veto',
                    'fingerprint_method':'canonical original Python function source (not version-sensitive ast.dump)',
                    'v10_frozen_detector_source_sha256':V10_FROZEN_DETECTOR_SHA256},
        'fields':specs,
        'references':['https://www.legacysurvey.org/dr11/description/',
                      'https://doi.org/10.1051/0004-6361/202557968',
                      'https://doi.org/10.26093/cds/vizier.22610011'],
        'limitations':['NGC 7713 vicinity is not known to be unsearched',
                       'Survey uses existing public sky-subtracted coadded images',
                       'Stars/galaxies/host halo not automatically classified',
                       'Not calibrated for completeness, novelty, PSF or purity']}
    p=OUT/'FROZEN_DISCOVERY_PROTOCOL.json'
    if p.exists():
        old=json.loads(p.read_text(encoding='utf-8'))
        if old!=expected:
            raise RuntimeError('FROZEN PROTOCOL MISMATCH. Do not reuse or overwrite the old survey. '+str(p))
    else:
        OUT.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(expected,indent=2,allow_nan=False),encoding='utf-8')
    return expected

def get_image(spec):
    fd=OUT/'tiles'/spec['id'];fd.mkdir(parents=True,exist_ok=True)
    path=fd/'dr11_grz.fits';meta=fd/'download.json'
    q={'ra':f"{spec['ra']:.9f}",'dec':f"{spec['dec']:.9f}",
       'layer':DR_RELEASE,'size':PIXELS,'pixscale':f'{ARCSEC_PER_PIXEL:.2f}','bands':'grz'}
    url=SURVEY_SOURCE+'?'+urllib.parse.urlencode(q)
    if path.exists() and meta.exists():
        try:
            md=json.loads(meta.read_text(encoding='utf-8'))
            if md.get('url')==url and md.get('sha256')==sha256(path):
                bands,h=read_fits(path)
                if set(bands)==set('grz') and all(v.shape==(PIXELS,PIXELS) for v in bands.values()):
                    return path,url,'cached_verified'
        except Exception: pass
    req=urllib.request.Request(url,headers={'User-Agent':USER_AGENT})
    with urllib.request.urlopen(req,timeout=55) as response:
        raw=response.read(16_000_001)
    if not raw.startswith(b'SIMPLE  ') or len(raw)>16_000_000:
        raise ValueError('Survey did not return supported FITS pixels')
    tmp=fd/'image_download.part'
    tmp.write_bytes(raw)
    try:
        bands,h=read_fits(tmp)
        if set(bands)!=set('grz') or any(v.shape!=(PIXELS,PIXELS) for v in bands.values()):
            raise ValueError('Bad shape/band coverage')
        cr,cd=pixel_to_world(h,(PIXELS-1)/2,(PIXELS-1)/2)
        if sep_arcsec(cr,cd,spec['ra'],spec['dec'])>4 or abs(pixel_scale(h)-ARCSEC_PER_PIXEL)>.02:
            raise ValueError('WCS centre or pixel scale does not match request')
    except Exception:
        tmp.unlink(missing_ok=True);raise
    tmp.replace(path)
    meta.write_text(json.dumps({'url':url,'sha256':sha256(path),'utc':now()},indent=2),'utf-8')
    return path,url,'downloaded'

def assert_coverage(bands,h,spec):
    for b in 'grz':
        if b not in bands or bands[b].shape!=(PIXELS,PIXELS):
            raise ValueError('Missing band '+b)
        if np.isfinite(bands[b]).mean()<.98 or robust_std(bands[b])<1e-8:
            raise ValueError('Unusable/missing '+b+' band; NOT a no-galaxy field')
    ra,dec=pixel_to_world(h,(PIXELS-1)/2,(PIXELS-1)/2)
    if sep_arcsec(ra,dec,spec['ra'],spec['dec'])>4:
        raise ValueError('Pointing WCS invalid')

def run_one(spec):
    fd=OUT/'tiles'/spec['id']
    result_path=fd/'TILE_RESULT.json'
    if result_path.exists():
        try:
            old=json.loads(result_path.read_text('utf-8'))
            if old['field']==spec['id'] and old.get('status')=='OK':
                im=fd/'dr11_grz.fits';meta=fd/'download.json'
                md=json.loads(meta.read_text('utf-8'))
                if sha256(im)==md['sha256']:
                    print('  cached completed field:',len(old['positive']),'peaks')
                    return old
        except Exception:pass
    result={'field':spec['id'],'row':spec['row'],'col':spec['col'],
            'ra':spec['ra'],'dec':spec['dec'],'status':'FAILED'}
    try:
        p,url,state=get_image(spec)
        bands,h=read_fits(p)
        assert_coverage(bands,h,spec)
        px=pixel_scale(h)
        work,sig,valid,skies=prep_bands(bands)
        mask,maskdiag=compact_mask(work,sig,valid,px)
        pos=ranked_peaks(work,sig,mask,px,h,False)
        neg=ranked_peaks(work,sig,mask,px,h,True)
        for point in pos:
            point['field']=spec['id']
            point['host_offset_arcmin']=round(sep_arcsec(point['ra_deg'],point['dec_deg'],HOST_RA_DEG,HOST_DEC_DEG)/60,3)
            point['host_halo_warning']=point['host_offset_arcmin']<MIN_CENTRE_HOST_WARNING_ARCMIN
        result.update(status='OK',sha256=sha256(p),source=state,image_url=url,
                      pixels=PIXELS,mask_fraction=round(float(mask.mean()),4),mask_diagnostics=maskdiag,
                      positive=pos,inverted_count=len(neg),
                      inverted_top_scores=[round(a['score'],3) for a in neg[:5]],
                      sky_model_warning='Broad sky subtraction can remove faint galaxies')
        # Save a compact review plot for each field (first 15 candidates only).
        fig,ax=plt.subplots(figsize=(6.4,6.4))
        r=bands['r'];z=r[np.isfinite(r)];lo,hi=np.percentile(z,[9,99.7])
        ax.imshow(r,origin='lower',cmap='gray',vmin=lo,vmax=hi)
        for a in pos[:SHOW_PER_TILE]:
            if a['distance_to_boundary_px']<15:continue
            ax.plot(a['x'],a['y'],'o',mfc='none',mec='orange',ms=8,mew=1)
            ax.annotate(str(a['rank']),(a['x']+4,a['y']+4),color='yellow',fontsize=7)
        ax.set_title(spec['id']+' | raw v10 peaks, not galaxies')
        ax.set_xlabel('x [px]');ax.set_ylabel('y [px]')
        fig.savefig(fd/'TILE_REVIEW.png',dpi=105);plt.close(fig)
        result_path.write_text(json.dumps(result,indent=1,allow_nan=False),encoding='utf-8')
        print('  downloaded/cached:',state,' positive:',len(pos),'inverted:',len(neg),
              'masked:',round(float(mask.mean())*100,1),'%')
    except Exception as e:
        result['error']=type(e).__name__+': '+str(e)[:240]
        print('  FAILED (not a nondetection):',result['error'])
        result_path.parent.mkdir(parents=True,exist_ok=True)
        result_path.write_text(json.dumps(result,indent=1,allow_nan=False),encoding='utf-8')
    return result

def duplicate_merge(rows):
    # sort by score and keep every detection in a cluster as a member.
    # Duplicates merged using 12 arcsec on-sky -- can conflate close binaries!
    # Complete raw candidates are always preserved in ALL_PEAKS.csv.
    groups=[]
    for p in sorted(rows,key=lambda x:-x['score']):
        assigned=None
        for g in groups:
            r=g['representative']
            if sep_arcsec(p['ra_deg'],p['dec_deg'],r['ra_deg'],r['dec_deg'])<=DUPLICATE_MERGE_ARCSEC:
                assigned=g;break
        if assigned is None:
            groups.append({'representative':p,'members':[p]})
        else:assigned['members'].append(p)
    return groups

def parse_smdg(raw):
    hdr=None;out=[]
    for l in raw.decode('utf-8-sig','replace').splitlines():
        if not l or l.startswith('#'):continue
        vals=[v.strip() for v in l.split('\t')]
        if hdr is None:
            if set(SMUDGES_COLS).issubset(set(vals)):hdr=vals
            continue
        if len(vals)!=len(hdr):continue
        d=dict(zip(hdr,vals))
        try:
            out.append({'name':d['SMDG'],'ra':float(d['RAJ2000']),
                        'dec':float(d['DEJ2000'])})
        except (ValueError,KeyError):continue
    if hdr is None or len(out)<1000:raise ValueError('Incomplete/unknown VizieR catalogue')
    return out

def load_smdg():
    cache=OUT/'SMUDGES_REFERENCE_CATALOGUE.tsv';meta=OUT/'SMUDGES_CATALOGUE_INFO.json'
    if cache.exists() and meta.exists():
        try:
            sha=sha256(cache)
            if json.loads(meta.read_text('utf-8'))['sha256']==sha:
                return parse_smdg(cache.read_bytes()),{'source':'cache','sha256':sha}
        except Exception:pass
    errors=[]
    for base in ('https://vizier.cfa.harvard.edu','https://vizier.cds.unistra.fr'):
        query={'-source':SMUDGES_CATALOGUE,'-out':','.join(SMUDGES_COLS),'-out.max':'unlimited'}
        url=base+'/viz-bin/asu-tsv?'+urllib.parse.urlencode(query)
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':USER_AGENT}),timeout=60) as resp:
                raw=resp.read(5000001)
            if len(raw)>5000000:raise ValueError('Response too large')
            rows=parse_smdg(raw)
            cache.write_bytes(raw)
            sha=sha256(cache)
            meta.write_text(json.dumps({'url':url,'sha256':sha,'count':len(rows)},indent=2),'utf-8')
            return rows,{'source':url,'sha256':sha,'entries':len(rows)}
        except Exception as e:errors.append(f'{base}:{str(e)[:100]}')
    return [],{'source':'UNAVAILABLE','errors':errors,
               'warning':'Catalogue verification incomplete; no novelty claims permitted'}

def annotate_groups(groups,smdg):
    out=[]
    for i,g in enumerate(groups,1):
        p=g['representative']
        near=[]
        for r in smdg:
            ds=sep_arcsec(p['ra_deg'],p['dec_deg'],r['ra'],r['dec'])
            if ds<CATALOGUE_SEARCH_ARCSEC:near.append((ds,r))
        near.sort(key=lambda t:t[0])
        # A preview indicator, not a classifier, no score/probability invented.
        inner=p.get('r_inner_med');outer=p.get('r_outer_med')
        extended=bool(inner is not None and outer is not None and inner>0 and outer/inner>.065)
        clean=bool(p['masked_fraction_in_scale']<.12 and
            p['boundary_relative_to_filter_px']>2 and not p['host_halo_warning'])
        tier='A_REVIEW_FIRST' if clean and extended and (p.get('broad_z') or 0)>0 else 'B_REVIEW' if extended else 'C_OTHER_PEAK'
        out.append({**p,'candidate_id':f'GH11-{i:04d}',
            'n_tiles':len(set(x['field'] for x in g['members'])),
            'n_peaks_merged':len(g['members']),
            'tier':tier,
            'outer_to_inner_r':round(outer/inner,5) if inner is not None and outer is not None and inner>0 else None,
            'smdg_match':near[0][1]['name'] if near else '',
            'smdg_sep_arcsec':round(near[0][0],2) if near else None,
            'catalogue_screen':'SMUDGES_MATCH' if near else 'NO_SMUDGES_MATCH_NOT_EVIDENCE_OF_NOVELTY',
            'external_catalogues_status':'NOT_CHECKED_SIMBAD_NED_GAIA_IDMAGE_ELVES',
            'science_status':'UNVERIFIED_IMAGE_PEAK'})
    priority={'A_REVIEW_FIRST':0,'B_REVIEW':1,'C_OTHER_PEAK':2}
    out.sort(key=lambda r:(priority[r['tier']],r['smdg_match']!='',
             r['host_halo_warning'],r['masked_fraction_in_scale'],
             -min(50,r['score'])))
    for i,row in enumerate(out,1):row['review_order']=i
    return out

def csv_fields(rows):
    standard=('candidate_id','review_order','tier','ra_deg','dec_deg','score','scale_arcsec',
           'host_offset_arcmin','host_halo_warning','outer_to_inner_r',
           'masked_fraction_in_scale','masked_fraction_in_ring','boundary_relative_to_filter_px',
           'review_flags','broad_g','broad_r','broad_z','r_inner_med','r_outer_med',
           'n_tiles','n_peaks_merged','smdg_match','smdg_sep_arcsec',
           'catalogue_screen','external_catalogues_status','science_status','field','rank','x','y')
    return standard

def save_montage(shortlist):
    if not shortlist:return
    samples=shortlist[:MAX_REVIEW]
    # Build 3 separate 8-source pages for readable real r-band cutouts.
    for page in range((len(samples)+7)//8):
        group=samples[page*8:(page+1)*8]
        fig,axes=plt.subplots(2,4,figsize=(13.2,7.3),constrained_layout=True)
        for ax in axes.flat:ax.set_axis_off()
        for ax,p in zip(axes.flat,group):
            path=OUT/'tiles'/p['field']/'dr11_grz.fits'
            b,h=read_fits(path);r=b['r']
            x,y=p['x'],p['y']; R=60
            xa,xb=max(0,x-R),min(PIXELS,x+R+1)
            ya,yb=max(0,y-R),min(PIXELS,y+R+1)
            cut=r[ya:yb,xa:xb];values=cut[np.isfinite(cut)]
            if values.size:
                lo,hi=np.percentile(values,[15,99.5])
                if hi<=lo:hi=lo+.01
                ax.imshow(cut,origin='lower',cmap='gray',vmin=lo,vmax=hi,
                          extent=[xa,xb,ya,yb])
                ax.plot(x,y,'+',color='cyan',ms=9,mew=1.4)
            ax.set_title(f"{p['candidate_id']} | {p['tier'].split('_')[0]} | tile {p['field'][-5:]}",fontsize=9)
            ax.set_axis_on();ax.tick_params(labelsize=7)
        fig.suptitle('GHOSTHUNTER v11 | NO OBJECT CLASSIFIED | r-band cutouts, individual stretches',fontsize=12)
        fig.savefig(OUT/f'SHORTLIST_CONTACT_SHEET_{page+1}.png',dpi=120);plt.close(fig)

def report_page(stats,rows):
    h=html.escape
    items=['<!doctype html><html><head><meta charset="utf-8"><title>GHOSTHUNTER v11 pilot</title>',
       '<style>body{max-width:1100px;margin:24px auto;font:15px Arial;color:#233;line-height:1.5}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:5px;text-align:left} .warning{padding:12px;background:#ffecc3} img{max-width:100%}</style></head><body>',
       '<h1>GHOSTHUNTER v11 — sky reconnaissance</h1>',
       '<div class="warning"><b>No discoveries claimed.</b> Results are candidate image peaks, not confirmed galaxies. '
       'SMUDGes is only one prior catalogue. All candidates require NED, SIMBAD, Gaia, recent galaxy surveys, and independent imaging checks. '
       'No depth/PSF map, completeness or purity calculation. Known host halo may contaminate measurements.</div>',
       '<p>'+h(json.dumps(stats))+'</p>',
       '<h2>Highest-priority human review candidates</h2>',
       '<table><tr><th>ID</th><th>Tier</th><th>RA</th><th>Dec</th><th>score*</th><th>r ratio</th><th>SMUDGes</th><th>viewer</th></tr>']
    for p in rows[:50]:
        url=f"https://www.legacysurvey.org/viewer?ra={p['ra_deg']:.8f}&dec={p['dec_deg']:.8f}&layer=ls-dr11&zoom=15"
        items.append('<tr><td>'+h(p['candidate_id'])+'</td><td>'+h(p['tier'])+
                     '</td><td>'+str(p['ra_deg'])+'</td><td>'+str(p['dec_deg'])+
                     '</td><td>'+str(round(p['score'],2))+'</td><td>'+h(str(p['outer_to_inner_r']))+
                     '</td><td>'+h(str(p['smdg_match']))+'</td><td><a href="'+h(url)+'">sky viewer</a></td></tr>')
    items.append('</table><p>*Scores are uncalibrated empirical ranks, not sigma or probabilities. All tiers are heuristic triage and not scientific classifications.</p>')
    for i in range(1,4):
        if (OUT/f'SHORTLIST_CONTACT_SHEET_{i}.png').exists():
            items.append(f'<h2>Review sheet {i}</h2><img src="SHORTLIST_CONTACT_SHEET_{i}.png">')
    items.append('</body></html>')
    (OUT/'OPEN_ME_DISCOVERY_PILOT.html').write_text('\n'.join(items),encoding='utf-8')

def finalize(proto,fields):
    good=[f for f in fields if f['status']=='OK']
    peaks=[p for f in good for p in f['positive']]
    groups=duplicate_merge(peaks)
    smdg,cat_status=load_smdg()
    rows=annotate_groups(groups,smdg)
    save_montage(rows)
    cols=csv_fields(rows)
    write_csv(OUT/'ALL_MERGED_CANDIDATES.csv',rows,cols)
    write_csv(OUT/'REVIEW_FIRST_50.csv',rows[:50],cols)
    write_csv(OUT/'ALL_RAW_PEAKS.csv',peaks,('field','rank','x','y','ra_deg','dec_deg',
        'score','scale_arcsec','review_flags','masked_fraction_in_scale','boundary_relative_to_filter_px'))
    fov=(PIXELS*ARCSEC_PER_PIXEL)
    footprint_deg2=round((((GRID_SIDE-1)*STEP_ARCSEC+fov)/3600.)**2,4)
    summary={'requested_fields':len(proto['fields']),'processed':len(good),
             'failed_fields':len(fields)-len(good),'raw_positive_peaks':len(peaks),
             'merged_image_peaks':len(rows),'tier_a_review_count':sum(p['tier']=='A_REVIEW_FIRST' for p in rows),
             'smdg_screen':cat_status,'nominal_grid_bbox_area_deg2':footprint_deg2,
             'actual_coverage_area_warning':'Overlap, masks, failed tiles, and sky variations mean usable surveyed area is smaller',
             'novelty_claims':0,'scientific_status':'IMAGE_CANDIDATES_ONLY_NO_DISCOVERY'}
    report={'version':VERSION,'utc':now(),'protocol':proto,'summary':summary,
            'field_summaries':[{k:v for k,v in f.items() if k!='positive'} for f in fields],
            'limitations':['No reliable completeness/purity from this survey',
              'No SIMBAD, NED, Gaia or contemporary literature catalogue screening',
              'SMDG matches are incomplete sky catalog checks',
              'No host distance assignment for any candidate',
              'No new galaxy claim without independent source confirmation']}
    (OUT/'SURVEY_MANIFEST.json').write_text(json.dumps(report,indent=2,allow_nan=False),'utf-8')
    report_page(summary,rows)
    return summary

def self_test_v11():
    specs=make_grid()
    assert len(specs)==49
    assert (abs(sep_arcsec(specs[24]['ra'],specs[24]['dec'],HOST_RA_DEG,HOST_DEC_DEG))<1e-5)
    for col in range(6):
        a=specs[24+col-3]; b=specs[24+col-2]
        assert abs(sep_arcsec(a['ra'],a['dec'],b['ra'],b['dec'])-STEP_ARCSEC)<.15
    a=[{'ra_deg':30.,'dec_deg':-40.,'score':8.,'field':'a'},
       {'ra_deg':30+2/(3600*math.cos(math.radians(-40))),'dec_deg':-40.,'score':12.,'field':'b'},
       {'ra_deg':31.,'dec_deg':-40.,'score':5.,'field':'c'}]
    gr=duplicate_merge(a)
    assert len(gr)==2 and len(gr[0]['members'])==2
    return 'PASS (grid angular geometry, RA wrap, duplicate grouping)'

def main_v11():
    OUT.mkdir(parents=True,exist_ok=True)
    print('='*74)
    print('GHOSTHUNTER v11.1 | NGC 7713 7x7 SKY RECONNAISSANCE')
    print('No discovery claims; v10 detector copied; rerun F5 resumes')
    print('Output:',OUT)
    print('='*74)
    print('OFFLINE TEST:',self_test_v11())
    print('FROZEN v10 DETECTOR:',detector_fingerprint()==V10_FROZEN_DETECTOR_SHA256, '(source-text verified)')
    proto=freeze_protocol()
    manifest_path=OUT/'FIELD_PROGRESS.json'
    fields=[]
    for i,s in enumerate(proto['fields'],1):
        print(f'FIELD {i:02d}/{len(proto["fields"])} {s["id"]}',flush=True)
        result=run_one(s);fields.append(result)
        manifest_path.write_text(json.dumps([{k:v for k,v in f.items() if k!='positive'} for f in fields],indent=1,allow_nan=False),'utf-8')
    summary=finalize(proto,fields)
    print('\nFINISHED. Survey:',summary)
    print('Open:',OUT/'OPEN_ME_DISCOVERY_PILOT.html')
    print('Send SURVEY_MANIFEST.json, REVIEW_FIRST_50.csv, SHORTLIST_CONTACT_SHEET_1.png.')
    print('All points are UNVERIFIED. Absence from SMUDGes does NOT mean new.')

if __name__=='__main__':
    main_v11()
