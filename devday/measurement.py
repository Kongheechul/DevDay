"""Retain the original LK/Welch core; add paired evidence and point maps."""
import math
import re
from pathlib import Path
import numpy as np
from scipy.signal import welch, periodogram, find_peaks
from scipy.integrate import trapezoid
from .vendor.fanvib.pipeline import analyze_video

DEFAULT_MODES=('raw','translation','affine')

def validate_bands(bands, fps_values):
    nyquist=min(fps_values)/2
    if bands is None:
        bands=[('low',1.,min(10.,nyquist*.8))]
        if nyquist>15: bands.append(('higher',10.,min(100.,nyquist*.8)))
    if not bands: raise ValueError('At least one frequency band required')
    names=[]
    for name,lo,hi in bands:
        if name in names or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}',name) or not all(math.isfinite(v) for v in [lo,hi]) or not 0<lo<hi<nyquist:
            raise ValueError('Bands must have unique IDs and 0 < low < high < both Nyquist limits')
        names.append(name)
    return bands

def band_rms(signal, fs, lo, hi, nperseg=512):
    f,p=welch(signal,fs=fs,nperseg=min(nperseg,len(signal)),detrend='linear',axis=0)
    p=p.sum(axis=-1);sel=(f>=lo)&(f<=hi)
    if sel.sum()<2: return None
    return np.sqrt(np.maximum(0,trapezoid(p[sel],f[sel],axis=0)))

def candidate_peaks(signal,fs):
    f,p=periodogram(signal,fs=fs,window='hann',detrend='linear',axis=0);p=p.sum(axis=-1)
    ix=find_peaks(p)[0];ix=[i for i in ix if f[i]>=.5];ix=sorted(ix,key=lambda i:p[i],reverse=True)[:5]
    return [{'hz':float(f[i]),'psd_px2_per_hz':float(p[i])} for i in ix]

def measure(video, config, directory, bands):
    result=analyze_video(video,config,directory)
    d=np.load(Path(directory)/'tracks.npz',allow_pickle=False)
    tr=d['tr'][:,d['keep']];labels=d['labels'][d['keep']];raw=tr-tr[0]
    bg_names=[r['id'] for r in config['rois'] if r['role']=='background'];bg=np.isin(labels,bg_names)
    translation=raw-np.median(raw[:,bg],axis=1)[:,None,:]
    signals=np.load(Path(directory)/'signals.npz',allow_pickle=False)
    fs=config['capture_fps'];regions={};point_values={};rel={}
    for name,lo,hi in bands:
        values=band_rms(translation,fs,lo,hi,config['nperseg'])
        point_values[name]=values
    for roi in config['rois']:
        n=roi['id'];mask=labels==n;r=result['regions'][n]
        rec={'status':r['status'],'initial_points':r['initial_points'],'retained_points':r['retained_points'],'bands':{},'peaks':{}}
        if r['status']=='measured':
            for mode in DEFAULT_MODES:
                rec['peaks'][mode]=candidate_peaks(signals[n+'__'+mode],fs)
            for name,lo,hi in bands:
                vals=point_values[name]
                row={'point_median_rms_px':None if vals is None else float(np.median(vals[mask])), 'region_median_signal_rms_by_mode':{}, 'full_survivor_third_rms_px':[]}
                for mode in DEFAULT_MODES:
                    v=band_rms(signals[n+'__'+mode],fs,lo,hi,config['nperseg'])
                    row['region_median_signal_rms_by_mode'][mode]=None if v is None else float(v)
                for part in np.array_split(translation[:,mask],3):
                    v=band_rms(part,fs,lo,hi,config['nperseg'])
                    row['full_survivor_third_rms_px'].append(None if v is None else float(np.median(v)))
                rec['bands'][name]=row
        regions[n]=rec
    for a,b in config['relative_pairs']:
        name=a+'_minus_'+b;key=name+'__raw'
        if key not in signals.files: continue
        rel[name]={'a':a,'b':b,'bands':{}}
        for bn,lo,hi in bands:
            v=band_rms(signals[key],fs,lo,hi,config['nperseg'])
            rel[name]['bands'][bn]=None if v is None else float(v)
    return {'metadata':result['metadata'],'quality':result['quality'],'warnings':result['warnings'],'regions':regions,'relative':rel}, {'points':tr[0],'labels':labels,'band_values':point_values}

def safe_ratio(candidate,reference):
    if candidate is None or reference is None or reference<=1e-8: return None
    return candidate/reference

def build_evidence(plan,metadata,measurements,bands,conditions):
    comparisons=[];relative=[];allowed=[];setup_ok=plan.same_setup_assessment=='consistent'
    ref=measurements.get('reference',{});cand=measurements.get('candidate',{})
    semantics={r.id:r for r in plan.regions}
    for region in plan.regions:
        if region.role!='target': continue
        a=ref.get('regions',{}).get(region.id,{});b=cand.get('regions',{}).get(region.id,{})
        for bn,lo,hi in bands:
            av=a.get('bands',{}).get(bn,{});bv=b.get('bands',{}).get(bn,{})
            ar=av.get('point_median_rms_px');br=bv.get('point_median_rms_px')
            eligible=setup_ok and region.semantic_confidence!='low' and a.get('retained_points',0)>=3 and b.get('retained_points',0)>=3 and ar is not None and br is not None
            eid=region.id+':'+bn
            row={'evidence_id':eid,'roi_id':region.id,'part_name':region.part_name,'band_hz':[lo,hi], 'reference_rms_px':ar,'candidate_rms_px':br,'difference_px':None if ar is None or br is None else br-ar,'ratio':safe_ratio(br,ar),'reference_points':a.get('retained_points',0),'candidate_points':b.get('retained_points',0),'eligible_for_interpretation':eligible,'sparse_tracking':min(a.get('retained_points',0),b.get('retained_points',0))<10,'third_ratios':[safe_ratio(y,x) for x,y in zip(av.get('full_survivor_third_rms_px',[]),bv.get('full_survivor_third_rms_px',[]))], 'reference_compensation_sensitivity':av.get('region_median_signal_rms_by_mode',{}),'candidate_compensation_sensitivity':bv.get('region_median_signal_rms_by_mode',{})}
            comparisons.append(row)
            if eligible: allowed.append(eid)
    for name,a in ref.get('relative',{}).items():
        b=cand.get('relative',{}).get(name)
        if b is None: continue
        for bn,lo,hi in bands:
            av=a['bands'][bn];bv=b['bands'][bn];eid='relative:'+name+':'+bn
            eligible=setup_ok and all(semantics[x].semantic_confidence!='low' and min(ref['regions'][x]['retained_points'],cand['regions'][x]['retained_points'])>=3 for x in [a['a'],a['b']]) and av is not None and bv is not None
            relative.append({'evidence_id':eid,'a':a['a'],'b':a['b'],'band_hz':[lo,hi],'reference_rms_px':av,'candidate_rms_px':bv,'ratio':safe_ratio(bv,av),'eligible_for_interpretation':eligible})
            if eligible: allowed.append(eid)
    return {'schema_version':'devday.evidence/1','roi_plan':plan.model_dump(),'videos':metadata,'conditions':conditions,'measurements':measurements,'region_comparisons':comparisons,'relative_comparisons':relative,'allowed_evidence_ids':allowed,'method':{'point_metric':'median across retained points of detrended 2D Welch band RMS','primary_compensation':'background median translation','relative_metric':'difference of ROI median trajectories; not a calibrated joint angle','unit':'image_pixel','physical_calibration':None,'thirds':'same full-survivor points in three temporal thirds; not independent redetection or repeated experiments'},'limitations':['Only one recording per condition; no validated abnormality threshold or diagnosis accuracy.','Same setup is inferred/user-declared, not guaranteed by coordinate validity.','Low-frequency movement may include camera stabilization/parallax/compression/tracking noise.','Capture FPS is user supplied; exported frame preservation is not independently verified.','Spectral peaks are not proven rotor RPM or fault orders.','Motion location is not excitation source.','Sparse/low-semantic-confidence ROIs must not support localization.']}
