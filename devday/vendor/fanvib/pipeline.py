"""Shi-Tomasi -> forward/backward LK -> camera compensation -> PSD.

All-frame survivor selection and coordinate-wise median aggregation preserve
our experimental pipeline. Fault localization is deliberately unimplemented.
"""
from pathlib import Path
import json
import math
import cv2
import numpy as np
from scipy.signal import detrend, welch, periodogram, find_peaks
from scipy.integrate import trapezoid


def read_first_frame(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    playback_fps = float(cap.get(cv2.CAP_PROP_FPS))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Video has no decodable frame")
    return cap, frame, playback_fps


def roi_mask(shape, roi):
    h, w = shape
    mask = np.zeros(shape, np.uint8)
    kind = roi.get("shape")
    if kind == "rectangle":
        x, y, rw, rh = map(float, roi["bounds"])
        if rw <= 0 or rh <= 0 or x < 0 or y < 0 or x + rw > w or y + rh > h:
            raise ValueError("ROI rectangle outside decoded frame or empty")
        mask[int(y):int(y + rh), int(x):int(x + rw)] = 255
    elif kind == "polygon":
        pts = np.asarray(roi["points"], dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
            raise ValueError("Polygon needs at least three finite (x,y) points")
        if np.any(pts < 0) or np.any(pts[:, 0] >= w) or np.any(pts[:, 1] >= h):
            raise ValueError("Polygon outside decoded frame")
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    elif kind == "annulus":
        cx, cy = map(float, roi["center"])
        rx, ry = map(float, roi["radii"])
        inner, outer = float(roi.get("inner", .87)), float(roi.get("outer", 1.04))
        if not 0 <= cx < w or not 0 <= cy < h or min(rx, ry) <= 0 or not 0 <= inner < outer:
            raise ValueError("Invalid annulus")
        yy, xx = np.indices(shape)
        r = np.sqrt(((xx - cx) / rx)**2 + ((yy - cy) / ry)**2)
        selected = (r > inner) & (r < outer)
        half = roi.get("half")
        if half not in (None, "upper", "lower"):
            raise ValueError("Annulus half must be upper or lower")
        if half == "upper": selected &= yy < cy
        if half == "lower": selected &= yy >= cy
        mask[selected] = 255
    else:
        raise ValueError(f"Unknown ROI shape: {kind}")
    if not mask.any():
        raise ValueError("Empty ROI")
    return mask


def spectrum(signal, fs, nperseg):
    signal = detrend(np.asarray(signal), axis=0, type="linear")
    fw, pw = welch(signal, fs, nperseg=min(nperseg, len(signal)), axis=0)
    ff, pf = periodogram(signal, fs, window="hann", axis=0)
    return fw, pw.sum(axis=1), ff, pf.sum(axis=1)


def summarize(signal, fs, nperseg):
    f, p, ff, pf = spectrum(signal, fs, nperseg)
    idx = find_peaks(p)[0]
    idx = [i for i in idx if f[i] > 0]
    idx = sorted(idx, key=lambda i: p[i], reverse=True)[:8]
    return {"welch_candidate_peaks": [{"hz": float(f[i]), "psd_px2_per_hz": float(p[i])} for i in idx],
            "welch_bin_hz": float(f[1]-f[0]), "periodogram_bin_hz": float(ff[1]-ff[0]),
            "detrended_rms_px": float(np.sqrt(np.mean(np.sum(detrend(signal, axis=0)**2, axis=1)))),
            "integrated_psd_rms_px": float(np.sqrt(trapezoid(p, f)))}


def analyze_video(video, config, output_dir):
    """Run a local video using supplied semantic ROIs; return a JSON-safe dict.

    capture_fps describes preserved capture-frame spacing, NOT playback rate.
    Returns measured motion locations, not diagnosed fault locations.
    """
    fs = float(config.get("capture_fps", 0))
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("capture_fps must be supplied explicitly and be positive")
    segment = config.get("nperseg", 512)
    if not isinstance(segment, int) or segment < 8:
        raise ValueError("nperseg must be an integer >= 8")
    rois = config.get("rois", [])
    names = [r.get("id") for r in rois]
    if not names or any(not isinstance(n, str) or not n or not n.replace('_', '').isalnum() for n in names) or len(set(names)) != len(names):
        raise ValueError("ROI ids must be unique nonempty letters/digits/underscores")
    bg_names = [r["id"] for r in rois if r.get("role") == "background"]
    if not bg_names:
        raise ValueError("At least one static background ROI is required")
    for r in rois:
        if r.get("role") not in ("background", "target"):
            raise ValueError("Every ROI needs target/background role")
    for pair in config.get("relative_pairs", []):
        if len(pair) != 2 or any(n not in names for n in pair) or pair[0] == pair[1]:
            raise ValueError("Invalid relative_pairs")
    cap, first, playback = read_first_frame(video)
    try:
        h, w = first.shape[:2]
        expected = config.get("frame_size")
        if expected is not None and expected != [w, h]:
            raise ValueError(f"Decoded frame is {[w,h]}, config expects {expected}; inspect orientation first")
        prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        masks = {r["id"]: roi_mask(prev.shape, r) for r in rois}
        points, labels = [], []
        for name, mask in masks.items():
            p = cv2.goodFeaturesToTrack(prev, 150, .008, 8, mask=mask, blockSize=7)
            if p is not None:
                points.append(p); labels.extend([name] * len(p))
        if not points:
            raise ValueError("No trackable features in ROIs")
        p = np.concatenate(points); labels = np.array(labels)
        alive = np.ones(len(p), bool); tracks = [p[:, 0].copy()]
        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame.shape != first.shape: raise ValueError("Frame size changed")
            cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            q, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p, None, winSize=(21,21), maxLevel=3, criteria=(3,30,.001))
            if q is None: raise ValueError("Optical flow failed")
            back, sb, _ = cv2.calcOpticalFlowPyrLK(cur, prev, q, None, winSize=(21,21), maxLevel=3, criteria=(3,30,.001))
            if back is None: raise ValueError("Backward optical flow failed")
            alive &= (st[:,0] > 0) & (sb[:,0] > 0) & np.isfinite(q[:,0]).all(axis=1) & (np.linalg.norm(back[:,0]-p[:,0],axis=1) < .7)
            tracks.append(q[:,0].copy()); p = q; prev = cur
    finally:
        cap.release()
    tr = np.array(tracks)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(output_dir)/"tracks.npz", tr=tr, labels=labels, keep=alive)
    if len(tr) < 8: raise ValueError("At least eight decodable frames are required")
    bg = alive & np.isin(labels, bg_names)
    if bg.sum() < 6: raise ValueError("Fewer than six full-duration background features; use better ROI/video")
    threshold = float(config.get("ransac_threshold_px", 1))
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("ransac_threshold_px must be positive")
    affine = np.empty_like(tr); translation = np.empty_like(tr)
    residual, inliers = [], []
    for t in range(len(tr)):
        A, ins = cv2.estimateAffinePartial2D(tr[0,bg],tr[t,bg],method=cv2.RANSAC,ransacReprojThreshold=threshold)
        if A is None: raise ValueError(f"Background transform failed at frame {t}; no identity fallback")
        affine[t] = tr[t] - (tr[0] @ A[:,:2].T + A[:,2])
        translation[t] = tr[t] - tr[0] - np.median(tr[t,bg]-tr[0,bg],axis=0)
        residual.append(float(np.median(np.linalg.norm(affine[t,bg],axis=1))))
        inliers.append(float(ins.mean()))
    signals, regions = {}, {}
    for roi in rois:
        name = roi["id"]; sel = alive & (labels == name)
        r = {"roi": roi, "initial_points": int((labels == name).sum()), "retained_points": int(sel.sum()), "status": "measured" if sel.sum() >= 3 else "insufficient_features"}
        if sel.sum() >= 3:
            for mode, arr in [("raw", tr-tr[0]),("affine",affine),("translation",translation)]:
                s = np.median(arr[:,sel],axis=1); signals[f"{name}__{mode}"] = s
                r[mode] = summarize(s,fs,segment)
        regions[name] = r
    relative = {}
    for a,b in config.get("relative_pairs", []):
        pair = f"{a}_minus_{b}"; relative[pair] = {}
        for mode in ("raw","affine","translation"):
            ka,kb = f"{a}__{mode}",f"{b}__{mode}"
            if ka in signals and kb in signals:
                s = signals[ka] - signals[kb]; signals[f"{pair}__{mode}"] = s
                relative[pair][mode] = summarize(s,fs,segment)
    warnings = ["Candidate spectral peaks are measurements, not automatic vibration/fault detections.",
                "Semantic ROIs are supplied by API or reviewed plan; motion location is not fault source.",
                "Handheld parallax, rolling shutter, stabilization and exposure may affect results.",
                "Inspect raw/affine/translation spectra and repeat measurements; fitting residuals are not calibrated confidence."]
    if abs(playback-fs) > .01:
        warnings.append("Capture and playback FPS differ: results assume each decoded frame preserves one consecutive capture frame. Export resampling/trimming must be checked independently.")
    result = {"schema_version":"0.1", "status":"measurement_only", "video":Path(video).name,
              "metadata":{"playback_fps":playback,"capture_fps":fs,"timebase_source":"explicit_user_config","frame_preservation_verified":False,"decoded_frames":len(tr),"capture_duration_if_frames_preserved_s":len(tr)/fs,"nyquist_hz":fs/2,"decoded_frame_size":[w,h],"coordinate_system":"decoded upright pixels; inspect first frame"},
              "quality":{"full_duration_background_points":int(bg.sum()),"median_affine_inlier_fraction":float(np.median(inliers)),"median_background_point_residual_px":float(np.median(residual))},
              "regions":regions,"relative_motion":relative,"diagnosis":{"vibration_confirmed":None,"abnormal_region":None,"fault_source":None},"warnings":warnings}
    out = Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"analysis.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    np.savez_compressed(out/"signals.npz",capture_fps=fs,**signals)
    annotation = first.copy()
    for i,(name,mask) in enumerate(masks.items()):
        color = ((70+i*67)%230,(190+i*43)%230,(100+i*97)%230)
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotation,contours,-1,color,2)
        ys,xs = np.where(mask); cv2.putText(annotation,name,(int(xs.min()),max(15,int(ys.min())-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    cv2.imwrite(str(out/"roi.png"),annotation)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,1,figsize=(10,8),layout="constrained")
    for name,r in regions.items():
        if r["status"] != "measured": continue
        f,p,_,_ = spectrum(signals[f"{name}__affine"],fs,segment)
        axes[0].semilogy(f,np.maximum(p,1e-16),label=name)
    for name,r in regions.items():
        if r["status"] != "measured" or r["roi"]["role"] != "target": continue
        for mode in ("raw","affine","translation"):
            _,_,f,p = spectrum(signals[f"{name}__{mode}"],fs,segment)
            axes[1].semilogy(f,np.maximum(p,1e-16),label=f"{name}: {mode}")
    for a in axes:
        a.set(xlim=(0,fs/2),xlabel="Frequency (Hz, configured capture timebase)",ylabel="PSD (pixel²/Hz)")
        a.legend(fontsize=7); a.grid(alpha=.2)
    axes[0].set_title("Welch: affine-compensated region measurements (not fault locations)")
    axes[1].set_title("Hann periodogram: compensation sensitivity")
    fig.savefig(out/"spectra.png",dpi=140); plt.close(fig)
    return result
