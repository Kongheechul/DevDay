import math
from pathlib import Path
import cv2
import numpy as np

def prepare_video(path, capture_fps, out_dir, count=3):
    if not math.isfinite(capture_fps) or capture_fps<=0: raise ValueError('Supply true capture_fps, not playback FPS')
    if not 1<=count<=8: raise ValueError('Representative frame count must be 1..8')
    path=Path(path).resolve()
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError(f'Cannot open {path.name}')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    try:
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));playback=float(cap.get(cv2.CAP_PROP_FPS));rotation=float(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        if n<8: raise ValueError('At least eight frames required')
        indices=np.unique(np.linspace(0,n-1,count,dtype=int));frames=[];size=None
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read()
            if not ok: raise ValueError(f'Cannot decode representative frame {i}')
            current=[im.shape[1],im.shape[0]]
            if size is not None and current!=size: raise ValueError('Frame size changed')
            size=current;p=out/f'frame_{i:06d}.jpg'
            if not cv2.imwrite(str(p),im,[cv2.IMWRITE_JPEG_QUALITY,92]): raise OSError('Cannot save frame')
            frames.append({'path':str(p.resolve()),'frame_index':int(i),'capture_time_if_preserved_s':float(i/capture_fps)})
        return {'filename':path.name,'frame_size':size,'frame_count':n,'playback_fps':playback,'capture_fps':capture_fps,'rotation_metadata_deg':rotation,'capture_duration_if_preserved_s':n/capture_fps,'frame_preservation_verified':False,'frames':frames}
    finally: cap.release()
