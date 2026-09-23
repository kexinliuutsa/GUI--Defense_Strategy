from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

N_TIMING = 32
FEATURE_DIM = 50

def timestamps_of(tr):
    return np.asarray([int(round(float(e.timestamp_us))) for e in tr], dtype=np.int64)

def resample_1d(x, n=N_TIMING):
    x=np.asarray(x,dtype=float)
    if len(x)==0: return np.zeros(n)
    if len(x)==1: return np.repeat(x,n)
    src=np.linspace(0.0,1.0,len(x)); dst=np.linspace(0.0,1.0,n)
    return np.interp(dst,src,x)

def timing_shape(tr):
    ts=timestamps_of(tr)
    if len(ts)<2:
        dt=np.asarray([1.0]); bad_fraction=0.0
    else:
        raw_dt=np.diff(ts.astype(float))
        bad_fraction=float((raw_dt<=0).mean())
        dt=np.maximum(raw_dt,1.0)
    mean_dt=max(float(np.mean(dt)),1e-12)
    rel=dt/mean_dt
    seq=resample_1d(rel)
    centered=seq-np.mean(seq)
    fft=np.abs(np.fft.rfft(centered))
    fft8=np.zeros(8)
    m=min(8,max(len(fft)-1,0))
    if m: fft8[:m]=fft[1:1+m]
    q10,q25,q50,q75,q90=np.quantile(rel,[.10,.25,.50,.75,.90])
    stats=np.asarray([np.std(rel),q10,q25,q50,q75,q90,np.max(rel),np.min(rel),q75-q25,bad_fraction],dtype=float)
    out=np.nan_to_num(np.concatenate([seq,fft8,stats]),nan=0.0,posinf=0.0,neginf=0.0)
    if out.shape!=(FEATURE_DIM,): raise RuntimeError(f"timing_shape dimension mismatch: {out.shape}")
    return out

def _valid_gestures(session):
    out=[]
    for g in session:
        try:
            if g is not None and len(g)>0 and hasattr(g[0],"timestamp_us"): out.append(g)
        except TypeError:
            pass
    return out

def _gesture_matrix(sessions):
    gs=[]
    for s in sessions: gs.extend(_valid_gestures(s))
    if not gs: return np.zeros((0,FEATURE_DIM),dtype=float)
    return np.vstack([timing_shape(g) for g in gs])

def _aggregate(scores: Sequence[float]) -> float:
    a=np.asarray(scores,dtype=float); a=a[np.isfinite(a)]
    if a.size==0: return float("-inf")
    return float(np.quantile(a,0.90))

@dataclass
class ExactTemporalShapeOverlay:
    model: object
    threshold: float

    @classmethod
    def fit(cls,human_sessions,raw_sessions):
        hx=_gesture_matrix(human_sessions); rx=_gesture_matrix(raw_sessions)
        if len(hx)==0 or len(rx)==0: raise ValueError("Need non-empty Human and Raw gestures")
        x=np.vstack([hx,rx])
        y=np.concatenate([np.zeros(len(hx),dtype=int),np.ones(len(rx),dtype=int)])
        model=make_pipeline(StandardScaler(),SVC(C=3.0,kernel="rbf",gamma="scale",class_weight="balanced",probability=False,cache_size=1000))
        model.fit(x,y)
        hs=np.asarray([cls._score_with_model(model,s) for s in human_sessions],dtype=float)
        hf=hs[np.isfinite(hs)]
        if hf.size==0: raise ValueError("No finite Human session scores")
        threshold=np.nextafter(float(np.max(hf)),np.inf)
        return cls(model=model,threshold=float(threshold))

    @staticmethod
    def _score_with_model(model,session):
        gs=_valid_gestures(session)
        if not gs: return float("-inf")
        x=np.vstack([timing_shape(g) for g in gs])
        return _aggregate(model.decision_function(x))

    def score(self,session): return self._score_with_model(self.model,session)
    def detect(self,session): return bool(self.score(session)>=self.threshold)

    def audit(self,human_sessions,raw_sessions):
        hs=np.asarray([self.score(s) for s in human_sessions],dtype=float)
        rs=np.asarray([self.score(s) for s in raw_sessions],dtype=float)
        hf=hs[np.isfinite(hs)]; rf=rs[np.isfinite(rs)]
        return {
            "representation":"exact_EXP34_35_timing_shape_50D",
            "classifier":"StandardScaler + RBF-SVC(C=3,class_weight=balanced)",
            "session_aggregation":"q90_gesture_margin",
            "threshold":float(self.threshold),
            "human_fpr_in_sample":float(np.mean(hf>=self.threshold)) if hf.size else float("nan"),
            "raw_recall_in_sample":float(np.mean(rf>=self.threshold)) if rf.size else float("nan"),
            "human_score_max":float(np.max(hf)) if hf.size else float("nan"),
            "human_score_q99":float(np.quantile(hf,.99)) if hf.size else float("nan"),
            "raw_score_median":float(np.median(rf)) if rf.size else float("nan"),
            "n_human_sessions":int(hf.size),
            "n_raw_sessions":int(rf.size),
        }
