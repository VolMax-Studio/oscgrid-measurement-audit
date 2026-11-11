#!/usr/bin/env python3
"""Pre-registered paired MLP check. Separate workspace; no VolMax repo modification."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CSV=ROOT/"unpacked/labeled/labeled.csv"
FEATURES=["IA","IC","UA BB","UB BB","UC BB"]
RATIOS={"train":0.6,"val":0.1,"test":0.3}
SEEDS=(443,444,445)
SIZE=32
STRIDE=10
BATCH=128
MAX_EPOCHS=100
EARLY_PATIENCE=10
torch.set_num_threads(3)
torch.set_num_interop_threads(1)

spec=importlib.util.spec_from_file_location("pinned_oscgrid_author_models",HERE/"author_models.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MLP=mod.MLP

def load_data():
    cols=pd.read_csv(CSV,nrows=0).columns.tolist()
    ml=[c for c in cols if c.startswith(("ML_1","ML_2","ML_3"))]
    usecols=["sample","file_name"]+FEATURES+ml
    dtype={c:"float32" for c in FEATURES+ml}
    frame=pd.read_csv(CSV,usecols=usecols,dtype=dtype,low_memory=False)
    frame=frame.dropna(subset=FEATURES).reset_index(drop=True)
    m1=frame[[c for c in ml if c.startswith("ML_1")]].eq(1).any(axis=1)
    m2=frame[[c for c in ml if c.startswith("ML_2")]].eq(1).any(axis=1)
    m3=frame[[c for c in ml if c.startswith("ML_3")]].eq(1).any(axis=1)
    frame["target"]=np.select([~(m1|m2|m3),m1&~m2&~m3,m2&~m3,m3],[0,1,2,3],-1).astype("int8")
    del m1,m2,m3
    if (frame["target"]<0).any():raise RuntimeError("Invalid labels")
    frame["recording_id"]=frame["file_name"].str.extract(r"^([0-9a-f]{32})",expand=False)
    if frame["recording_id"].isna().any():raise RuntimeError("Recording ID parse mismatch")
    frame["block_id"]=((frame["file_name"]!=frame["file_name"].shift()) |
                       (frame["target"]!=frame["target"].shift())).cumsum().astype("int32")
    block=frame.groupby("block_id",sort=True).agg(file_name=("file_name","first"),
        recording_id=("recording_id","first"), target=("target","first"),length=("target","size"))
    block=block.loc[block["length"]>=SIZE].copy()
    if len(block)!=1672:raise RuntimeError(f"Unexpected eligible block count: {len(block)}")
    return frame,block

def split_assignments(block,which):
    if which=="author":
        shuffled=block.sample(frac=1,random_state=443)
        assignments={}
        for target,g in shuffled.groupby("target"):
            maxima={s:g["length"].sum()*v for s,v in RATIOS.items()}
            used={s:0 for s in RATIOS}
            for key,row in g.iterrows():
                for s in ("val","test","train"):
                    if used[s]+row["length"]<=maxima[s] or s=="train":
                        assignments[key]=s
                        used[s]+=row["length"]
                        break
        block["split"]=block.index.map(assignments)
    else:
        recordings=sorted(set(block["recording_id"]))
        idx=np.arange(len(recordings))
        trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=443).split(idx,groups=recordings))
        train,val_sub=next(GroupShuffleSplit(n_splits=1,test_size=1/7,random_state=444).split(
            trainval,groups=np.array(recordings)[trainval]))
        rec_train={recordings[trainval[i]] for i in train}
        rec_val={recordings[trainval[i]] for i in val_sub}
        rec_test={recordings[i] for i in test}
        if rec_train & rec_val or rec_train & rec_test or rec_val & rec_test:raise RuntimeError("Record-level leakage")
        block["split"]=block.recording_id.map(lambda r:
           "train" if r in rec_train else ("val" if r in rec_val else "test"))
    return block

def make_windows(frame,block):
    # Train-only StandardScaler; only eligible block rows, as in author code.
    feat=frame[FEATURES].to_numpy(dtype=np.float32,copy=False)
    tgt=frame["target"].to_numpy(dtype=np.int8,copy=False)
    ids=frame["block_id"].to_numpy(dtype=np.int32,copy=False)
    endpoints=np.flatnonzero(np.r_[True,ids[1:]!=ids[:-1],True])
    bounds={int(ids[a]):(int(a),int(b)) for a,b in zip(endpoints[:-1],endpoints[1:])}
    idxs={}
    for split in ("train","val","test"):
        chunks=[np.arange(bounds[int(i)][0],bounds[int(i)][1]-SIZE+1,STRIDE,dtype=np.int32)
                for i in block.index[block["split"]==split]]
        idxs[split]=np.concatenate(chunks)
    allrows={}
    for split in ("train","val","test"):
        starts=[bounds[int(i)] for i in block.index[block["split"]==split]]
        allrows[split]=np.concatenate([np.arange(a,b,dtype=np.int32) for a,b in starts])
    if not all(len(i)>0 for i in idxs.values()):raise RuntimeError("Some split has zero windows")
    scaler=StandardScaler().fit(feat[allrows["train"]])
    # Scale using only training observations, then materialize windows.
    windows={}
    for split in ("train","val","test"):
        indices=idxs[split][:,None]+np.arange(SIZE,dtype=np.int32)
        values=feat[indices]
        values=(values-scaler.mean_.astype("float32"))/scaler.scale_.astype("float32")
        labels=tgt[idxs[split]]
        counts=np.bincount(labels,minlength=4).tolist()
        windows[split]=(torch.from_numpy(np.ascontiguousarray(values,dtype=np.float32)),
                        torch.from_numpy(labels.astype(np.int64)))
        print(split,"windows",len(labels),"class",counts,flush=True)
    class_rows=np.bincount(tgt[allrows["train"]],minlength=4)
    if np.any(class_rows==0):raise RuntimeError("Train class missing")
    weights=torch.from_numpy((len(allrows["train"])/(4*class_rows)).astype("float32"))
    return windows,weights,{
        "eligible_raw_rows":{x:len(allrows[x]) for x in allrows},
        "windows":{x:len(idxs[x]) for x in idxs},
        "class_rows":{x:np.bincount(tgt[allrows[x]],minlength=4).tolist() for x in allrows},
        "class_windows":{x:np.bincount(windows[x][1].numpy(),minlength=4).tolist() for x in windows},
        "class_weights":weights.tolist(),
    }

def validation(model,ds):
    model.eval()
    xx,yy=ds
    predictions=[]
    with torch.no_grad():
        for x in xx.split(1024):
            predictions.append(torch.argmax(model(x),dim=1).cpu())
    pr=torch.cat(predictions).numpy()
    return accuracy_score(yy.numpy(),pr)

def train_run(which,windows,weights,seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    m=MLP(frame_size=32,channel_num=5,hidden_size=128,output_size=4,num_layers=2,dropout=.3)
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    tr=TensorDataset(*windows["train"])
    loader=DataLoader(tr,batch_size=BATCH,shuffle=True,num_workers=0)
    best_val=-1;best_state=None;best_epoch=0;stale=0
    start=time.time()
    for e in range(1,MAX_EPOCHS+1):
        m.train()
        for xx,yy in loader:
            optimizer.zero_grad()
            loss=criterion(m(xx),yy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),10)
            optimizer.step()
        acc=validation(m,windows["val"])
        if acc>best_val:
            best_val=acc;best_epoch=e
            best_state={k:v.detach().clone() for k,v in m.state_dict().items()}
            stale=0
        else:stale+=1
        if e==1 or e%5==0 or stale>=EARLY_PATIENCE:
            print(which,"seed",seed,"epoch",e,"best_val",round(best_val,4),
                "current_val",round(acc,4),"elapsed_s",round(time.time()-start,1),flush=True)
        if stale>=EARLY_PATIENCE:break
    if best_state is None:raise RuntimeError("No validation checkpoint")
    m.load_state_dict(best_state)
    m.eval()
    truth=windows["test"][1].numpy()
    pred=[]
    with torch.no_grad():
        for x in windows["test"][0].split(1024):
            pred.append(torch.argmax(m(x),dim=1).cpu().numpy())
    pred=np.concatenate(pred)
    pp,rr,ff,nn=precision_recall_fscore_support(truth,pred,labels=[0,1,2,3],zero_division=0)
    return {
        "split":which,"seed":seed,"epochs_executed":e,"selected_epoch":best_epoch,
        "validation_best_accuracy":float(best_val),
        "elapsed_seconds":round(time.time()-start,2),
        "test_accuracy":float(accuracy_score(truth,pred)),
        "test_macro_f1":float(f1_score(truth,pred,average="macro")),
        "test_weighted_f1":float(f1_score(truth,pred,average="weighted")),
        "per_class_precision":pp.tolist(),"per_class_recall":rr.tolist(),
        "per_class_f1":ff.tolist(),"per_class_support":nn.tolist(),
        "confusion":confusion_matrix(truth,pred,labels=[0,1,2,3]).tolist()
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["audit","run"],default="audit")
    ap.add_argument("--seed",type=int,choices=SEEDS)
    ap.add_argument("--split",choices=["author","group"])
    args=ap.parse_args()
    orig=hashlib.sha256((HERE/"MLP_PAIRED_PROTOCOL_v0.1.md").read_bytes()).hexdigest()
    expected="6ab8a1deeb503c990617510d8ffb5b1d1aaa0b0146898010fb31e92b73d787d4"
    if orig!=expected:raise RuntimeError("Prereg protocol unexpectedly edited")
    frame,blocks=load_data()
    for which in ([args.split] if args.split else ["author","group"]):
        b=split_assignments(blocks.copy(),which)
        rows=b.groupby("split").length.sum().to_dict()
        nfiles=b.groupby("split").recording_id.nunique().to_dict()
        overlaps={a+"_"+c:len(set(b.loc[b.split==a,"recording_id"])&
                             set(b.loc[b.split==c,"recording_id"]))
                  for a,c in (("train","test"),("train","val"),("val","test"))}
        print("SPLIT",which,"row counts",rows,"source recording counts",nfiles,"overlaps",overlaps,flush=True)
        if which=="author" and rows!={"train":722721,"val":120316,"test":361183}:
            raise RuntimeError("Author block split failed to reproduce")
        if which=="group" and any(overlaps.values()):raise RuntimeError("Group leakage detected")
        windows,weights,sizes=make_windows(frame,b)
        manifest={"split_method":which,"protocol_sha256":orig,
                  "source_model_sha256":hashlib.sha256((HERE/"author_models.py").read_bytes()).hexdigest(),
                  "row_counts":rows,"source_counts":nfiles,"source_overlap":overlaps,**sizes}
        (HERE/f"MLP_{which}_SPLIT.json").write_text(json.dumps(manifest,indent=2)+"\n")
        if args.mode=="audit":continue
        seeds=[args.seed] if args.seed else SEEDS
        for seed in seeds:
            result=train_run(which,windows,weights,seed)
            dest=HERE/f"MLP_{which}_seed{seed}.json"
            dest.write_text(json.dumps(result,indent=2)+"\n")
            print("FINAL",which,seed,"macroF1",round(result["test_macro_f1"],4),
                  "classF1",[round(x,4) for x in result["per_class_f1"]],
                  "path",dest.name,flush=True)

if __name__=="__main__":
    main()
