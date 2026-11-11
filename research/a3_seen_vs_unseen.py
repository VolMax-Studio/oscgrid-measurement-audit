#!/usr/bin/env python3
"""A3: author block split, classify test (source recording, class) seen in train vs unseen.
Exploratory 3-seed validation, isolated workspace. Model code imported from pinned author_models.py.
"""
import collections,csv,hashlib,json,random,time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import run_paired_mlp as base

HERE=Path(__file__).resolve().parent
OUT=HERE/"a3_seen_vs_unseen_results"
SEEDS=(443,444,445)
THRESHOLDS=(1,6,32)
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
 return h.hexdigest()
def save(p,x):
 with open(p,"w",encoding="utf8") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")

def window_record_ids(frame,blocks,split):
 ids=frame["block_id"].to_numpy(dtype=np.int32,copy=False)
 cuts=np.flatnonzero(np.r_[True,ids[1:]!=ids[:-1],True])
 bounds={int(ids[a]):(int(a),int(b)) for a,b in zip(cuts[:-1],cuts[1:])}
 out=[]
 for bid,row in blocks.loc[blocks.split==split].iterrows():
  a,b=bounds[int(bid)]
  n=len(range(a,b-32+1,10))
  out.extend([row.recording_id]*n)
 return np.array(out,dtype=str)

def evaluate(truth,pred,recordids,train_pair_set):
 records=[]
 for k in range(4):
  for rid in sorted(set(recordids[truth==k].tolist())):
   subset=(recordids==rid)&(truth==k)
   n=int(subset.sum());correct=int((pred[subset]==k).sum())
   records.append({"class":k,"recording_id":rid,"test_true_windows":n,
     "correct_class_windows":correct,"record_recall":correct/n,
     "seen_same_class_in_train":bool((rid,k) in train_pair_set)})
 summ={}
 for k in range(4):
  for threshold in THRESHOLDS:
   for group in (True,False):
    vals=[r for r in records if r["class"]==k and r["seen_same_class_in_train"]==group
          and r["test_true_windows"]>=threshold]
    summ[f"class{k}_min{threshold}_seen{group}"]={
      "n_recordings":len(vals),
      "equal_record_mean_recall":float(np.mean([r["record_recall"] for r in vals])) if vals else None,
      "equal_record_median_recall":float(np.median([r["record_recall"] for r in vals])) if vals else None,
      "n_class_test_windows":int(sum(r["test_true_windows"] for r in vals)),
      "fraction_recordings_majority_hit":float(np.mean([r["record_recall"]>.5 for r in vals])) if vals else None}
 return records,summ

def train_model(seed,w,weights):
 torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 model=base.MLP(frame_size=32,channel_num=5,hidden_size=128,output_size=4,num_layers=2,dropout=.3)
 crit=torch.nn.CrossEntropyLoss(weight=weights)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 tr=DataLoader(TensorDataset(*w["train"]),batch_size=128,shuffle=True,num_workers=0,drop_last=False)
 assert len(w["train"][1])%128!=1, "Author split has batch singleton"
 best=-1.; best_epoch=0; stale=0;best_state=None;start=time.monotonic()
 for ep in range(1,101):
  model.train()
  for inputs,y in tr:
   opt.zero_grad()
   loss=crit(model(inputs),y)
   loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),10)
   opt.step()
  acc=base.validation(model,w["val"])
  if acc>best:
   best=acc;best_epoch=ep;stale=0
   best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
  else:stale+=1
  if ep==1 or ep%10==0 or stale>=10:
   print("A3",seed,"epoch",ep,"val_best",round(best,5),"elapsed",round(time.monotonic()-start,1),flush=True)
  if stale>=10:break
 if best_state is None:raise RuntimeError("No trained model")
 model.load_state_dict(best_state)
 model.eval()
 pred=[]
 with torch.no_grad():
  for x in w["test"][0].split(1024):
   pred.append(torch.argmax(model(x),dim=1).numpy())
 pred=np.concatenate(pred)
 return model,best_state,pred,{"selected_epoch":best_epoch,"trained_epochs":ep,"best_val_accuracy":best,
                               "train_seconds":round(time.monotonic()-start,2)}

def main():
 protocol=HERE/"A3_FROZEN_PROTOCOL_v0.1.md"
 assert sha(base.CSV)=="fa3c891163c27880296c240b77c1d23d2ce41be36858ca233f774be9e86f1854"
 frame,blocks=base.load_data()
 b=base.split_assignments(blocks.copy(),"author")
 assert b.groupby("split").length.sum().to_dict()=={"train":722721,"val":120316,"test":361183}
 train_pairs={(str(row.recording_id),int(row.target))
              for row in b.loc[b.split=="train"].itertuples()}
 w,weights,_=base.make_windows(frame,b)
 rid=window_record_ids(frame,b,"test")
 truth=w["test"][1].numpy()
 assert len(rid)==len(truth)==34691
 # PRE-FLIGHT counts frozen before model execution
 pairs_per_class={}
 for k in range(4):
  cls_set=set(rid[truth==k].tolist())
  seen=[r for r in cls_set if (r,k) in train_pairs]
  pairs_per_class[str(k)]={"seen_same_class_recordings":len(seen),
   "not_seen_same_class_recordings":len(cls_set)-len(seen)}
 print("A3 PREFLIGHT",pairs_per_class,flush=True)
 assert pairs_per_class["3"]=={"seen_same_class_recordings":1,
                               "not_seen_same_class_recordings":14}
 OUT.mkdir(exist_ok=True)
 manifest={"protocol_sha256":sha(protocol),"script_sha256":sha(Path(__file__)),
   "dataset_sha256":sha(base.CSV),"author_model_sha256":sha(HERE/"author_models.py"),
   "author_train_sha256":sha(HERE/"author_train.py"),
   "model_seeds":SEEDS,"same_class_train_seen_preflight":pairs_per_class,
   "split_row_counts":b.groupby("split").length.sum().to_dict(),
   "thresholds_true_class_windows":THRESHOLDS,
   "drop_last_train":False}
 save(OUT/"A3_PRETRAIN_MANIFEST.json",manifest)
 for seed in SEEDS:
  result_path=OUT/f"A3_seed{seed}.json"
  if result_path.exists():raise RuntimeError(f"Will not overwrite {result_path}")
  model,state,pred,meta=train_model(seed,w,weights)
  records,summary=evaluate(truth,pred,rid,train_pairs)
  ckpt=OUT/f"A3_checkpoint_seed{seed}.pt"
  torch.save({"state_dict":state,"selected_epoch":meta["selected_epoch"]},ckpt)
  result={"protocol_sha256":manifest["protocol_sha256"],"seed":seed,
    "train_meta":meta,"checkpoint_filename":ckpt.name,"checkpoint_sha256":sha(ckpt),
    "test_accuracy":float(accuracy_score(truth,pred)),
    "pooled_macro_f1":float(f1_score(truth,pred,average="macro")),
    "pooled_class_f1":f1_score(truth,pred,labels=[0,1,2,3],average=None).tolist(),
    "confusion":confusion_matrix(truth,pred,labels=[0,1,2,3]).tolist(),
    "by_record_class":records,"group_summaries":summary}
  save(result_path,result)
  print("A3 FINAL",seed,"pooledF1",[round(x,4) for x in result["pooled_class_f1"]],
    "class3_seen",summary["class3_min1_seenTrue"],"class3_unseen",summary["class3_min1_seenFalse"],flush=True)
 print("A3 DONE 3/3",flush=True)

if __name__=="__main__":
 main()
