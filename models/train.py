"""Train a group-separated RF and compare with deterministic DIP flags."""
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from models.features import extract_features
from dip.quality import analyze_quality
ROOT=Path(__file__).resolve().parents[1]; TARGETS=["blur","noise","exposure","colorcast","low_contrast"]
def train(manifest=None):
    m=pd.read_csv(manifest or ROOT/"data/generated/manifest.csv"); X=[]
    for p in m.image_path: X.append(extract_features(np.asarray(Image.open(ROOT/p).convert("RGB"))))
    X=np.asarray(X); y=m[[f"label_{t}" for t in TARGETS]].values
    tr=np.where(m.split.values=="train")[0]; va=np.where(m.split.values=="validation")[0]; te=np.where(m.split.values=="test")[0]
    model=RandomForestClassifier(n_estimators=120,max_depth=8,class_weight="balanced",random_state=2026,n_jobs=1)
    model.fit(X[tr],y[tr]); pred=model.predict(X[te]).astype(int)
    base=np.zeros_like(pred)
    for j,t in enumerate(TARGETS):
        if t in ("blur","noise","exposure","low_contrast","colorcast"):
            mapping={"blur":"blur","noise":"noise","exposure":"exposure","low_contrast":"low_contrast","colorcast":"colorcast"}
            for k,i in enumerate(te): base[k,j]=int(mapping[t] in analyze_quality(np.asarray(Image.open(ROOT/m.iloc[i].image_path).convert("RGB")))["flags"])
    out=ROOT/"outputs/model"; out.mkdir(parents=True,exist_ok=True); joblib.dump({"model":model,"targets":TARGETS,"feature_names":["laplacian","tenengrad","brightness","clipping","contrast","noise","coverage","centering","bg_std"]},out/"visionguard_rf.joblib")
    result={"n_train":len(tr),"n_validation":len(va),"n_test":len(te),"model_macro_f1":float(f1_score(y[te],pred,average="macro",zero_division=0)),"model_micro_f1":float(f1_score(y[te],pred,average="micro",zero_division=0)),"baseline_macro_f1":float(f1_score(y[te],base,average="macro",zero_division=0)),"baseline_micro_f1":float(f1_score(y[te],base,average="micro",zero_division=0)),"model_report":classification_report(y[te],pred,target_names=TARGETS,zero_division=0,output_dict=True),"baseline_report":classification_report(y[te],base,target_names=TARGETS,zero_division=0,output_dict=True)}
    (out/"metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if "report" not in k},indent=2)); return result
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Dataset manifest path (defaults to data/generated/manifest.csv)")
    train(parser.parse_args().manifest)
