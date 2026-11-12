"""Create deterministic, grouped synthetic product images and defect labels."""
import argparse, csv, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

ROOT=Path(__file__).resolve().parents[1]
LABELS=["blur","noise","exposure","colorcast","low_contrast"]
SPLITS=("train","validation","test")
OWNER_FILE=".visionguard_dataset.json"
def make_base(seed,size=256):
    rng=np.random.default_rng(seed); im=Image.new("RGB",(size,size),(246,246,246)); d=ImageDraw.Draw(im)
    kind=seed%4; col=tuple(int(x) for x in rng.integers(35,220,3)); cx=128+rng.integers(-12,13); cy=128+rng.integers(-12,13)
    if kind==0: d.rounded_rectangle((cx-48,cy-60,cx+48,cy+60),radius=14,fill=col,outline=(30,30,30),width=3); d.ellipse((cx-24,cy-35,cx+24,cy+5),fill=tuple(255-x for x in col))
    elif kind==1: d.ellipse((cx-55,cy-55,cx+55,cy+55),fill=col,outline=(30,30,30),width=3); d.ellipse((cx-23,cy-23,cx+23,cy+23),fill=(235,225,200))
    elif kind==2: d.polygon([(cx,cy-65),(cx+54,cy+48),(cx-54,cy+48)],fill=col,outline=(30,30,30)); d.rectangle((cx-18,cy+5,cx+18,cy+48),fill=(245,210,95))
    else: d.rounded_rectangle((cx-60,cy-35,cx+60,cy+35),radius=22,fill=col,outline=(25,25,25),width=3); d.line((cx-40,cy,cx+40,cy),fill=(245,245,245),width=5)
    return im
def apply_defect(im,label,rng):
    a=np.asarray(im).astype(np.float32)
    if label=="blur": return im.filter(ImageFilter.GaussianBlur(2.2))
    if label=="noise": return Image.fromarray(np.uint8(np.clip(a+rng.normal(0,20,a.shape),0,255)))
    if label=="exposure": return Image.fromarray(np.uint8(np.clip(a*.48,0,255)))
    if label=="colorcast": a[:,:,0]*=1.35; return Image.fromarray(np.uint8(np.clip(a,0,255)))
    if label=="low_contrast": return ImageEnhance.Contrast(im).enhance(.35)
    return im
def _manifest_path(path):
    resolved=Path(path).resolve()
    try: return str(resolved.relative_to(ROOT.resolve()))
    except ValueError: return str(resolved)
def _prepare_output(root, default_root):
    """Reset only a dedicated VisionGuard data directory; refuse unknown contents."""
    root.mkdir(parents=True,exist_ok=True)
    owner=root/OWNER_FILE
    is_default=root.resolve()==default_root.resolve()
    if not is_default and not owner.is_file() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty unowned dataset directory: {root}")
    for split in SPLITS:
        path=root/split
        if path.is_dir(): shutil.rmtree(path)
        elif path.exists(): path.unlink()
    manifest=root/"manifest.csv"
    if manifest.exists(): manifest.unlink()
    if owner.exists(): owner.unlink()
def generate_dataset(out_dir=None,groups=60,seed=2026,real_dir=None):
    default_root=(ROOT/"data"/"generated").resolve()
    root=Path(out_dir or default_root).resolve(); _prepare_output(root,default_root); rows=[]
    originals=[]
    if real_dir:
        for p in sorted(Path(real_dir).glob("*")):
            try: originals.append((p.stem,Image.open(p).convert("RGB").resize((256,256))))
            except Exception: continue
    if not originals: originals=[(f"synthetic_{i:04d}",make_base(i)) for i in range(groups)]
    rng=np.random.default_rng(seed)
    assignments={}; strata=[]
    for stratum in range(6):
        members=np.array([i for i in range(len(originals)) if i%6==stratum]); rng.shuffle(members)
        if len(members): strata.append(members)
    test_quota={j:1 for j in range(len(strata))}; val_quota={j:0 for j in range(len(strata))}
    ntest=min(len(originals)-len(strata),max(len(strata),round(len(originals)*.2)))
    while sum(test_quota.values()) < ntest:
        eligible = [j for j in range(len(strata)) if test_quota[j] < len(strata[j]) - 1]
        if not eligible:
            break
        j = min(eligible, key=lambda i: test_quota[i] / len(strata[i]))
        test_quota[j] += 1
    nval=min(len(originals)-ntest-len(strata),round(len(originals)*.1))
    while sum(val_quota.values()) < nval:
        eligible = [j for j in range(len(strata)) if test_quota[j] + val_quota[j] < len(strata[j]) - 1]
        if not eligible:
            break
        j = min(eligible, key=lambda i: val_quota[i] / len(strata[i]))
        val_quota[j] += 1
    for j,members in enumerate(strata):
        ntest_j=test_quota[j]; nval_j=val_quota[j]
        for rank,idx in enumerate(members):
            assignments[int(idx)]="test" if rank<ntest_j else "validation" if rank<ntest_j+nval_j else "train"
    for i,(gid,clean) in enumerate(originals):
        split=assignments[i]
        name=f"{gid}_{i:04d}"; od=root/split/name; od.mkdir(parents=True,exist_ok=True); clean.save(od/"clean.png")
        labels=[]
        # Every source has a clean record and one randomly selected defect, no split leakage.
        if i%6: labels=[LABELS[(i-1)%len(LABELS)]]
        defect=clean.copy()
        for label in labels: defect=apply_defect(defect,label,rng)
        defect.save(od/"input.png")
        rows.append({"group_id":gid,"split":split,"clean_path":_manifest_path(od/"clean.png"),"image_path":_manifest_path(od/"input.png"),**{f"label_{x}":int(x in labels) for x in LABELS}})
    with (root/"manifest.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (root/OWNER_FILE).write_text(json.dumps({"generator":"visionguard","seed":int(seed),"groups":len(rows)},indent=2))
    print(f"Generated {len(rows)} grouped sources at {root}; split counts: "+str({s:sum(r['split']==s for r in rows) for s in ("train","validation","test")}))
    return root/"manifest.csv"
def main():
    p=argparse.ArgumentParser(); p.add_argument("--groups",type=int,default=60); p.add_argument("--seed",type=int,default=2026); p.add_argument("--real-dir"); p.add_argument("--out-dir"); a=p.parse_args(); generate_dataset(a.out_dir,a.groups,a.seed,a.real_dir)
if __name__=="__main__": main()
