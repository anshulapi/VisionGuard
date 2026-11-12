"""Generate and inspect a reproducible sample through the complete DIP pipeline."""
from pathlib import Path
import argparse
import json, numpy as np
import pandas as pd
from PIL import Image
from data.generate_dataset import generate_dataset
from agent.planner import auto_fix
from dip.enhancement import clahe_lab
from dip.quality import QualityThresholds
ROOT=Path(__file__).resolve().parents[1]
def _resolve(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p
def main(manifest=None):
    if manifest:
        manifest_path=Path(manifest).resolve(); frame=pd.read_csv(manifest_path)
        test=frame[frame["split"]=="test"]
        if test.empty: raise ValueError("Manifest has no test sample for the demo")
        sample=test.iloc[0]; source_path=_resolve(sample["image_path"])
    else:
        manifest_path=generate_dataset(groups=60)
        frame=pd.read_csv(manifest_path); sample=frame[frame["split"]=="test"].iloc[0]
        source_path=_resolve(sample["image_path"])
    image=np.asarray(Image.open(source_path).convert("RGB")); fixed,report=auto_fix(image)
    out=ROOT/"outputs/demo"; out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(image).save(out/"original.png"); Image.fromarray(fixed).save(out/"corrected.png")
    seg=report["before"]["segmentation"]; Image.fromarray(seg["mask"]).save(out/"mask.png")
    previews=out/"intermediates"; previews.mkdir(parents=True,exist_ok=True)
    for stale in previews.glob("*.png"):
        stale.unlink()
    Image.fromarray(image).save(previews/"original.png")
    preview_notes=[]
    q=report["before"]["quality"]
    if "low_contrast" in q["flags"] and q["noise_sigma"] < QualityThresholds().max_noise*.75:
        Image.fromarray(clahe_lab(image,clip_limit=1.2)).save(previews/"clahe_diagnostic.png")
        preview_notes.append("CLAHE diagnostic preview only; not the corrected output.")
    else:
        preview_notes.append("CLAHE preview omitted because low contrast was not diagnosed or measured noise was too high.")
    clean_report={**report,
      "before":{"quality":report["before"]["quality"],"compliance":report["before"]["compliance"],"segmentation":{k:v for k,v in seg.items() if k!="mask"}},
      "after":{"quality":report["after"]["quality"],"compliance":report["after"]["compliance"],"segmentation":{k:v for k,v in report["after"]["segmentation"].items() if k!="mask"}},
      "intermediate_previews":preview_notes}
    (out/"inspection.json").write_text(json.dumps(clean_report,indent=2))
    print(f"Demo outputs: {out}")
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Use an existing dataset manifest without regenerating samples")
    main(parser.parse_args().manifest)
