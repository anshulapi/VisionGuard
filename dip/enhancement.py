"""Conservative image enhancement operations; images use RGB convention."""
import cv2, numpy as np

def _u8(im):
    a=np.asarray(im)
    if a.size==0: raise ValueError("empty image")
    return np.clip(a,0,255).astype(np.uint8)
def clahe_lab(image, clip_limit=2.0, tile_grid=(8,8)):
    a=_u8(image); clahe=cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=tile_grid)
    if a.ndim==2:
        return clahe.apply(a)
    if a.shape[2]==4:
        color,alpha=a[:,:,:3],a[:,:,3:4]
    else:
        color,alpha=a[:,:,:3],None
    lab=cv2.cvtColor(color,cv2.COLOR_RGB2LAB); l,aa,bb=cv2.split(lab)
    l=clahe.apply(l); out=cv2.cvtColor(cv2.merge((l,aa,bb)),cv2.COLOR_LAB2RGB)
    return np.concatenate((out,alpha),axis=2) if alpha is not None else out
def gamma_correct(image,gamma=1.0):
    if gamma<=0: raise ValueError("gamma must be positive")
    lut=np.clip((np.arange(256)/255.0)**gamma*255,0,255).astype(np.uint8)
    return cv2.LUT(_u8(image),lut)
def median_denoise(image,ksize=3):
    if ksize<3 or ksize%2==0: raise ValueError("ksize must be odd and >= 3")
    a=_u8(image)
    if a.ndim==3 and a.shape[2]==4:
        return np.dstack([cv2.medianBlur(a[:,:,i],ksize) for i in range(3)]+[a[:,:,3]])
    return cv2.medianBlur(a,ksize)
def bilateral_denoise(image,d=7,sigma_color=35,sigma_space=35):
    a=_u8(image)
    if a.ndim==3 and a.shape[2]==4:
        filtered=cv2.bilateralFilter(a[:,:,:3],d,sigma_color,sigma_space)
        return np.dstack((filtered,a[:,:,3]))
    return cv2.bilateralFilter(a,d,sigma_color,sigma_space)
def nlm_denoise(image,h=5,h_color=5,template_window=7,search_window=21):
    """Edge-preserving non-local-means denoising for moderate colour noise."""
    a=_u8(image)
    if a.ndim==2:
        return cv2.fastNlMeansDenoising(a,None,h,template_window,search_window)
    if a.shape[2]==4:
        filtered=cv2.fastNlMeansDenoisingColored(a[:,:,:3],None,h,h_color,template_window,search_window)
        return np.dstack((filtered,a[:,:,3]))
    return cv2.fastNlMeansDenoisingColored(a,None,h,h_color,template_window,search_window)
def unsharp_mask(image,amount=.6,radius=1.2):
    a=_u8(image); blur=cv2.GaussianBlur(a,(0,0),radius)
    return np.clip(a.astype(float)+amount*(a.astype(float)-blur.astype(float)),0,255).astype(np.uint8)
def white_balance(image,strength=.35):
    source=_u8(image)
    if source.ndim==2:
        return source.copy()
    alpha=source[:,:,3:4] if source.shape[2]==4 else None
    a=source[:,:,:3].astype(np.float32)
    # A robust border estimate reduces bias from the product's own dominant colour.
    border=np.concatenate((a[0],a[-1],a[:,0],a[:,-1]),axis=0)
    means=np.median(border,axis=0); gray=float(np.mean(means))
    gain=np.clip(gray/(means+1e-6),.85,1.18); out=a*(1-strength+strength*gain)
    balanced=np.clip(out,0,255).astype(np.uint8)
    return np.concatenate((balanced,alpha),axis=2) if alpha is not None else balanced
def enhance(image, operations, output_dir=None):
    """Apply named operations sequentially and optionally write PNG intermediates."""
    from pathlib import Path
    out=_u8(image).copy(); saved={"original":out.copy()}
    funcs={"clahe":clahe_lab,"gamma":gamma_correct,"median":median_denoise,"bilateral":bilateral_denoise,"nlm":nlm_denoise,"sharpen":unsharp_mask,"white_balance":white_balance}
    for i,op in enumerate(operations,1):
        name=op if isinstance(op,str) else op["name"]; kwargs={} if isinstance(op,str) else op.get("kwargs",{})
        if name not in funcs: raise ValueError(f"Unknown operation: {name}")
        out=funcs[name](out,**kwargs); saved[f"{i}_{name}"]=out.copy()
    if output_dir:
        p=Path(output_dir); p.mkdir(parents=True,exist_ok=True)
        for n,img in saved.items(): cv2.imwrite(str(p/f"{n}.png"),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    return out,saved
