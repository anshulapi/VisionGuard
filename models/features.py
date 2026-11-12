import numpy as np
from dip.quality import analyze_quality
from dip.segmentation import segment_product
def extract_features(image):
    q=analyze_quality(image); s=segment_product(image,refine=False)
    return np.array([q["laplacian_variance"],q["tenengrad"],q["mean_brightness"],q["clipping_fraction"],q["rms_contrast"],q["noise_sigma"],s["coverage"],s["centering_score"],s["background_std"]],dtype=float)
