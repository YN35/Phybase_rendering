import math
import torch

def sphere_gaussian(_v,_lobe_dir,sharpness,_intensity):
    """
    SGを計算する
    
    Parameters
    ----------
    _v : Tensor
        計算したい方向
    _lobe_dir : Tensor
        軸の向き ε
    sharpness : float
        山のとんがり具合 λ
    _intensity : Tensor
        山の高さ μ
        
    Returns
    -------
    _l : Tensor
    """
    return _intensity * math.e ** (sharpness * (torch.dot(_v,_lobe_dir) - 1))