import math
import torch
import numpy as np

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


def normalize(_vec):
    return _vec / torch.norm(_vec)

def angle(_vec1, _vec2):
    """
    ２つのベクトルがなす角を計算
    vec2 - vec1
    """
    theta = math.acos(np.dot(_vec1, _vec2) / (np.linalg.norm(_vec1)*np.linalg.norm(_vec2)))
    return theta


def plot_half_sphere(N):
    f = (np.sqrt(5)-1)/2
    arr = np.linspace(0, N, N)
    theta = np.arcsin(arr/N)
    phi = 2*np.pi*arr*f
    x = np.cos(theta)*np.cos(phi)
    y = np.cos(theta)*np.sin(phi)
    z = np.sin(theta)
    plots = np.array([x,y,z])
    return plots

def rotate_vec(_vec, _lobe, theta):
    c = math.cos(theta)
    s = math.sin(theta)
    n_1 = _lobe[0]
    n_2 = _lobe[1]
    n_3 = _lobe[2]
    _vec = np.array([[c+(n_1**2)*(1-c), n_1*n_2*(1-c)-n_3*s, n_1*n_3*(1-c)+n_2*s],
                     [ n_2*n_1*(1-c)+n_3*s,c+(n_2**2)*(1-c), n_2*n_3*(1-c)-n_1*s],
                     [ n_3*n_1*(1-c)-n_2*s, n_3*n_2*(1-c)+n_1*s,c+(n_3**2)*(1-c)]]) @ _vec
    return _vec