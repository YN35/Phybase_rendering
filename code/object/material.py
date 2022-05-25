import imp
import torch
import math
import util
from object.shape import Shape
import numpy as np

shape = Shape()
class Material():
    
    def __init__(self) -> None:
        self.dtype = torch.float
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.sharpness = 0.9
        self._intensity = torch.tensor([0.5,0.5,0.5],device=self.device,dtype=self.dtype)#反射率大きいほど反射する
        
    def albedo(self,_x):
        return torch.tensor([0.2,0.2,0.2],device=self.device,dtype=self.dtype) 
    
    def brdf(self,_omega_0,_omega_i,_x_reflect):
        """
        f_r:光の反射特性の計算
        
        Parameters
        ----------
        _omega_0 : Tensor
            カメラの方向(面からの)
        _omega_i : Tensor
            計算している入射光の方向
        _x_reflect : Tensor
            反射面の座標
            
        Returns
        -------
        _x : Tensor
            f_r
        """
        #1に近いほど金属っぽくてかてかする
        roughness = 0.5
        _s = torch.tensor([0.2,0.2,0.2],device=self.device,dtype=self.dtype)
        
        _nomal = shape.get_nomal(_x_reflect)
        _nomal = util.normalize(_nomal)
        _omega_i = _omega_i+0.0001 if torch.dot(_omega_i,_nomal)==0 else _omega_i
        
        _h = (_omega_0 + _omega_i) / torch.norm(_omega_0 + _omega_i)
        _F = _s + (1 - _s) * 2 **-((5.55473 * torch.dot(_omega_0,_h)+6.8316) * torch.dot(_omega_0,_h))
        k = ((roughness + 1)**2) / 8
        _G = (torch.dot(_omega_0,_nomal) / (torch.dot(_omega_0,_nomal)*(1 - k) + k)) * (torch.dot(_omega_i,_nomal) / (torch.dot(_omega_i,_nomal)*(1 - k) + k))
        _M = (_F * _G) / (4 * torch.dot(_nomal,_omega_0) * torch.dot(_nomal,_omega_i))
        _D =  util.sphere_gaussian(_h,_nomal,2/(roughness**4),1/(math.pi * roughness**4))
        _f_s = _M * _D
        # _f_s = util.sphere_gaussian(_h,_nomal,self.sharpness / (torch.dot(4*_h,_omega_0)), _M * self._intensity)
        
        _f_r = (self.albedo(_x_reflect) / math.pi) + _f_s#############
        if np.count_nonzero(np.isnan(_f_r.to('cpu').detach().numpy().copy())) > 0 or np.count_nonzero(np.isinf(_f_r.to('cpu').detach().numpy().copy())) > 0:
            print('dsadjaskdklja')
        return _f_r
    

    def env_sphere_gaussian(self,i):
        #_lobe_dir(照明が存在する方向　その座標から光が飛んでくる), sharpness, _intensity
        return torch.tensor([-1,-1,-1],device=self.device,dtype=self.dtype), 0.9, torch.tensor([1,1,1],device=self.device,dtype=self.dtype)