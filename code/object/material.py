import torch
import math
import util
from object.shape import Shape

shape = Shape()
class Material():
    
    def __init__(self) -> None:
        self.dtype = torch.float
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.sharpness = 0.8
        self._intensity = torch.tensor([0.5,0.5,0.5],device=self.device,dtype=self.dtype)#反射率大きいほど反射する
        
    def albedo(self,_x):
        return torch.tensor([0.2,0.2,0.2],device=self.device,dtype=self.dtype) 
    
    def brdf(self,_omega_0,_omega_i,_x_reflect):
        """
        f_r:光の反射特性の計算
        
        Parameters
        ----------
        _omega_0 : Tensor
            カメラの座標
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
        roughness = 0.9
        _s = torch.tensor([0.2,0.2,0.2],device=self.device,dtype=self.dtype) 
        
        _nomal = shape.get_nomal(_x_reflect)
        
        _h = (_omega_0 + _omega_i) / torch.norm(_omega_0 + _omega_i)
        _F = _s + (1 - _s) * 2 ** - (5.55473 * torch.dot(_omega_0,_h)+6.8316) * torch.dot(_omega_0,_h)
        k = ((roughness + 1) ** 2) / 8
        _G = torch.dot(_omega_0,_nomal) / (torch.dot(_omega_0,_nomal) * (1 - k) + k) * torch.dot(_omega_i,_nomal) / (torch.dot(_omega_i,_nomal) * (1 - k) + k)
        _M = (_F * _G) / (4 * torch.dot(_omega_0,_nomal) * torch.dot(_omega_i,_nomal))
        #_D = 
        #_f_s = _M * _D
        _f_s = util.sphere_gaussian(_h,_nomal,self.sharpness / (torch.dot(4*_h,_omega_0)), _M * self._intensity)
        
        _f_r = (self.albedo(_x_reflect) / math.pi) + _f_s#############
        return _f_r
    

    def env_sphere_gaussian(self,i):
        return torch.tensor([0,0,-1],device=self.device,dtype=self.dtype), 0.7, torch.tensor([0.5,0.5,0.5],device=self.device,dtype=self.dtype)