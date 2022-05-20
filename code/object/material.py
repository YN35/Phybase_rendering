import torch
import util
from object.shape import Shape

shape = Shape()
class Material():
    
    def __init__(self) -> None:
        self._lobe_dir = 
        self.sharpness = 
        self._intensity = 
    
    def brdf(self,_omega_0,_omega_i):
        """
        f_r:光の反射特性の計算
        
        Parameters
        ----------
        _omega_0 : Tensor
            カメラの座標
        _omega_i : Tensor
            計算している入射光の方向
            
        Returns
        -------
        _x : Tensor
            f_r
        """
        _h = (_omega_0 + _omega_i) / torch.norm(_omega_0 + _omega_i)
        _M_x = 
        
        _f_s = util.sphere_gaussian(_h,get_,self.sharpness / (torch.dot(4*_h,_omega_0)), _M_x * self._intensity)
        return 
    

    def env_sphere_gaussian(self,_omega_i,i):
        pass
        