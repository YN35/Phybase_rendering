import numpy as np
import torch
import math
from ray import Ray
from object.material import Material
from object.shape import Shape

ray = Ray()
mate = Material()
shape = Shape()

class Camera():
    
    def __init__(self) -> None:
        self.dtype = torch.float
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def render(self):
        pass
    
    def get_pixel_color(self,_x_cam,_cam_dir,ray_num):
        """

        Parameters
        ----------
        _x_cam : Tensor
            カメラの座標
        _cam_dir : Tensor
            カメラの向き
        ray_num : float
            入ってくる光を計算する方向を計算する数(ray_num)
        """
        sigma = 0
        
        #反射面の座標を算出
        _x_reflect = ray.ray_marching(_x_cam,_cam_dir)
        _nomal_surface = shape.get_nomal(_x_reflect)
        #面から見たカメラの方向
        _omega_0 = _cam_dir - _x_reflect
        
        phi_lis = np.rand(ray_num) * math.pi
        z_lis = np.rand(ray_num)
        
        #90度回転
        for i in range(ray_num):
            _omega_i = torch.tensor([math.sqrt(1-z_lis[i]**2) * math.cos(math.radians(phi_lis[i])),
                                     math.sqrt(1-z_lis[i]**2) * math.sin(math.radians(phi_lis[i])),
                                     z_lis[i]],device=self.device,dtype=self.dtype)
            
            sigma = ray.incident_light(_omega_i) * mate.brdf(_omega_0,_omega_i) * torch.dot(_omega_i,_nomal_surface) + sigma
            
        return sigma
        
        
        
        
        
    
    
    