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
        self.width = 300
        self.hight = 200
        self.fov = 90
        self._main_cam_dir = torch.tensor([5,5,5],device=self.device,dtype=self.dtype)
        self._main_cam_dir = self._main_cam_dir / torch.norm(self._main_cam_dir)
        self._view_up = torch.tensor([1,0,0],device=self.device,dtype=self.dtype)#カメラをどっちを上とするか
        self.void = torch.tensor([0,0,0],device=self.device,dtype=self.dtype)
        self.num_sg = 1
    
    def render(self):
        """
        dads
        """
        w_ini = math.sin(math.radians(self.fov / 2))
        w = w_ini
        h = w_ini * (self.hight / self.width)
        w_delta = w_ini / (self.hight / 2)
        h_delta = w_ini / (self.width / 2)
        # _cam_dir = torch.tensor([-1,-1,-1],device=self.device,dtype=self.dtype)
        # _cam_dir = _cam_dir / torch.norm(_cam_dir)
        # print(self.get_pixel_color(torch.tensor([5,5,5],device=self.device,dtype=self.dtype), _cam_dir, 1000))
        for x_n in range(self.width):
            for y_n in range(self.hight):
                
            w = w + w_delta
            h = h + h_delta
                
        
            
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
        if _x_reflect == 'nothing':
            return self.void
        
        _nomal_surface = shape.get_nomal(_x_reflect)
        #面から見たカメラの方向
        _omega_0 = _cam_dir - _x_reflect
        
        phi_lis = np.random.rand(ray_num) * math.pi
        z_lis = np.random.rand(ray_num)
        for i in range(ray_num):
            _omega_i = torch.tensor([math.sqrt(1-z_lis[i]**2) * math.cos(math.radians(phi_lis[i])),
                                     math.sqrt(1-z_lis[i]**2) * math.sin(math.radians(phi_lis[i])),
                                     z_lis[i]],device=self.device,dtype=self.dtype)
            
            sigma = ray.incident_light(_omega_i,self.num_sg) * mate.brdf(_omega_0,_omega_i,_x_reflect) * torch.dot(_omega_i,_nomal_surface) + sigma
            
        return sigma
        
        
        
        
        
    
    
    