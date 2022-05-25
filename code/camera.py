import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from PIL import Image
import util
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
        self.width = 1000
        self.hight = 500
        self.fov = 70
        self._cam_location = torch.tensor([5,0,0],device=self.device,dtype=self.dtype)
        self._cam_dir = torch.tensor([-1,0,0],device=self.device,dtype=self.dtype)
        self._view_up = torch.tensor([0,0,1],device=self.device,dtype=self.dtype)#カメラをどっちを上とするか
        self.void = torch.tensor([0,0,0],device=self.device,dtype=self.dtype)
        self.num_sg = 1
        
    def save_image(self,out_image):
        
        Image.fromarray(out_image.astype(np.uint8)).save("out_image.bmp")
        np.save('out_image', out_image)
        
    def filter(self,out_image):
        
        q90, q10 = np.percentile(out_image, [99 ,1])
        iqr = q90 - q10
        out_image = np.where(out_image == 0., q10 - (iqr/4), out_image)
        out_image = np.where(out_image > q90 + (iqr/4), q90 + (iqr/4), out_image)
        out_image = np.where(out_image < q10 - (iqr/4), q10 - (iqr/4), out_image)
        out_image[0,0,:] = q90 + (iqr/4)
        out_image = out_image - np.min(out_image)
        out_image = out_image * (255 / np.max(out_image))
        
        return out_image
    
    
    def render(self):
        """
        dads
        """
        # _cam_dir = torch.tensor([-1,-1,-1],device=self.device,dtype=self.dtype)
        # _cam_dir = _cam_dir / torch.norm(_cam_dir)
        # print(self.get_pixel_color(torch.tensor([5,5,5],device=self.device,dtype=self.dtype), _cam_dir, 1000))
        out_image = np.zeros((self.hight,self.width,3))
        w = math.tan(math.radians(self.fov/2))
        h = w * (self.hight / self.width)
        _Z =  -(self._cam_dir / torch.norm(self._cam_dir))
        _X = torch.cross(self._view_up,_Z)
        _Y = torch.cross(_Z,_X)
        _u = 2 * w * _X
        _v = 2 * h * _Y
        _w = - w*_X - h*_Y - _Z
        for x in range(self.width):
            for y in range(self.hight):
                
                # x = 80
                # y = 45
            
                _p = _u * (x / self.width) + _v * (y / self.hight) + _w
                out_image[y,x,:] = self.get_pixel_color(self._cam_location, _p, 10).to('cpu').detach().numpy().copy()
            print(x,y,out_image[y,x,:],_p)
        
        # np.save('raw_out_image', out_image)
        return out_image
        
            
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
        
        #反射面の座標を算出
        _x_reflect = ray.ray_marching(_x_cam,_cam_dir)
        if _x_reflect == 'nothing':
            return self.void
        
        _nomal_surface = shape.get_nomal(_x_reflect)
        _nomal_surface_num = _nomal_surface.to('cpu').detach().numpy().copy()
        #面から見たカメラの方向
        _omega_0 = _x_cam - _x_reflect
        _omega_0 = util.normalize(_omega_0)
        
        #半球状にプロット
        plots = util.plot_half_sphere(ray_num)
        theta = -util.angle(_nomal_surface_num,np.array([0,0,1]))
        plots = util.rotate_vec(plots, np.cross(_nomal_surface_num,np.array([0,0,1])), theta)
        
        for i in range(ray_num):
            plots[:,i] = plots[:,i] / np.linalg.norm(plots[:,i])
        
        sigma = 0
        for i in range(ray_num):
            _omega_i = torch.tensor([plots[0,i],plots[1,i],plots[2,i]],device=self.device,dtype=self.dtype)
            sigma = ray.incident_light(_omega_i,self.num_sg) * mate.brdf(_omega_0,_omega_i,_x_reflect) * torch.dot(_omega_i,_nomal_surface) + sigma
            
            if np.count_nonzero(np.isnan(sigma.to('cpu').detach().numpy().copy())) > 0 or np.count_nonzero(np.isinf(sigma.to('cpu').detach().numpy().copy())) > 0:
                print('dsadjaskdklja')
            
        return sigma / ray_num
        
        
        
        
        
    
    
    