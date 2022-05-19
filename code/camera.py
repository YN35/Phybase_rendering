from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import torch

class Camera():
    
    def __init__(self) -> None:
        self.dtype = torch.float
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def rendering(self):
    
    def get_pixel_color(self,_x_cam,_omega_0,ray_num):
        """

        Parameters
        ----------
        _x_cam : Tensor
            カメラの座標
        _omega_0 : Tensor
            カメラの向き
        ray_num : float
            入ってくる光を計算する方向を計算する数(ray_num^2)
        """
        delta = 
        
        #反射面の座標を算出
        _x_reflect = ray_marching(_x_cam,_omega_0)
        _nomal_surface = get_nomal(_x_reflect)
        
        #90度回転
        _omega_i = torch.mul(torch.tensor([[0,-1,0],[1,0,0],[0,0,1]],device=self.device,dtype=self.dtype),_nomal_surface)
        for x in range(ray_num):
            _omega_i = torch.mul(torch.tensor([[0,-1,0],[1,0,0],[0,0,1]],device=self.device,dtype=self.dtype),_omega_i) 
            for y in range(ray_num):
        
        
        
        
        
        
        
        
    
    
    