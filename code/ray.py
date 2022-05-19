import torch

from object.shape import Shape

shape = Shape()

class Ray():
    
    def __init__(self) -> None:
        pass
    
    def ray_marching(self,_x_cam,_cam_dir,epsilon=0.001):
        """
        光線と面が交差する点を求める
        
        Parameters
        ----------
        _x_cam : Tensor
            カメラの座標
        _cam_dir : Tensor
            カメラの向き
        epsilon : float (0.001)
            求める精度、許す誤差
            
        Returns
        -------
            _x : 光線と面が交差する点
        """
        sdf = shape.sdf(_x_cam)
        _x = _x_cam + _cam_dir * sdf
        sdf = shape.sdf(_x)
        
        while sdf > epsilon:
            _x = _x + (_cam_dir * sdf)
            sdf = shape.sdf(_x)
            
        return _x
        
    
    def incident_light(self,_omega_i,num_sg):
        """
        任意の方向からの入射光を計算
        
        Parameters
        ----------
        _omega_i : Tensor
            求めたい方向
        num_sg : int
            sgの数
            
        Returns
        -------
            _l_i : RGBの入射光のデータ
        """
        for i in range(num_sg):
            sphere_gaussian(_omega_i)