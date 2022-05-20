import torch
import util
from object.shape import Shape
from object.material import Material

shape = Shape()
mate = Material()

class Ray():
    
    def __init__(self) -> None:
        self.dtype = torch.float
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def ray_marching(self,_x_cam,_cam_dir,epsilon=0.001):
        """
        光線と面が交差する点を求める
        
        Parameters
        ----------
        _x_cam : Tensor
            カメラの座標
        _cam_dir : Tensor
            カメラの向き(単位ベクトル)
        epsilon : float (0.001)
            求める精度、許す誤差
            
        Returns
        -------
            _x : 光線と面が交差する点
        """
        sdf = shape.sdf(_x_cam)
        _x = _x_cam + (_cam_dir * sdf)
        sdf = shape.sdf(_x)
        
        while sdf > epsilon:
            _x = _x + (_cam_dir * sdf)
            sdf = shape.sdf(_x)
            if sdf == torch.tensor(float('inf'),device=self.device,dtype=self.dtype):
                return 'nothing'
            
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
        _l_i = torch.tensor([0,0,0],device=self.device,dtype=self.dtype)
        for i in range(num_sg):
            _lobe_dir, sharpness, _intensity = mate.env_sphere_gaussian(i)
            _l_i = util.sphere_gaussian(_omega_i, _lobe_dir, sharpness, _intensity) + _l_i
        
        return _l_i