import torch

class Shape():
    
    def __init__(self) -> None:
        pass
    
    def sdf(self,_x):
        return torch.norm(_x) - 1
    
    def get_nomal(self,_x):
        """
        SDFを微分した関数により任意の面における法線を計算する
        任意座標における最も近い面の法線を返す(任意の座標における面からの距離が変化する方向を計算する)
        
        Parameters
        ----------
        _x : Tensor
            法線を計算したい面の座標
        
        Returns
        -------
        _nomal : Tensor
            法線 正規化済み
        """
        return _x / torch.norm(_x)
        