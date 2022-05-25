import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def angle(_vec1, _vec2):
    """
    ２つのベクトルがなす角を計算
    vec2 - vec1
    """
    theta = math.acos(np.dot(_vec1, _vec2) / (np.linalg.norm(_vec1)*np.linalg.norm(_vec2)))
    return theta


def plot_half_sphere(N):
    f = (np.sqrt(5)-1)/2
    arr = np.linspace(0, N, N)
    theta = np.arcsin(arr/N)
    phi = 2*np.pi*arr*f
    x = np.cos(theta)*np.cos(phi)
    y = np.cos(theta)*np.sin(phi)
    z = np.sin(theta)
    plots = np.array([x,y,z])
    return plots

def rotate_vec(_vec, _lobe, theta):
    c = math.cos(theta)
    s = math.sin(theta)
    n_1 = _lobe[0]
    n_2 = _lobe[1]
    n_3 = _lobe[2]
    _vec = np.array([[c+(n_1**2)*(1-c), n_1*n_2*(1-c)-n_3*s, n_1*n_3*(1-c)+n_2*s],
                     [ n_2*n_1*(1-c)+n_3*s,c+(n_2**2)*(1-c), n_2*n_3*(1-c)-n_1*s],
                     [ n_3*n_1*(1-c)-n_2*s, n_3*n_2*(1-c)+n_1*s,c+(n_3**2)*(1-c)]]) @ _vec
    return _vec


plots = plot_half_sphere(1000)

nomal = np.array([0.2,0.4,0.5])
nomal = nomal / np.linalg.norm(nomal)
theta = -angle(nomal,np.array([0,0,1]))
print(theta)

plots = rotate_vec(plots, np.cross(nomal,np.array([0,0,1])), theta)

for i in range(1000):
    print(np.linalg.norm(plots[:,i]))

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((2,2,2))
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.plot(plots[0,:],plots[1,:],plots[2,:], "o", ms=2, mew=0.5)
ax.quiver(0, 0, 0, nomal[0], nomal[1], nomal[2],
          color = "red", length = 1,
          arrow_length_ratio = 0.1)
ax.quiver(0, 0, 0, 0, 0, 1,
          color = "red", length = 1,
          arrow_length_ratio = 0.1)
ax.quiver(0, 0, 0,  np.cross(nomal,np.array([0,0,1]))[0],  np.cross(nomal,np.array([0,0,1]))[1],  np.cross(nomal,np.array([0,0,1]))[2],
          color = "red", length = 1,
          arrow_length_ratio = 0.1)
plt.show()  