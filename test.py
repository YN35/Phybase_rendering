import torch

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

a = torch.tensor([1,2,3],device=device,dtype=dtype)
b = torch.tensor([1,2,3],device=device,dtype=dtype)

#掛け算
print(torch.mul(a,b))
print(a*4)
print(a*b)
print(a**b)
#内積
print(torch.dot(a,b))

print('ノルム',torch.norm(a))
#引き算
print(torch.sub(a,b))
print(a-b)
print(a+b)

print(b[0])