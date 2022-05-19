import torch

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

a = torch.tensor([[1,2,3],[4,5,6]],device=device,dtype=dtype)
b = torch.tensor([1,2,3],device=device,dtype=dtype)

print(torch.mul(a,b))