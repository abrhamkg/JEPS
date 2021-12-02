import torch

tensor_of_zeros = torch.zeros(3)

print("Tensor of zeros:\n", tensor_of_zeros)

tensor_3_by_4 = torch.zeros(3, 4)

print("Tensor 3x4:\n", tensor_3_by_4)

tensor_2_by_3_by_1 = torch.zeros(2, 3, 1)

print("Tensor 2x3x1:\n", tensor_2_by_3_by_1)

print("Tensor of zeros shape:\n", tensor_of_zeros.shape)
print("Tensor 3x4 shape:\n", tensor_3_by_4.shape)
print("Tensor 2x3x1 shape:\n", tensor_2_by_3_by_1.shape)

print("Tensor of zeros ndim:\n", tensor_of_zeros.ndim)
print("Tensor 3x4 ndim:\n", tensor_3_by_4.ndim)
print("Tensor 2x3x1 ndim:\n", tensor_2_by_3_by_1.ndim)

# A tensor of ones maybe created by replacing torch.zeros with torch.ones
# A tensor of random normal values maybe created by replacing torch.zeros with torch.randn

print("Tensor of zeros dtype:\n", tensor_of_zeros.dtype)

tensor_of_int_zeros = torch.zeros(3, dtype=int)

print("Tensor of int zeros dtype:\n", tensor_of_int_zeros.dtype)

tensor_of_zeros_again = tensor_of_int_zeros.float()

print("Tensor of zeros (cast) dtype:\n", tensor_of_zeros_again.dtype)

