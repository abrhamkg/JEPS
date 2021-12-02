import torch

a = torch.zeros(4, 5)
b = torch.randn(3, 5)
c = torch.ones(3, 4)
d = torch.ones(1, 5)
e = torch.randn(4, 5)
f = torch.randn(3, 1)
g = torch.randn(4)

# Add a and e componenet wise
i = a + e
# Subtract 8 from all elements of b
j = b - 8

print("Tensor g\n", g)

column_sum_e = e.sum(dim=1)
row_sum_e = e.sum(dim=0)

print("Column sum e shape", column_sum_e.shape)
print("Row sum e shape", row_sum_e.shape)

# Find the "maximum row": Row with the maximum value for each colum

row_max_e = e.max(dim=0)
column_max_e = e.max(dim=1)

print("Column sum e shape", column_max_e)
print("Row sum e shape", row_max_e)

# Torch cat

a_b_cat_dim0  = torch.cat([a, b], 0)
b_c_cat_dim1 = torch.cat([b, c], 1)

print()
print("Column sum e shape", a_b_cat_dim0.shape)
print("Row sum e shape", b_c_cat_dim1.shape)

# Broadcasting

broadcasting_1 = c + g
broadcasting_2 = f * b

print("broadcasting_1 shape", broadcasting_1.shape)
print("broadcasting_2 shape", broadcasting_2.shape)

