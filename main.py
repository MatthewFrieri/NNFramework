import mytorch as mt

a = mt.Tensor([3.5, 1.5])
b = mt.Tensor([1.7, 4.6])
c = mt.Tensor([4.9, 5.2])

z = a - 4 / (2 + mt.Tensor.log(b) * 3 + c / a / 2)
z.show_tree()
z.back(on=1)

print(a.grad)
print(b.grad)
print(c.grad)
