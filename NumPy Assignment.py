import numpy as np
arr1 = np.array([[1, 2, 4], [7, 13, 21]])
arr2 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print("arr1:\n", arr1)
print("Shape of arr1:", arr1.shape, "; No. of Dimensions:", arr1.ndim)
print("arr2:\n", arr2)
print("Shape of arr2:", arr2.shape, "; No. of Dimensions:", arr2.ndim)
arr3 = np.zeros((2, 3))
arr4 = np.ones((2, 3))
arr5 = np.full((2, 3), 5)
arr6 = np.arange(0, 22, 2)
arr7 = np.linspace(0, 20, 11)
print(f"arr3: {arr3}")
print(f"arr4: {arr4}")
print(f"arr5: {arr5}")
print(f"arr6: {arr6}")
print(f"arr7: {arr7}")
arr8 = np.random.rand(2,3)
arr9 = np.random.randint(5, 9, size = (2, 3))
print(arr8)
print(arr9)
#accesing arrays in NumPy
arr = np.array([[1, 3, 5, 7], [9, 11, 13, 15], [17, 19, 21, 23]])
print(arr[1, 2])
print(arr[1:3, 1:2])
print(arr[1:3, 3:4])
multiples = arr[arr%3 == 0]
print("Values in the array which are a multiple of 3:\n")
print(multiples)
#Reshaping
reshaped_arr = arr.reshape(2, 6)
print(reshaped_arr)
print(np.transpose(arr))
#Broadcasting
a = np.array([1,2,3])
b = np.array([4,5]).reshape(2,1)
print(a+b)
#Miscellaneous
a = np.array([1,2,3])
x = np.transpose(arr)
y = np.transpose(a)
print(np.multiply(x,y))
print(x*y)
print(np.matmul(x,y))
print(np.dot(x,y))










