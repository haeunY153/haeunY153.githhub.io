---
layout: single
title: "[Pytorch] 설치부터 기본 사용법"
categories: Pytorch
tags: [python, pytorch]
toc: True
author_profile: False
sidebar:
    nav: "docs"
---


### Pytorch 주피터 노트북 설치하기


```python
pip install torch torchvision torchaudio
```

    Collecting torch
      Downloading torch-1.12.1-cp39-cp39-win_amd64.whl (161.8 MB)
    Collecting torchvision
      Downloading torchvision-0.13.1-cp39-cp39-win_amd64.whl (1.1 MB)
    Collecting torchaudio
      Downloading torchaudio-0.12.1-cp39-cp39-win_amd64.whl (969 kB)
    Requirement already satisfied: typing-extensions in c:\anaconda\lib\site-packages (from torch) (4.1.1)
    Requirement already satisfied: numpy in c:\anaconda\lib\site-packages (from torchvision) (1.21.6)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in c:\anaconda\lib\site-packages (from torchvision) (9.0.1)
    Requirement already satisfied: requests in c:\anaconda\lib\site-packages (from torchvision) (2.27.1)
    Requirement already satisfied: certifi>=2017.4.17 in c:\anaconda\lib\site-packages (from requests->torchvision) (2022.9.24)
    Requirement already satisfied: idna<4,>=2.5 in c:\anaconda\lib\site-packages (from requests->torchvision) (3.3)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\anaconda\lib\site-packages (from requests->torchvision) (1.26.9)
    Requirement already satisfied: charset-normalizer~=2.0.0 in c:\anaconda\lib\site-packages (from requests->torchvision) (2.0.4)
    Installing collected packages: torch, torchvision, torchaudio
    Successfully installed torch-1.12.1 torchaudio-0.12.1 torchvision-0.13.1
    Note: you may need to restart the kernel to use updated packages.
    

---


### 기본 연산 및 동작
- numpy와 유사함


```python
import numpy as np
import torch

n_array = np.arange(10).reshape(2, 5)
print(n_array)
print('dim: ', n_array.ndim, ', shape: ', n_array.shape)
```

    [[0 1 2 3 4]
     [5 6 7 8 9]]
    dim:  2 , shape:  (2, 5)
    


```python
# Array to Tensor

data =[[3, 5], [10, 5]]
x_data = torch.tensor(data)
x_data

#ndarray to Tensor

array = np.array(data)
tensor_array = torch.from_numpy(array)
tensor_array
```




    tensor([[ 3,  5],
            [10,  5]], dtype=torch.int32)




```python
# Operations like numpy generally

torch.ones_like(x_data)
```




    tensor([[1, 1],
            [1, 1]])




```python
n_array = np.random.random((2, 2))
t = torch.from_numpy(n_array)
t
```




    tensor([[0.8497, 0.9555],
            [0.9499, 0.4401]], dtype=torch.float64)




```python
t[1:]
```




    tensor([[0.9499, 0.4401]], dtype=torch.float64)




```python
t[:1]
```




    tensor([[0.8497, 0.9555]], dtype=torch.float64)




```python
n = np.random.random((3, 5))
t2 = torch.from_numpy(n)
t2
```




    tensor([[0.0011, 0.7782, 0.9890, 0.3174, 0.9397],
            [0.9175, 0.6137, 0.1440, 0.6878, 0.7808],
            [0.8496, 0.1207, 0.6045, 0.6222, 0.9297]], dtype=torch.float64)




```python
t2[1:2, 2:]
```




    tensor([[0.1440, 0.6878, 0.7808]], dtype=torch.float64)




```python
t2[1:3]
```




    tensor([[0.9175, 0.6137, 0.1440, 0.6878, 0.7808],
            [0.8496, 0.1207, 0.6045, 0.6222, 0.9297]], dtype=torch.float64)




```python
t2.flatten()
```




    tensor([0.0011, 0.7782, 0.9890, 0.3174, 0.9397, 0.9175, 0.6137, 0.1440, 0.6878,
            0.7808, 0.8496, 0.1207, 0.6045, 0.6222, 0.9297], dtype=torch.float64)




```python
t2.shape
```




    torch.Size([3, 5])




```python
t2.dtype
```




    torch.float64




```python
t2.numpy()
```




    array([[0.00113862, 0.77818119, 0.98904411, 0.31744628, 0.93970248],
           [0.9174607 , 0.61374777, 0.14404259, 0.68778041, 0.78075312],
           [0.84956723, 0.12072346, 0.60445122, 0.62224636, 0.9297356 ]])




```python
torch.ones_like(t2)
```




    tensor([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]], dtype=torch.float64)




```python
t2.device
```




    device(type='cpu')




```python
x_data = t2

if torch.cuda.is_available():
    x_data_cuda = x_data.to('cuda')

x_data.device
```




    device(type='cpu')



### 형변환, 차원 축소추가

- view: reshape 과 동일하게 tensor의 shape 변환 (강의에서는 shape보다 view 권장; 메모리 저장)
- squeeze: 차원 개수가 1인 차원 삭제 (압축)
- unsqueeze: 차원 개수가 1인 차원 추가 (사이즈 맞출 때 종종 사용)


```python
tensor_ex = torch.rand(size=(2, 3, 2))
tensor_ex
```




    tensor([[[0.2809, 0.2419],
             [0.6830, 0.6607],
             [0.8493, 0.3878]],
    
            [[0.2236, 0.8167],
             [0.1403, 0.5048],
             [0.6939, 0.0065]]])




```python
tensor_ex.view([-1, 6]) # 1 by 6 2개로 변환
```




    tensor([[0.2809, 0.2419, 0.6830, 0.6607, 0.8493, 0.3878],
            [0.2236, 0.8167, 0.1403, 0.5048, 0.6939, 0.0065]])




```python
tensor_ex.reshape([-1, 6])
```




    tensor([[0.2809, 0.2419, 0.6830, 0.6607, 0.8493, 0.3878],
            [0.2236, 0.8167, 0.1403, 0.5048, 0.6939, 0.0065]])




```python
# view, reshape 차이점 데이터 유지 측면에서 다름

a = torch.zeros(3, 2)
b = a.view(2, 3)
a.fill_(1)
```




    tensor([[1., 1.],
            [1., 1.],
            [1., 1.]])




```python
c= torch.zeros(3, 2)
d= c.t().reshape(6)
d.fill_(1)
```




    tensor([1., 1., 1., 1., 1., 1.])




```python
data = [[1, 2], [3, 4]]
t = torch.tensor(data)
```


```python
t
```




    tensor([[1, 2],
            [3, 4]])




```python
t.unsqueeze(0)
```




    tensor([[[1, 2],
             [3, 4]]])




```python
t.unsqueeze(1)
```




    tensor([[[1, 2]],
    
            [[3, 4]]])




```python
t.unsqueeze(2)
```




    tensor([[[1],
             [2]],
    
            [[3],
             [4]]])




```python
t.squeeze(0)
```




    tensor([[1, 2],
            [3, 4]])




```python
tensor_ex = torch.rand(size=(2,2))
tensor_ex.unsqueeze(0).shape
```




    torch.Size([1, 2, 2])




```python
tensor_ex.unsqueeze(1).shape
```




    torch.Size([2, 1, 2])




```python
tensor_ex.unsqueeze(2).shape
```




    torch.Size([2, 2, 1])




```python
n1 = np.arange(10).reshape(2, 5)
t1 = torch.FloatTensor(n1)
```


```python
t1
```




    tensor([[0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.]])




```python
t1 + t1
```




    tensor([[ 0.,  2.,  4.,  6.,  8.],
            [10., 12., 14., 16., 18.]])




```python
t1+ 10
```




    tensor([[10., 11., 12., 13., 14.],
            [15., 16., 17., 18., 19.]])




```python
t1*5
```




    tensor([[ 0.,  5., 10., 15., 20.],
            [25., 30., 35., 40., 45.]])




```python
t1 - t1
```




    tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])




```python
n2 = np.arange(10).reshape(5, 2)
t2 = torch.FloatTensor(n2)
```


```python
t2
```




    tensor([[0., 1.],
            [2., 3.],
            [4., 5.],
            [6., 7.],
            [8., 9.]])



- 행렬계산: `mm`사용!!


```python
t1.mm(t2) 
```




    tensor([[ 60.,  70.],
            [160., 195.]])




```python
t1.dot(t2)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Input In [64], in <cell line: 1>()
    ----> 1 t1.dot(t2)
    

    RuntimeError: 1D tensors expected, but got 2D and 2D tensors



```python
t1.matmul(t2) # broadcasting 처리
```




    tensor([[ 60.,  70.],
            [160., 195.]])




```python
a = torch.rand(5, 2, 3)
b = torch.rand(5)

a.mm(b)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Input In [73], in <cell line: 4>()
          1 a = torch.rand(5, 2, 3)
          2 b = torch.rand(5)
    ----> 4 a.mm(b)
    

    RuntimeError: self must be a matrix



```python
a = torch.rand(5, 2, 3)
b = torch.rand(3)
a.matmul(b)

```




    tensor([[1.5056, 0.8395],
            [1.4356, 0.5943],
            [0.6893, 0.5864],
            [0.4444, 1.0078],
            [0.7802, 0.7125]])




```python
a
```




    tensor([[[0.8785, 0.7743, 0.8483],
             [0.1465, 0.6375, 0.7575]],
    
            [[0.7441, 0.8085, 0.8749],
             [0.4431, 0.3386, 0.1816]],
    
            [[0.2375, 0.8554, 0.1997],
             [0.1116, 0.4353, 0.5250]],
    
            [[0.0756, 0.3286, 0.4108],
             [0.6210, 0.2485, 0.7427]],
    
            [[0.6676, 0.2477, 0.2849],
             [0.3762, 0.0232, 0.7308]]])




```python
b
```




    tensor([0.7524, 0.4632, 0.5729])




```python
a[0].mm(torch.unsqueeze(b,1))
a[1].mm(torch.unsqueeze(b,1))
a[2].mm(torch.unsqueeze(b,1))
a[3].mm(torch.unsqueeze(b,1))
a[4].mm(torch.unsqueeze(b,1))
```




    tensor([[0.7802],
            [0.7125]])



### 수식 변환지원 `nn.functional`


```python
import torch as F

tensor = torch.FloatTensor([0.5, 0.7, 0.1])
h_tensor = F.softmax(tensor, dim=0)
h_tensor
```




    tensor([0.3458, 0.4224, 0.2318])




```python
y = torch.randint(5, (10, 5))
y
```




    tensor([[4, 2, 3, 1, 3],
            [2, 3, 1, 1, 1],
            [1, 3, 3, 2, 3],
            [3, 4, 4, 1, 0],
            [3, 2, 3, 3, 1],
            [2, 4, 2, 2, 0],
            [3, 2, 1, 4, 2],
            [3, 3, 0, 0, 0],
            [2, 0, 4, 0, 2],
            [0, 3, 2, 4, 2]])




```python
y_label = y.argmax(dim=1)
y_label
```




    tensor([0, 1, 1, 1, 0, 1, 3, 0, 2, 3])




```python
torch.nn.functional.one_hot(y_label)
```




    tensor([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])



### 자동미분 -> backward 함수 사용
> https://tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html


```python
w = torch.tensor(2.0, requires_grad=True)
y = w**2
z = 10*y+2
z.backward()
```


```python
z
```




    tensor(42., grad_fn=<AddBackward0>)




```python
w.grad
```




    tensor(40.)




```python
# 변화도 gradient 확인 

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2
external_grad = torch.tensor([1, 1])
Q.backward(gradient=external_grad)
```


```python
a.grad
```




    tensor([36., 81.])




```python
b.grad
```




    tensor([-12.,  -8.])




```python
# 수집된 변화도가 올바른지 확인

print(9*a**2 == a.grad)
print(-2*b == b.grad)
```

    tensor([True, True])
    tensor([True, True])
    
