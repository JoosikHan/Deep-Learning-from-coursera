## Week 2 Quiz - Neural Network Basics

1. What does a neuron compute?

    - [ ] A neuron computes an activation function followed by a linear function (z = Wx + b)

    - [x] A neuron computes a linear function (z = Wx + b) followed by an activation function

    - [ ] A neuron computes a function g that scales the input x linearly (Wx + b)

    - [ ] A neuron computes the mean of all features before applying the output to an activation function

    Note: The output of a neuron is a = g(Wx + b) where g is the activation function (sigmoid, tanh, ReLU, ...).
    
2. Which of these is the "Logistic Loss"?

    cross-entropy loss function. [click](https://en.wikipedia.org/wiki/Cross_entropy)
    
    
3. Suppose img is a (32,32,3) array, representing a 32x32 image with 3 color channels red, green and blue. How do you reshape this into a column vector?

     `x = img.reshape((32 * 32 * 3, 1))`
    
4. Consider the two following random arrays "a" and "b":

    ```
    a = np.random.randn(2, 3) # a.shape = (2, 3)
    b = np.random.randn(2, 1) # b.shape = (2, 1)
    c = a + b
    ```
    
    What will be the shape of "c"?
    
    b is copied 3 times (broadcasting). Therfore, `c.shape = (2, 3)`
    
    
5. Consider the two following random arrays "a" and "b":

    ```
    a = np.random.randn(4, 3) # a.shape = (4, 3)
    b = np.random.randn(3, 2) # b.shape = (3, 2)
    c = a * b
    ```
    
    What will be the shape of "c"?
    
    Error.
    The "*" operator between matrices is element-wise multiplication.
    Therefore, identify the same dimension between two matrices.
    

6. Suppose you have n_x input features per example. Recall that X=[x^(1), x^(2)...x^(m)]. What is the dimension of X?

    `(n_x, m)`

    Note: A stupid way to validate this is use the formula Z^(l) = W^(l)A^(l) when l = 1, then we have
    
    - X.shape = (n_x, m) 
    - Z^(1).shape = (n^(1), m)
    - W^(1).shape = (n^(1), n_x)
    
7. Recall that `np.dot(a,b)` performs a matrix multiplication on a and b, whereas `a*b` performs an element-wise multiplication.

    Consider the two following random arrays "a" and "b":

    ```
    a = np.random.randn(12288, 150) # a.shape = (12288, 150)
    b = np.random.randn(150, 45) # b.shape = (150, 45)
    c = np.dot(a, b)
    ```
    
    What is the shape of c?
    
    `c.shape = (12288, 45)`
    A(m, n) B(n, l) => np.dot(A,B) => (m, l) n must be same.
    
8. Consider the following code snippet:

    ```
    # a.shape = (3,4)
    # b.shape = (4,1)
    for i in range(3):
      for j in range(4):
        c[i][j] = a[i][j] + b[j]
    ```
    
    How do you vectorize this?

    `c = a + b.T`

9. Consider the following code:

    ```
    a = np.random.randn(3, 3)
    b = np.random.randn(3, 1)
    c = a * b
    ```
    
    What will be c?
    b columns is copied three times to be (3,3)
    `c.shape = (3, 3)` => broadcasting   
    
10. Consider the following computation graph.

    <p align="center">
      <img src="https://github.com/JoosikHan/Deep-Learning-from-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Images/compute%20forward.PNG"/>
    </p>
    
    - [ ] J = (c - 1) * (b + a)
    - [x] J = (a - 1) * (b + c)
    - [ ] J = a*b + b*c + a*c
    - [ ] J = (b - 1) * (c + a)
