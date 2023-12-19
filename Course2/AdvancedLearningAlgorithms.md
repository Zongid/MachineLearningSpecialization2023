---
aliases: 
author: Li Yaozong
date: 2023-09-20
time: 2023-09-20 16:00
cover: https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://coursera-course-photos.s3.amazonaws.com/78/f17b8da6f949f28a5e35a70bf5a837/MLS.course-banners-01_Course-Logo-2---DLAI-web.png
description: Advanced Learning Algorithms
source: Coursera
status: Completed
tags:
  - cs
  - MachineLearing
  - notes
url: https://www.coursera.org/learn/advanced-learning-algorithms/home/welcome
---

![cover](https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://coursera-course-photos.s3.amazonaws.com/78/f17b8da6f949f28a5e35a70bf5a837/MLS.course-banners-01_Course-Logo-2---DLAI-web.png)

# Week1 Neural Networks

Learning Objectives

- Get familiar with the diagram and components of a neural network
- Understand the concept of a "layer" in a neural network
- Understand how neural networks learn new features.
- Understand how activations are calculated at each layer.
- Learn how a neural network can perform classification on an image.
- Use a framework, TensorFlow, to build a neural network for classification of an image.
- Learn how data goes into and out of a neural network layer in TensorFlow
- Build a neural network in regular Python code (from scratch) to make predictions.
- (Optional): Learn how neural networks use parallel processing (vectorization) to make computations faster.

## Neural networks intuition

___
***Welcome!***

___
***Neurons and the brain***

**Neural networks**
+ Origins: Algorithms that try to mimic the brain
+ Used in the 1980's and early 1990's
+ Fell out of favor in the late 1990's.
+ Resurgence from around 2005
+ speech $\rightarrow$ images $\rightarrow$ text(NLP) $\rightarrow$ ...

![](./images/Pasted%20image%2020230920175622.png)
![](./images/Pasted%20image%2020230920180034.png)

___
***Demand Prediction***

![](./images/Pasted%20image%2020230920180517.png)

> $a$: ***activation***, output of logistic regression algorithm

![](./images/Pasted%20image%2020230920181502.png)

Price, shipping cost, maketing, material $\Rightarrow$ affordability, awareness and percived quality $\Rightarrow$ probability of being a top seller

***activations***: the degree that the biological neuron is sending a high output or sending many electrical impulses to other neurons to the downstream from it. 

![](./images/Pasted%20image%2020230920182138.png)
![](./images/Pasted%20image%2020230920182444.png)

> <font color="#00b0f0">multilayer perception</font> = neural network architecture

___
***Example: Recognizing lmages***

![](./images/Pasted%20image%2020230920201036.png)
![](./images/Pasted%20image%2020230920201236.png)

## Practice quiz: Neural networks intuition

1. Which of these are terms used to refer to components of an artificial neural network? (hint: three of these are correct)

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/36518785-fa48-4735-8807-5da27651090bimage2.png?expiry=1695340800000&hmac=8AJnSg-bxk2_1Xef0I3wQcfINq1JFkD-9JZGLkw7XK0)

+ activation function
+ layers
+ axon
+ neurons

My Answer: A, B, D $\checkmark[ChatGPT+ERNIE\ Bot]$

> [!check] Correct
> 
> + activation function: Yes, an activation is the number calculated by a neuron (and “activations” in the figure above is a vector that is output by a layer that contains multiple neurons).
> + Layer: Yes, a layer is a grouping of neurons in a neural network
> + neurons: Yes, a neuron is a part of a neural network

2. True/False? Neural networks take inspiration from, but do not very accurately mimic, how neurons in a biological brain learn.

My Answer: True  $\checkmark[ChatGPT+ERNIE\ Bot]$

> True. [ChatGPT]
> 
> Neural networks take inspiration from how neurons in a biological brain work and learn, but they do not very accurately mimic the intricacies of biological neural systems. While artificial neural networks are loosely inspired by the basic structure and function of biological neurons, they are highly simplified mathematical models that are designed for specific computational tasks. Biological brains are incredibly complex and still not fully understood, and artificial neural networks are a simplification of those systems for practical computational purposes.

> [!check] Correct
> 
> Artificial neural networks use a very simplified mathematical model of what a biological neuron does.

## Neural network model

___
***Neural network layer***

![](./images/Pasted%20image%2020230920210823.png)

> [<font color=red>2</font>]: a quantity associated with layer <font color=red>2</font> of the neural network

![](./images/Pasted%20image%2020230920211609.png)
![](./images/Pasted%20image%2020230920211903.png)

___
***More complex neural networks***

![](./images/Pasted%20image%2020230920213459.png)
![](./images/Pasted%20image%2020230920213847.png)

> [!question] Quiz
> 
> Can you fill in the superscripts and subscripts for the second neuron?
> 
> ![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/mrRWkOajSr-0VpDmo2q_CA_748d8f1e177b4af3a252415f8a08b3f1_mls-c2-w1-l2-s02-ivq.png?expiry=1695340800000&hmac=ZSDDmaAmrnZBzePBlA113bNTwoxxjpQYjz8q_gjYj6w)
> 
> + $a_2^{[3]}=g(\vec{w}_2^{[3]}\cdot\vec{a}^{[2]}+b_2^{[3]})$ $\checkmark$
> + $a_2^{[3]}=g(\vec{w}_2^{[3]}\cdot\vec{a}^{[3]}+b_2^{[3]})$
> + $a_2^{[3]}=g(\vec{w}_2^{[3]}\cdot a_2^{[2]}+b_2^{[3]})$
> 
> A Correct
> 
> This is correct. Please continue the lecture video to learn why!

![](./images/Pasted%20image%2020230920215208.png)

___
***Inference: making predictions (forward propagation)***

> + 前向传播(Forward Propagation)
> + Backward Propagation/Back Propagation

![](./images/Pasted%20image%2020230920215732.png)
![](./images/Pasted%20image%2020230920215832.png)

> [!NOTE] Note
> 
> **a pretty typical choice**: the number of hidden units **decreases** as you get **closer** to the **output layer**

___
***Lab: Neurons and Layers***

[Lab: Neurons and Layers](C2W1/C2_W1_Lab01_Neurons_and_Layers.ipynb)

## Practice quiz: Neural network model

<font  face="Times New Roman" color=green size=5> <ins>2023-10-26 17:56 $\downarrow$ </ins> </font>

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/5f7ddfc8-51f8-4824-8be3-b9710fa4f466image3.png?expiry=1698451200000&hmac=UlFk58pQDpGW4UZVAzBdG6hhOaaQb6BLxr6cNNkyDoc)

1. For a neural network, what is the expression for calculating the activation of the third neuron in layer 2? Note, this is different from the question that you saw in the lecture video.

+ A. $a_3^{[2]}=g(\vec{w}_3^{[2]}\cdot\vec{a}^{[1]}+b_3^{[2]})$
+ B. $a_3^{[2]}=g(\vec{w}_2^{[3]}\cdot\vec{a}^{[1]}+b_2^{[3]})$
+ C. $a_3^{[2]}=g(\vec{w}_2^{[3]}\cdot\vec{a}^{[2]}+b_2^{[3]})$
+ D. $a_3^{[2]}=g(\vec{w}_3^{[2]}\cdot\vec{a}^{[2]}+b_3^{[2]})$

My Answer: A

> [!check] Correct
> 
> Yes! The superscript [2] refers to layer 2. The subscript 3 refers to the neuron in that layer. The input to layer 2 is the activation vector from layer 1.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/5f7ddfc8-51f8-4824-8be3-b9710fa4f466image4.png?expiry=1698451200000&hmac=S0iNfcNrxer2AySyBkGl8hnFM9_s3Zfmxt1zBEwJxGM)

2. For the handwriting recognition task discussed in lecture, what is the output $a_1^{[3]}$ ​?

+ A. A vector of several numbers that take values between 0 and 1
+ B. The estimated probability that the input image is of a number 1, a number that ranges from 0 to 1.
+ C. A vector of several numbers, each of which is either exactly 0 or 1
+ D. A number that is either exactly 0 or 1, comprising the network’s prediction

My Answer: B

> [!check] Correct
> 
> Yes! The neural network outputs a single number between 0 and 1.

## TensorFlow implementation

___
***Inference in Code***

<font  face="Times New Roman" color=green size=5><u>2023-09-22 14:00 $\downarrow$ </u></font>

One of the remarkable things about neural networks is the same algorithm can be applied to so many different applications. 

![](./images/Pasted%20image%2020230922141946.png)
![](./images/Pasted%20image%2020230922142407.png)
![](./images/Pasted%20image%2020230922142944.png)

___
***Data in TensorFlow***

![](./images/Pasted%20image%2020230922143600.png)

represent data
+ logistic regression: 1D vectors
+ TensorFlow: Matrices

![](./images/Pasted%20image%2020230922144747.png)
![](./images/Pasted%20image%2020230922145007.png)

___
***Building a neural network***

![](./images/Pasted%20image%2020230922145958.png)

```python
layer_1 = Dense(units=3, activation="sigmoid")
layer_2 = Dense(units=1, activation="sigmoid")
model = Aequential([layer_1, layer_2])
```

$$\Downarrow$$

```python
model = Sequential([
	Dense(units=3, activation="sigmoid"),
	Dense(units=1, activation="sigmoid")
])
```

![](./images/Pasted%20image%2020230922150453.png)


___
***Lab: Coffee Roasting in Tensorflow***

[Lab: Coffee Roasting in Tensorflow](C2W1/C2_W1_Lab02_CoffeeRoasting_TF.ipynb)

## Practice quiz: TensorFlow implementation

1. For the the following code:
```python
model= Sequential([
Dense(units=25,activation="sigmoid"),
Dense(units=15,activation="sigmoid"),
Dense(units=10,activation="sigmoid"),
Dense(units=1, activation="sigmoid")])
```
This code will define a neural network with how many layers?
+ A. 3
+ B. 5
+ C. 25
+ D. 4

My Answer: D $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Yes! Each call to the "Dense" function defines a layer of the neural network.

2. How do you define the second layer of a neural network that has 4 neurons and a sigmoid activation?

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/98c635c3-33fc-4d9c-9af5-6f1f18da35aeimage2.png?expiry=1695513600000&hmac=pYSo2qqV4nq_UQmFeAB2gei_OmKpZPLB_uMAblN9mOc)

+ A. `Dense(layer=2, units=4, activation ='sigmoid')`
+ B. `Dense(units=4)`
+ C. `Dense(units=4,activation='sigmoid')`
+ D. `Dense(units=[4],activation=['sigmoid'])`

My Answer: C $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Yes! This will have 4 neurons and a sigmoid activation.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/98c635c3-33fc-4d9c-9af5-6f1f18da35aeimage3.png?expiry=1695513600000&hmac=WfXX7bxVvmvy0MgzwVxhlq8GqCEPFUOSTbigRFdJKJM)

3. If the input features are temperature (in Celsius) and duration (in minutes), how do you write the code for the first feature vector x shown above?
+ A. `x= np.array([[200.0,17.0]])`
+ B. `x= np.array([[200.0 + 17.0]])`
+ C. `x=np.array([[200.0],[17.0]])`
+ D. `x=np.array([['200.0','17.0']])`

My Answer: A $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Yes! A row contains all the features of a training example. Each column is a feature.

## Neural network implementation in Python

___
***Forward prop in a single layer***

![](./images/Pasted%20image%2020230922154136.png)

___
***General implementation of forward propagation***

![](./images/Pasted%20image%2020230922164341.png)

___
***Lab: CoffeeRoastingNumPy***

[Lab: CoffeeRoastingNumPy](C2W1/C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb)

## Practice quiz: Neural network implementation in Python

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bH5WifgQRe--Von4ENXvAg_2ccc430b8c03400ca13f22e9fa161aa1_frame_8527.jpg?expiry=1695513600000&hmac=1xtGH3yeawxxMZzt3c3I-KRPz80dv7ypFgwZhwX9Fgw)

1. According to the lecture, how do you calculate the activation of the third neuron in the first layer using NumPy?
+ A. z1_3=np.dot(w1_3,x)+b1_3; a1_3=sigmoid(z1_3)
+ B. layer_1=Dense(units=3,activation='sigmoid'); a_1=layer_1(x)
+ C. z1_3=w1_3\*x+b; a1_3=sigmoid(z1_3)

My Answer: A $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Correct. Use the numpy.dot function to take the dot product. The sigmoid function shown in lecture can be a function that you write yourself (see course 1, week 3 of this specialization), and that will be provided to you in this course.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/38b928e9-0b01-456d-9549-be803e348303image1.png?expiry=1695513600000&hmac=gYmaQhXcbGRqEPLuq9BVcr2XPi-jG0bqitoqqKeUWvw)

2. According to the lecture, when coding up the numpy array W, where would you place the w parameters for each neuron?
+ A. In the rows of W.
+ B. In the columns of W.

My Answer: B $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Correct. The w parameters of neuron 1 are in column 1. The w parameters of neuron 2 are in column 2, and so on.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/38b928e9-0b01-456d-9549-be803e348303image1.png?expiry=1695513600000&hmac=gYmaQhXcbGRqEPLuq9BVcr2XPi-jG0bqitoqqKeUWvw)

3. For the code above in the "dense" function that defines a single layer of neurons, how many times does the code go through the "for loop"? Note that W has 2 rows and 3 columns.
+ A. 2 times
+ B. 3 times
+ C. 5 times
+ D. 6 times

My Answer: B $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Yes! For each neuron in the layer, there is one column in the numpy array W. The for loop calculates the activation value for each neuron. So if there are 3 columns in W, there are 3 neurons in the dense layer, and therefore the for loop goes through 3 iterations (one for each neuron).

## Speculations on artificial general intelligence (AGl)

***ls there a path to AGl?***

![](./images/Pasted%20image%2020230922171033.png)
![](./images/Pasted%20image%2020230922171158.png)
![](./images/Pasted%20image%2020230922171300.png)
![](./images/Pasted%20image%2020230922171507.png)
![](./images/Pasted%20image%2020230922171707.png)

## Vectorization(optional)

___
***How neural networks are implemented efficiently***

![](./images/Pasted%20image%2020230922174643.png)

___
***Matrix multiplication***

<font  face="Times New Roman" color=green size=5><u>2023-09-23 16:37 $\downarrow$ </u></font>

**Dot products**

![](./images/Pasted%20image%2020230923165357.png)

> transpose: 转置

**Vector matrix multiplication**

![](./images/Pasted%20image%2020230923165650.png)

**matrix matrix multiplication**

![](./images/Pasted%20image%2020230923170427.png)

___
***Matrix multiplication rules***

<font  face="Times New Roman" color=green size=5><u>2023-09-24 14:08 $\downarrow$ </u></font>

**Matrix multiplication rules**

![](./images/Pasted%20image%2020230924144312.png)

> [!question] Quiz
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/8b-O3yRqQoe_jt8kajKHxQ_397d1360a2164e8d83723dc17711f8f1_MLS-C2-W1-L6-S03-IVQ.png?expiry=1695686400000&hmac=mx0IDElqoGmh45OZJ2WIhkwkQGLISvG0GBKEMAbihkE)
> 
> Can you calculate the value at row 2, column 3?
> 
> + ==(-1 x 7) + (-2 x 8) = -23== $\checkmark$
> + (0.1 x 5) + (0.2 x 6) = 1.7
> + (1 x 3) + (2 x 4) = 11
> 
> Correct
> 
> This is correct.  Take row 2 of $X^T$ and column 3 of $W$

![](./images/Pasted%20image%2020230924144446.png)

___
***Matrix multiplication code***

**Matrix multiplication in NumPy**

$$A=\begin{bmatrix}1&-1&0.1\\
2&-2&0.2\end{bmatrix}\quad A^T=\begin{bmatrix}1&2\\
-1&-2\\
0.1&0.2\end{bmatrix}$$

$$W=\begin{bmatrix}3&5&7&9\\
4&6&8&0\end{bmatrix}$$

$$Z=A^TW=\begin{bmatrix}11&17&23&9\\
-11&-17&-23&-9\\
1.1&1.7&2.3&0.9\end{bmatrix}$$

```python
import numpy as np

A = np.array([[1, -1, 0.1], [2, -2, 0.2]])
# AT = np.array([[1, 2], [-1, -2], [0.1, 0.2]])
AT = A.T  # transpose
W = np.array([[3, 5, 7, 9], [4, 6, 8, 0]])
Z = np.matmul(AT, W)  # method1 clearer
# Z = AT @ W  # method2

print(Z)
# output:
# [[ 11.   17.   23.    9. ]
#  [-11.  -17.  -23.   -9. ]
#  [  1.1   1.7   2.3   0.9]]
```

**Dense layer vectorized**

![](./images/Pasted%20image%2020230924150236.png)

## Practice Lab: Neural networks

[Practice Lab: Neural networks](C2_W1/C2_W1_Assignment.ipynb)

# Week2 Neural network training

Learning Objectives

- Train a neural network on data using TensorFlow
- Understand the difference between various activation functions (sigmoid, ReLU, and linear)
- Understand which activation functions to use for which type of layer
- Understand why we need non-linear activation functions
- Understand multiclass classification
- Calculate the softmax activation for implementing multiclass classification
- Use the categorical cross entropy loss function for multiclass classification
- Use the recommended method for implementing multiclass classification in code
- (Optional): Explain the difference between multi-label and multiclass classification

## Neural Network Training

***TensorFlow implementation***

**Train a Neural Network in TensorFlow**

![](./images/Pasted%20image%2020230924150854.png)

> epochs: number of steps in gradient descent

___
***Training Details***

**Model Training Steps**

![](./images/Pasted%20image%2020230924151819.png)

1. Create the model

![](./images/Pasted%20image%2020230924151924.png)

2. Loss and cost functions

![](./images/Pasted%20image%2020230924153050.png)

3. Gradient descent

![](./images/Pasted%20image%2020230924153410.png)

**Neural network libraies**

Use code libraries instead of coding "from scratch"

![](./images/Pasted%20image%2020230924153429.png)

Good to understand the implementation (for tuning and debugging).

## Practice quiz: Neural Network Training

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/141c2e0d-b88f-4876-a375-6b18af36255fimage3.png?expiry=1695859200000&hmac=uDPjpGiddV_QQyUkVGdevPwvxR-yAEtgHyBIq21YjCA)

1. Here is some code that you saw in the lecture:

```python
model.compile(loss=BinaryCrossentropy())
```

For which type of task would you use the binary cross entropy loss function?
+ A. A classification task that has 3 or more classes (categories)
+ B. BinaryCrossentropy() should not be used for any task.
+ C. regression tasks (tasks that predict a number)
+ D. binary classification (classification with exactly 2 classes)

My Answer: D $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Yes! Binary cross entropy, which we've also referred to as logistic loss, is used for classifying between two classes (two categories).

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/141c2e0d-b88f-4876-a375-6b18af36255fimage3.png?expiry=1695859200000&hmac=uDPjpGiddV_QQyUkVGdevPwvxR-yAEtgHyBIq21YjCA)

2. Here is code that you saw in the lecture:

```python
model= Sequential([
Dense(units=25,activation='sigmoid’),
Dense(units=15,activation='sigmoid’),
Dense(units=1,activation='sigmoid’)
])
model.compile(loss=BinaryCrossentropy())
model.fit(X,y,epochs=100)
```

Which line of code updates the network parameters in order to reduce the cost?

+ A. None of the above -- this code does not update the network parameters.
+ B. `model.compile(loss=BinaryCrossentropy())`
+ C. `model.fit(X,y,epochs=100)`
+ D. `model= Sequential([...])`

My Answer: C $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Yes! The third step of model training is to train the model on data in order to minimize the loss (and the cost)

## Activation Functions

___
***Alternatives to the sigmoid activation***

**Demand Prediction Example**

![](./images/Pasted%20image%2020230924154403.png)

> ReLU: Rectified linear unit

**Examples of Activation Functions**

![](./images/Pasted%20image%2020230924154713.png)

___
***Choosing activation functions***

**Output layer**

> [!NOTE] Note
> 
> Binary classification $\rightarrow$ Sigmoid  
> Regression problem $\rightarrow$ different activation function

![](./images/Pasted%20image%2020230924155639.png)

**Hidden layer**

> ReLU: most common choice
> + faster
> + flat

![](./images/Pasted%20image%2020230924160246.png)

**Choosing Activation Summary**

![](./images/Pasted%20image%2020230924160942.png)

___
***Why do we need activation functions?***

![](./images/Pasted%20image%2020230924162902.png)

Linear Example

![](./images/Pasted%20image%2020230924163123.png)

example

![](./images/Pasted%20image%2020230924163255.png)

___
***Lab: ReLU activation***

## Practice quiz: Activation Functions

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/abeab774-084a-4758-9d01-bc275c77c747image4.png?expiry=1695859200000&hmac=QsBtDQ78VtCZOW4Q2gxdA45T6gLBDK8yyMe5ZbU4zCY)

1. Which of the following activation functions is the most common choice for the hidden layers of a neural network?
+ ReLU (rectified linear unit)
+ Linear
+ Most hidden layers do not use any activation function
+ Sigmoid

My Answer: ReLU $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Yes! A ReLU is most often used because it is faster to train compared to the sigmoid. This is because the ReLU is only flat on one side (the left side) whereas the sigmoid goes flat (horizontal, slope approaching zero) on both sides of the curve.


![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/abeab774-084a-4758-9d01-bc275c77c747image3.png?expiry=1695859200000&hmac=HTBMVICzUtuyeGgbxUieEzqAhhWv2pdYDHJqBD14BZA)

2. For the task of predicting housing prices, which activation functions could you choose for the output layer? Choose the 2 options that apply.
+ Sigmoid
+ ReLU
+ linear

My Answer: ReLU, linear $\checkmark[ChatGPT]$

> [!check] Correct
> 
> + Yes! ReLU outputs values 0 or greater, and housing prices are positive values.
> + Yes! A linear activation function can be used for a regression task where the output can be both negative and positive, but it's also possible to use it for a task where the output is 0 or greater (like with house prices).

3. True/False? A neural network with many layers but no activation function (in the hidden layers) is not effective; that's why we should instead use the linear activation function in every hidden layer.

+ My Answer: False $\checkmark$
+ ChatGPT Answer: True $[ChatGPT]$

> [!check] Correct
> 
> Yes! A neural network with many layers but no activation function is not effective. A linear activation is the same as "no activation function".

## Multiclass Classification

___
***Multiclass***

> Multiclass classification problem:
> Target $y$ can take on more than <font color=red>two</font> possible values

**Multiclass classification example**

![](./images/Pasted%20image%2020230924165500.png)

___
***Softmax***

> **Softmax regression algorithm** is a generalization of logistic regression, which is a binary classification algorithm to the multiclass classification contexts.

> [!question] Quiz
> 
> What do you think $a_4$​ is equal to?
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/ZHisrR1fSkO4rK0dXzpDxQ_9abeffdf3af449a3bb85ff30488bd0f1_MLS-C2-W2-L3-S02-IVQ.png?expiry=1695686400000&hmac=EiM2Sfi36KMxM9iszfsCNGmhkFAJ57mw8XhdDaJl7cY)
> 
> + ==0.35== $\checkmark$
> + 0.40
> + -0.40
> 
> Correct
> 
> This is correct. Please continue the video to see why!

![](./images/Pasted%20image%2020230924171136.png)

**Cost**

![](./images/Pasted%20image%2020230924172707.png)

___
***Neural Network with Softmax output***

![](./images/Pasted%20image%2020230924215905.png)

![](./images/Pasted%20image%2020230924220511.png)

> what sparse categorical refers to is that you're still classified $y$ into categories.
> 
> sparse refers to that $y$ can only take on **one** of these 10 values.

> MNIST: MNIST是**Mixed National Institue of Standards and Technology database**的简称，中文叫做美国国家标准与技术研究所数据库。

___
***lmproved implementation of softmax***

<font  face="Times New Roman" color=green size=5><u>2023-09-26 13:46 $\downarrow$</u></font>

```python
x1 = 2.0/10000
print(f"{x1:.18f}") # print 18 digits to the right of decimal point
# 0.000200000000000000
```

```python
x2 = 1+(1/10000) - (1-1/10000)
print(f"{x2:.18f}")
# 0.000199999999999978
# ERROR
```

**Numerical Roundoff Errors**

![](./images/Pasted%20image%2020230926135753.png)

**More numerically accurate implementation of softmax**

![](./images/Pasted%20image%2020230926140251.png)

**MNIST (more numerically accurate)**

![](./images/Pasted%20image%2020230926140416.png)

**logistic regression (more numerically accurate)**

![](./images/Pasted%20image%2020230926140517.png)

___
***Classification with multiple outputs (Optional)***

**Multi-label Classification**

![](./images/Pasted%20image%2020230926140925.png)

![](./images/Pasted%20image%2020230926141320.png)

Alternatively, train one neural network with three outputs

![](./images/Pasted%20image%2020230926141347.png)

___
***Lab: Softmax***

[Lab: Softmax](C2W2/C2_W2_SoftMax.ipynb)

___
***Lab: Multiclass***

[Lab: Multiclass](C2W2/C2_W2_Multiclass_TF.ipynb)

## Practice quiz: Multiclass Classification

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/f38d2d9d-5e70-4900-bd84-baf812439294image2.png?expiry=1695859200000&hmac=8Z0D4edLDYZXpaAodcWZAb9NAYVx_PG47itj_bROsik)

1. For a multiclass classification task that has 4 possible outputs, the sum of all the activations adds up to 1. For a multiclass classification task that has 3 possible outputs, the sum ofall the activations should add up to ....
- A Less than 1
- B lt will vary, depending on the input x.
- C 1
- D More than 1

My Answer: C $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Yes! The sum of all the softmax activations should add up to 1. One way to see this is that if $e^{z_1}=10,e^{z_2}=20,e^{z_3}=30$, then the sum of $a_1​+a_2​+a_3$​ is equal to $\frac{e^{z_1}+e^{z_2}+e^{z_3}}{e^{z_1}+e^{z_2}+e^{z_3}}$​​ which is 1.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/f38d2d9d-5e70-4900-bd84-baf812439294image4.png?expiry=1695859200000&hmac=ujX0xj2v-LRcKoy6DDvUnG_tE1RtqOYYaLPrxvSDEwU)

2. For multiclass classification, the cross entropy loss is used for training the model. lf there are 4 possible classes for the output, and for a particular training example, the true class of the example is class 3 (y=3), then what does the cross entropy loss simplify to? [Hint: This loss should get smaller when a3 gets larger.]
+ A $z_3/(z_1+z_2+z_3+z_4)$
+ B $-log(a_3)$
+ C $z_3$
+ D $\frac{-log(a_1)+-log(a_2)+-log(a_3)+-log(a_4)}{4}$

My Answer: B $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Correct. When the true label is 3, then the cross entropy loss for that training example is just the negative of the log of the activation for the third neuron of the softmax. All other terms of the cross entropy loss equation $(-log(a_1),-log(a_2),and-log(a_4))$ are ignored

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/f38d2d9d-5e70-4900-bd84-baf812439294image5.png?expiry=1695859200000&hmac=KikwpRdx35zukULGTtEcIuN1fFv-QhFFP885V8C72VI)

3. For multiclass classification, the recommended way to implement softmax regression is to set from_logits=True in the loss function, and also to define the model's output layer with...
+ A a'softmax' activation
+ B a'linear' activation

+ My Answer: A $×[ChatGPT]$
+ ChatGPT answer: B a'linear' activation $[ChatGPT]$
+ Correct Answer: B

> [!failure] Incorrect
> 
> Feedback: When you set from_logits=True, then it expects the output layer of the model to be 'linear' (logits), because the loss function calculates the softmax itself with a more numerically stable method.

> [!check] B Correct
> 
> Yes! Set the output as linear, because the loss function handles the calculation of the softmax with a more numerically stable method.


## Additional Neural Network Concepts

***Advanced Optimization***

**Gradient Descent**

![](./images/Pasted%20image%2020230926155901.png)

**Adam Algorithm Intuition**

Adam: Adaptive Moment estimation
![](./images/Pasted%20image%2020230926160026.png)
![](./images/Pasted%20image%2020230926160146.png)

**MNIST Adam**

![](./images/Pasted%20image%2020230926160327.png)

___
***Additional Layer Types***

**Dense Layer**

![](./images/Pasted%20image%2020230926160657.png)

**Convolutional Layer**

![](./images/Pasted%20image%2020230926160933.png)

**Convolutional Neural Layer**

![](./images/Pasted%20image%2020230926161521.png)

## Practice quiz: Additional Neural Network Concepts

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/b08fec87-b710-4ece-9022-9dcf48ab1305image3.png?expiry=1695859200000&hmac=hPyqjbJ8-Yj8TR1gNiK45bvjG8T9xixayfxiAznEyyk)

1. The Adam optimizer is the recommended optimizer for finding the optimal parameters of the model. How do you use the Adam optimizer in TensorFlow?
+ A The call to model.compile() uses the Adam optimizer by default
+ B The call to model.compile() will automatically pick the best optimizer, whether it is gradient descent, Adam or something else. So there's no need to pick an optimizer manually.
+ C When calling model.compile, set optimizer=tf.keras.optimizers.Adam(learning_rate=le-3).
+ D The Adam optimizer works only with Softmax outputs So if a neural network has a Softmax output layer, TensorFlow will automatically pick the Adam optimizer.

My Answer: C $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Correct. Set the optimizer to Adam.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/b08fec87-b710-4ece-9022-9dcf48ab1305image4.png?expiry=1695859200000&hmac=pu1-1SJLSQCCTqEDC3vypfIiSSwJkl0IQsjH0apzE0c)

2. The lecture covered a different layer type where each single neuron of the layer doesnot look at all the values of the input vector that is fed into that layer. What is this name of the layer type discussed in lecture?
+ A convolutional layer
+ B lmage layer
+ C A fully connected layer
+ D 1D layer or 2D layer(depending on the input dimension)

My Answer: A ???

> [!check] Correct
> 
> Correct. For a convolutional layer, each neuron takes as input a subset of the vector that is fed into that layer.

## Back Propagation (Optional)

___
***What is a derivative? (Optional)***

**Derivative Example**

![](./images/Pasted%20image%2020230926163135.png)

**Informal Definition of Derivative**

![](./images/Pasted%20image%2020230926163430.png)

**More Derivative Examples**

![](./images/Pasted%20image%2020230926164042.png)

**Even More Derivative Examples**

```python
import sympy
J,w=sympy.symbols("J,w")
# J=w**2
J=w**3
# J=1/w
J
# output: w^3
dj_dw=sympy.diff(J,w)
dj_dw
# output: 3w^2
dj_dw.subs([(w,2)])
# output: 12
```

![](./images/Pasted%20image%2020230926165417.png)

**A note on derivative notation**

+ if $J(w)$ is a function of one variable (<font color="#00b0f0">w</font>),

d $\rightarrow$ $\frac{d}{dw}J(w)$

+ if $J(w_1,w_2,...,w_n)$ is a function of more than one variable,

$\partial$ $\rightarrow$ $\frac{\partial}{\partial w_i}J(w_1,w_2,...,w_n)$

simplified abbreviated forms: $\frac{\partial J}{\partial w_i}$ or $\frac{\partial}{\partial w_i}J$

> <font color="#00b0f0">"partial derivative"</font>

___
***Computation graph (Optional)***

**Small Neural Network Example**

![](./images/Pasted%20image%2020230926193921.png)

**Computing the Derivatives**

![](./images/Pasted%20image%2020230926195828.png)
![](./images/Pasted%20image%2020230926200107.png)

**Backprop is an efficient way to compute derivatives**

![](./images/Pasted%20image%2020230926200501.png)

___
***Larger neural network example (Optional)***

**Neural Network Example**

![](./images/Pasted%20image%2020230926201431.png)
![](./images/Pasted%20image%2020230926202016.png)

> the back prop procedure gives you a very efficient way to compute all of these derivatives. Which you can then feed into the gradient descent algorithm or the Adam optimization algorithm, to then **train the parameters** of your neural network.

___
***Optional Lab: Derivatives***

[Optional Lab: Derivatives](C2_W2_op/C2_W2_Derivatives.ipynb)

___
***Optional Lab: Back propagation***

[Optional Lab: Back propagation](C2_W2_op/C2_W2_Backprop.ipynb)


## Practice Lab: Neural network training

[Practice Lab: Neural network training](C2_W2/C2_W2_Assignment.ipynb)

# Week3 Advice for applying machine learning

Learning Objectives

- Evaluate and then modify your learning algorithm or data to improve your model's performance
- Evaluate your learning algorithm using cross validation and test datasets.
- Diagnose bias and variance in your learning algorithm
- Use regularization to adjust bias and variance in your learning algorithm
- Identify a baseline level of performance for your learning algorithm
- Understand how bias and variance apply to neural networks
- Learn about the iterative loop of Machine Learning Development that's used to update and improve a machine learning model
- Learn to use error analysis to identify the types of errors that a learning algorithm is making
- Learn how to add more training data to improve your model, including data augmentation and data synthesis
- Use transfer learning to improve your model's performance.
- Learn to include fairness and ethics in your machine learning model development
- Measure precision and recall to work with skewed (imbalanced) datasets

## Advice for applying machine learning

___
***Deciding what to try next***

some tips

**Debugging a learning algorithm**

![](./images/Pasted%20image%2020230926203108.png)

**Machine learning diagnostic**

Diagnostic:

A test that you run to gain insight into what is/isn't working with a learning algorithm, to gain guidance into improving its performance.

Diagnostics can take time to implement 
but doing so can be a very good use of your time

___
***Evaluating a model***

![](./images/Pasted%20image%2020230926203641.png)

**Evaluating your model**

![](./images/Pasted%20image%2020230926204259.png)

**Train/test procedure for linear regression (with squared error cost)**

![](./images/Pasted%20image%2020230926204550.png)
![](./images/Pasted%20image%2020230926204828.png)

**Train/test procedure for classification problem**

![](./images/Pasted%20image%2020230926205050.png)
![](./images/Pasted%20image%2020230926205207.png)

___
***Model selection and training/cross validation/test sets***

<font  face="Times New Roman" color=green size=5><u>2023-09-27 17:29 $\downarrow$</u></font>

**Model selection (choosing a model)**

![](./images/Pasted%20image%2020230927173122.png)

![](./images/Pasted%20image%2020230927174316.png)

> $w^{<d>}$: $d$: a $d$ order polynomial, the degree of polynomial

**Training/cross validation/test set**

+ training set
+ cross-validation set / validation set / development set / dev set
+ test set

![](./images/Pasted%20image%2020230927180329.png)

> The name **cross-validation** refers to that this is an extra dataset that we're going to use to check or cross check the validity or really the accuracy of different models.

Training error:

$$
J_{train}(\vec{w},b)=\frac{1}{2m_{train}}\biggl[\sum_{i=1}^{m_{train}}\bigl(f_{\vec{w},b}\bigl(\vec{x}^{(i)}\bigr)-y^{(i)}\bigr)^{2}\biggr]
$$

Cross validation error, validation error, dev error:

$$
J_{cv}(\vec{w},b)=\frac{1}{2m_{cv}}\biggl[\sum_{i=1}^{m_{cv}}\Bigl(f_{\vec{w},b}\left(\vec{x}_{cv}^{(i)}\right)-y_{cv}^{(i)}\Bigr)^{2}\biggr]
$$

Test error:

$$
J_{test}(\vec{w},b)=\frac{1}{2m_{test}}\biggl[\sum_{i=1}^{m_{test}}\biggl(f_{\vec{w},b}\left(\vec{x}_{test}^{(i)}\right)-y_{test}^{(i)}\biggr)^{2}\biggr]
$$

**Model selection**

![](./images/Pasted%20image%2020230927181330.png)

**Model selection - choosing a neural network architecture**

![](./images/Pasted%20image%2020230927181845.png)

> [!NOTE] Notes
> 
> Training set $\rightarrow$ dev set: pick model that has the lowest cross validation error $\rightarrow$ test set: an estimate of the generalization error

___
***Optional Lab: Model Evaluation and Selection***


## Practice quiz: Advice for applying machine learning

1. In the context of machine learning, what is a diagnostic?
+ A. A test that you run to gain insight into what is/isn't working with a learning algorithm.
+ B. An application of machine learning to medical applications, with the goal of diagnosing patients'conditions.
+ C. A process by which we quickly try as many different ways to improve an algorithm as possible, so as to see what works.
+ D. This refers to the process of measuring how well a learning algorithm does on a test set (data that the algorithm was not trained on).

My Answer: D 

> [!failure] Incorrect
> 
> A diagnostic does involve measurements, but not just on a test set.

ChatGPT Answer: A

> [!check] A Correct
> 
> Yes! A diagnostic is a test that you run to gain insight into what is/isn’t working with a learning algorithm, to gain guidance into improving its performance.

2. True/False? lt is always true that the better an algorithm does on the training set, the better it will do on generalizing to new data.

My Answer: False $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Actually, if a model overfits the training set, it may not generalize well to new data.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/392a50a6-14d1-40c4-9a21-b1f12202c20fimage3.png?expiry=1695945600000&hmac=aZEvXpt2tdNXDmy8BuDb7dxQnegE86LObKcMTxCTTEE)

3. For a classification task; suppose you train three different models using three different neural network architectures. Which data do you use to evaluate the three models in order to choose the best one?
+ A. The test set
+ B. All the data -- training, cross validation and test sets put together.
+ C. The cross validation set
+ D. The training set

My Answer: C $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Correct. Use the cross validation set to calculate the cross validation error on all three models in order to compare which of the three models is best.


## Bias and variance

___
***Diagnosing bias and variance***

**bias / variance**

![](./images/Pasted%20image%2020230927194832.png)

**Understanding bias and variance**

![](./images/Pasted%20image%2020230927195614.png)

**Diagnosing bias and variance**

![](./images/Pasted%20image%2020230927195949.png)

___
***Regularization and bias/variance***

<font  face="Times New Roman" color=green size=5><u>2023-09-29 12:44 $\downarrow$</u></font>

**Linear regression with regularization**

![](./images/Pasted%20image%2020230929124811.png)

**Choosing the regularization parameter $\lambda$**

![](./images/Pasted%20image%2020230929125153.png)

**Bias and variance as a function of regularization parameter $\lambda$** 

![](./images/Pasted%20image%2020230929125644.png)

___
***Establishing a baseline level of performance***

**Speech recognition example**

![](./images/Pasted%20image%2020230929130153.png)

**Establishing a baseline level of performance**

What is the level of error you can reasonably hope to get to?
+ Human level performance
+ Competing algorithms performance
+ Guess based on experience

**Bias/variance examples**

![](./images/Pasted%20image%2020230929130738.png)

___
***Learning curves***

![](./images/Pasted%20image%2020230929131242.png)

**High bias**

![](./images/Pasted%20image%2020230929131648.png)

**High variance**

![](./images/Pasted%20image%2020230929132110.png)

___
***Deciding what to try next revisited***

**Debugging a learning algorithm**

You've implemented regularized linear regression on housing prices

$$
J({\vec{w}},b)={\frac{1}{2m}}\sum_{i=1}^{m}\bigl(f_{{\vec{w}},b}\bigl({\vec{x}}^{(i)}\bigr)-y^{(i)}\bigr)^{2}+{\frac{\lambda}{2m}}\sum_{j=1}^{n}w_{j}^{2}
$$

But it makes unacceptably large errors in predictions. What do you try next?

| solution                                                  | purpose             |
| --------------------------------------------------------- | ------------------- |
| Get more training examples                                | fixed high variance |
| Try smaller sets of features                              | fixed high variance |
| Try getting additional features                           | fixed high bias     |
| Try adding polynomial features $(x_1^2,x_2^2,x_1x_2,etc)$ | fixed high bias     |
| Try decreasing $\lambda$                                  | fixed high bias     |
| Try increasing $\lambda$                                  | fixed high variance |

___
***Bias/variance and neural networks***

**The bias variance tradeoff**

![](./images/Pasted%20image%2020230929145442.png)

**Neural networks and bias variance**

Large neural networks are low bias machines

> if you make your neural network large enough, you can almost always fit your training set well.

![](./images/Pasted%20image%2020230929150207.png)

**Neural networks and regularization**

![](./images/Pasted%20image%2020230929150444.png)

A large neural network will usually do as **well or better** than a smaller one so long as **regularization** is chosen appropriately

**Neural network regularization**

![](./images/Pasted%20image%2020230929150735.png)

___
***Optional Lab: Diagnosing Bias and Variance***

[Optional Lab: Diagnosing Bias and Variance](C2W3/C2W3_Lab_02_Diagnosing_Bias_and_Variance.ipynb)

## Practice quiz: Bias and variance

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/0597681e-6ccc-4f13-bd62-6c322925cf90image5.png?expiry=1696118400000&hmac=y7jgcz0_GbRJGs6w9gP_PPwVvWc7opQm67flG9kqnWY)

1. If the model's cross validation error $J_cv$ is much higher than the training error $J_train$; this is an indication that the model has...
+ A. Low variance
+ B. Low bias
+ C. high variance
+ D. high bias

My Answer: C $\checkmark[ChatGPT]$

> [!check] Correct
> 
> When $J_{cv}​>>J_{train}$​ (whether $J_{train}$​ is also high or not, this is a sign that the model is overfitting to the training data and performing much worse on new examples.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/0597681e-6ccc-4f13-bd62-6c322925cf90image3.png?expiry=1696118400000&hmac=TWsmLLooVr_rZBA2ez0zVM6xNHFxRm94MHNY0zobsag)

2. Which of these is the best way to determine whether your model has high bias (has underfit the training data)?
+ A. See if the cross validation error is high compared to the baseline level of performance
+ B. Compare the training error to the cross validation error.
+ C. See if the training error is high (above 15% or so)
+ D. Compare the training error to the baseline level of performance

My Answer: D

CHatGPT Answer: B ×

> [!check] D Correct
> 
> D. Compare the training error to the baseline level of performance
> 
> Correct. If comparing your model's training error to a baseline level of performance (such as human level performance, or performance of other well-established models), if your model's training error is much higher, then this is a sign that the model has high bias (has underfit).

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/0597681e-6ccc-4f13-bd62-6c322925cf90image4.png?expiry=1696118400000&hmac=UCyYRckDE5vciTdOQkSCJeA9wo8DSLaiokqxHIyeODM)

3. You find that your algorithm has high bias. Which of these seem like good options for improving the algorithm's performance? Hint: two of these are correct.
+ A. Collect additional features or add polynomial features
+ B. Decrease the regularization parameter $\lambda$ (lambda)
+ C. Remove examples from the training set
+ D. Collect more training examples

My Answer: AB

ChatGPT Answer: AD

> [!check] Correct
> 
> + Collect additional features or add polynomial features
> > Correct. More features could potentially help the model better fit the training examples.
> 
> + Decrease the regularization parameter $\lambda$ (lambda)
> > Correct. Decreasing regularization can help the model better fit the training data.

4. You find that your algorithm has a training error of 2%, and a cross validation error of 20% (much higher than the training error). Based on the conclusion you would draw about whether the algorithm has a high bias or high variance problem, which of these seem like good options for improving the algorithm's performance? Hint: two of these are correct.
+ A. Collect more training data
+ B. Reduce the training set size
+ C. Increase the regularization parameter $\lambda$
+ D. Decrease the regularization parameter $\lambda$

My Answer: AC

ChatGPT Answer: AD ×

> [!check] Correct
> 
> + Collect more training data
> > Yes, the model appears to have high variance (overfit), and collecting more training examples would help reduce high variance.
> 
> + Increase the regularization parameter $\lambda$
> > Yes, the model appears to have high variance (overfit), and increasing regularization would help reduce high variance.


## Machine learning development process

___
***Iterative loop of ML development***

![](./images/Pasted%20image%2020230929153235.png)

**Spam classification example**

![](./images/Pasted%20image%2020230929153342.png)

**Building a spam classifier**

![](./images/Pasted%20image%2020230929153609.png)

How to try to reduce your spam classifier's error?
+ Collect more data. E.g., "Honeypot" project.
+ Develop sophisticated features based on email routing (from email header)
+ Define sophisticated features from email body. E.g, should "discounting" and "discount" be treated as the same word.
+ Design algorithms to detect misspellings. E.g., w4tches, med1cine, mOrtgage.

___
***Error analysis***

![](./images/Pasted%20image%2020230929155223.png)

___
***Adding data***

![](./images/Pasted%20image%2020230929164307.png)

**Data augmentation**

Augmentation: modifying an existing training example to create a new training example

![](./images/Pasted%20image%2020230929164634.png)

**Data augmentation by introducing distortions**

![](./images/Pasted%20image%2020230929164731.png)

**Data augmentation for speech**

![](./images/Pasted%20image%2020230929164950.png)

> One tip for data augmentation is that the changes or the distortions you make to the data, should be representative of the types of noise or distortions in the test set. 

**Data augmentation by introducing distortions**

![](./images/Pasted%20image%2020230929165300.png)

**Data synthesis**

Synthesis: using artificial data inputs to create a new training example.

**Artificial data synthesis for photo OCR**

![](https://www.publicdomainpictures.net/pictures/10000/velka/2185-1267942684soD9.jpg)
![](./images/Pasted%20image%2020230929165741.png)

**Engineering the data used by your system**

![](./images/Pasted%20image%2020230929165926.png)

___
***Transfer learning: using data from a different task***

**Transfer learning**

![](./images/Pasted%20image%2020230929171556.png)

**Why does transfer learning work?**

![](./images/Pasted%20image%2020230929171841.png)

**Transfer learning summary**

![](./images/Pasted%20image%2020230929172159.png)

> GPT3, BERTs, neural networks pre-trained in ImageNet: Pre-trained $\rightarrow$ fine tune

___
***Full cycle of a machine learning project***

<font  face="Times New Roman" color=green size=5><u>2023-09-30 16:40 $\downarrow$</u></font>

![](./images/Pasted%20image%2020230930164501.png)

**Deployment**

![](./images/Pasted%20image%2020230930165134.png)

> MLOps: this refers to the practice of how to systematically build and deploy and maintain machine learning systems.

___
***Fairness, bias, and ethics***

**Bias**

- Hiring tool that discriminates against women.
- Facial recognition system matching dark skinned individuals to criminal mugshots.
- Biased bank loan approvals.
- Toxic effect of reinforcing negative stereotypes.

**Adverse use cases**

Deepfakes

![](./images/Pasted%20image%2020230930165639.png)

+ Spreading toxic/incendiary speech through optimizing for engagement.
+ Generating fake content for commercial or political purposes.
+ Using ML to build harmful products, commit fraud etc.
+ Spam vs anti-spam : fraud vs anti-fraud.

**Guidelines**

+ Get a diverse team to brainstorm things that might go wrong, with emphasis on possible harm to vulnerable groups
+ Carry out literature search on standards/guidelines for your industry.
+ Audit systems against possible harm prior to deployment.
![](./images/Pasted%20image%2020230930170600.png)
+ Develop mitigation plan (if applicable), and after deployment, monitor for possible harm.

## Practice quiz: Machine learning development process

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/25ef86da-6935-47cf-a4af-8945fa6a0ed2image5.png?expiry=1696204800000&hmac=7EUpqYddkANkEPsbQEDLd6mY4m_dtPU2Lh7ZgqLb_wY)

1. Which of these is a way to do error analysis?
+ A. Manually examine a sample of the training examples that the model misclassified in order to identify common traits and trends.
+ B. Calculating the training error $J_{train}$
+ C. Collecting additional training data in order to help the algorithm do better.
+ D. Calculating the test error $J_{test}$

My Answer: A $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Correct. By identifying similar types of errors, you can collect more data that are similar to these misclassified examples in order to train the model to improve on these types of examples.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/25ef86da-6935-47cf-a4af-8945fa6a0ed2image3.png?expiry=1696204800000&hmac=3AU5akUuupk60yMEeeiBKozG2XzVx5YFf13lz-S8bxE)

2. We sometimes take an existing training example and modify it (for example, by rotating an image slightly) to create a new example with the same label. What is this process called?
+ A. Error analysis
+ B. Data augmentation
+ C. Machinelearning diagnostic
+ D. Bias/variance analysis

My Answer: B $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Yes! Modifying existing data (such as images, or audio) is called data augmentation.


![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/25ef86da-6935-47cf-a4af-8945fa6a0ed2image4.png?expiry=1696204800000&hmac=8lh6GpfDPSWegVjbe7E7oE45PL5A-WOhenusB3c7hYs)

3. What are two possible ways to perform transfer learning? Hint: two of the four choices are correct.
+ A. Given a dataset, pre-train and then further fine tune a neural network on the same dataset.
+ B. Download a pre-trained model and use it for prediction without modifying or re-training it.
+ C. You can choose to train all parameters of the model, including the output layers, as well as the earlier layers.
+ D. You can choose to train just the output layers' parameters and leave the other parameters of the model fixed.

My Answer: CD 

> [!check] Correct
> 
> + You can choose to train all parameters of the model, including the output layers, as well as the earlier layers.
> 
> > Correct. It may help to train all the layers of the model on your own training set. This may take more time compared to if you just trained the parameters of the output layers.
> 
> + You can choose to train just the output layers' parameters and leave the other parameters of the model fixed.
> 
> > Correct. The earlier layers of the model may be reusable as is, because they are identifying low level features that are relevant to your task.


## Skewed datasets (optional)

***Error metrics for skewed datasets***

**Rare disease classification example**

![](./images/Pasted%20image%2020230930172644.png)

**Precision/recall**

$y = 1$ in presence of rare class we want to detect.

![](./images/Pasted%20image%2020230930174054.png)

___
***Trading off precision and recall***

![](./images/Pasted%20image%2020230930175337.png)

**F1 Score**

How to compare precision/recall numbers?

![](./images/Pasted%20image%2020230930180541.png)

> the **harmonic mean** is a way of taking an average that emphasizes the **smaller** values more.

## Practice Lab: Advice for applying machine learning

[Practice Lab: Advice for applying machine learning](C2_W3/C2_W3_Assignment.ipynb)

# Week4 Decision trees

Learning Objectives

- See what a decision tree looks like and how it can be used to make predictions
- Learn how a decision tree learns from training data
- Learn the "impurity" metric "entropy" and how it's used when building a decision tree
- Learn how to use multiple trees, "tree ensembles" such as random forests and boosted trees
- Learn when to use decision trees or neural networks

## Decision trees

***Decision tree model***

**Cat classification example**

![](./images/Pasted%20image%2020230930193735.png)

**Decision tree**

![](./images/Pasted%20image%2020230930194136.png)
![](./images/Pasted%20image%2020230930194254.png)

___
***Learning Process***

<font  face="Times New Roman" color=green size=5><u>2023-10-01 14:45 $\downarrow$</u></font>

**Decision Tree Learning**

![](./images/Pasted%20image%2020231001144930.png)

**Decision 1**: How to choose what feature to split on at each node?

![](./images/Pasted%20image%2020231001145746.png)

**Decision 2**: When do you stop splitting?
+ When a node is 100% one class
+ When splitting a node will result in the tree exceeding a maximum depth
+ When improvements in purity score are below a threshold
+ When number of examples in a node is below a threshold

## Practice quiz: Decision trees

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/2f88deea-2113-4c00-9f88-79e87fa4e329image3.png?expiry=1696291200000&hmac=3WkzNQUfiD-Umt6YgLy57vvcF8ULjrZGEaFYuciE2Og)

1. Based on the decision tree shown in the lecture, if an animal has floppy ears, a round face shape and has whiskers, does the model predict that it's a cat or not a cat?
+ cat
+ Not a cat

My Answer: Cat

> [!check] Correct
> 
> Correct. If you follow the floppy ears to the right, and then from the whiskers decision node, go left because whiskers are present, you reach a leaf node for "cat", so the model would predict that this is a cat.


![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/2f88deea-2113-4c00-9f88-79e87fa4e329image4.png?expiry=1696291200000&hmac=exbw6mkjewLxfCsmEBLuHpfjJx83U7FvQeJ66y2pG8s)

2. Take a decision tree learning to classify between spam and non-spam email. There are 20 training examples at the root note, comprising 10 spam and 10 non-spam emails. lf the algorithm can choose from among four features, resulting in four corresponding splits, which would it choose (i.e., which has highest purity)?
+ A. Left split: 2 of 2 emails are spam. Right split: 8 of 18 emails are spam.
+ B. Left split: 10 of 10 emails are spam. Right split: 0 of 10 emails are spam.
+ C. Left split: 7 of 8 emails are spam. Right split: 3 of 12 emails are spam.
+ D. Left split: 5 of 10 emails are spam, Right split: 5 of 10 emails are spam.

My Answer: B

> [!check] Correct
> 
> Yes!

## Decision tree learning

___
***Measuring purity***

**Entropy as a measure of impurity**

![](./images/Pasted%20image%2020231001151536.png)
![](./images/Pasted%20image%2020231001151825.png)

___
***Choosing a split: Information Gain***

<font  face="Times New Roman" color=green size=5><u>2023-10-02 18:32 $\downarrow$</u></font>

**Choosing a split**

![](./images/Pasted%20image%2020231002184305.png)

**Information Gain**

![](./images/Pasted%20image%2020231002184539.png)

___
***Putting it together***

**Decision Tree Learning**

+ Start with all examples at the root node
+ Calculate information gain for all possible features, and pick the one with the highest information gain
+ Split dataset according to selected feature, and create left and right branches of the tree
+ Keep repeating splitting process until stopping criteria is met:
	- When a node is 100% one class
	- When splitting a node will result in the tree exceeding a maximum depth
	- Information gain from additional splits is less than threshold 
	- When number of examples in a node is below a threshold

**Recursive splitting**

![](./images/Pasted%20image%2020231002185718.png)

___
***Using one-hot encoding of categorical features***

**Features with three possible values**

![](./images/Pasted%20image%2020231002190052.png)

**One hot encoding**

![](./images/Pasted%20image%2020231002190323.png)

If a categorical feature can take on $k$ values, create $k$ binary features (0 or 1 valued).

**One hot encoding and neural networks**

![](./images/Pasted%20image%2020231002190722.png)

___
***Continuous valued features***

<font  face="Times New Roman" color=green size=5><u>2023-10-03 19:47 $\downarrow$</u></font>

**Continuous features**

![](./images/Pasted%20image%2020231003194929.png)

**Splitting on a continuous variable**

![](./images/Pasted%20image%2020231003195630.png)

___
***Regression Trees (optional)***

**Regression with Decision Trees: Predicting a number**

![](./images/Pasted%20image%2020231003195839.png)

**Regression with Decision Trees**

![](./images/Pasted%20image%2020231003200142.png)

**Choosing a split**

![](./images/Pasted%20image%2020231003203535.png)

___
***Lab: Optional Lab: Decision Trees***


## Practice quiz: Decision tree learning

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/f1fed11a-1ade-4b8c-b5cf-e6f9b11a2b23image4.png?expiry=1696464000000&hmac=K6qB45zR0-hnPaiuBIW9R-hyRIUxeajo9YnYlkUJ0XY)

1. Recall that entropy was defined in lecture as $H(p_1)=-p_1 log_2(p_1)-p_0 log_2(p_0)$, where $p_1$ is the fraction of positive examples and $p_0$ the fraction of negative examples.
At a given node of a decision tree,  6 of 10 examples are cats and 4 of 10 are not cats. Which expression calculates the entropy $H(p_1)$ ofthis group of 10 animals?

+ A. $(0.6)log_2(0.6)+(1-0.4)log_2(1-0.4)$
+ B. $(0.6)log_2(0.6)+(0.4)log_2(0.4)$
+ C. $-(0.6)log_2(0.6)-(0.4)log_2(0.4)$
+ D. $-(0.6)log_2(0.6)-(1-0.4)log_2(1-0.4)$

My Answer: C

> [!check] Correct
> 
> Correct. The expression is $-(p_1)log_2(p_1)-(p_0)log_2(p_0)$

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/f1fed11a-1ade-4b8c-b5cf-e6f9b11a2b23image2.png?expiry=1696464000000&hmac=m0UDiA1-OuBLRQdWBPlGV858NN3srI5jXLomFdpFiNc)

2. Recall that information was defined as follows:

$H(p_1^{root})-\left(w^{left}H(p_1^{left})+w^{right}H(p_1^{right})\right)$

Before a split, the entropy of a group of 5 cats and 5 non-cats is $H(5/10)$. After splitting on a particular feature, a group of 7 animals (4 of which are cats) has an entropy of $H(4/7)$. The other group of3 animals (1 is a cat) and has an entropy of $H(1/3)$. What is the expression for information gain?

+ A. $H(0.5)-\left(\frac47*H(4/7)+\frac47*H(1/3)\right)$
+ B. $H(0.5)-(7*H(4/7)+3*H(1/3))$
+ C. $H(0.5)-(H(4/7)+H(1/3))$
+ D. $H(0.5)-\left(\frac7{10}H(4/7)+\frac3{10}H(1/3)\right)$

My Answer: D $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Correct. The general expression is $H(p_1^{root})-\left(w^{left}H(p_1^{left})+w^{right}H(p_1^{right})\right)$

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/f1fed11a-1ade-4b8c-b5cf-e6f9b11a2b23image5.png?expiry=1696464000000&hmac=m6trgSDIuq3KD2FIZpMRc1u3XOYmnQ3ejuSFCjDH7M4)

3. To represent 3 possible values for the ear shape, you can define 3 features for ear shape: pointy ears, floppy ears, oval ears. For an animal whose ears are not pointy, not floppy, but are oval, how can you represent this information as a feature vector?
+ A. [1,1,0]
+ B. [0,1,0]
+ C. [0,0,1]
+ D. [1,0,0]

My Answer: C

> [!check] Correct
> 
> Yes! 0 is used to represent the absence of that feature (not pointy, not floppy), and 1 is used to represent the presence of that feature (oval).

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/f1fed11a-1ade-4b8c-b5cf-e6f9b11a2b23image6.png?expiry=1696464000000&hmac=dTBy3n2Tp7scUbMHryiuldznT1v2KLHpvyyzZrOJZ3I)

4. For a continuous valued feature (such as weight of the animal), there are 10 animals in the dataset. According to the lecture, what is the recommended way to find the best split for that feature?
+ A. Use a one-hot encoding to turn the feature into a discrete feature vector of 0's and 1's, then apply the algorithm we had discussed for discrete features.
+ B. Try every value spaced at regular intervals (e.g., 8, 8.5, 9, 9.5, 10, etc,) and find the split that gives the highest information gain.
+ C. Use gradient descent to find the value of the split threshold that gives the highest information gain.
+ D. Choose the 9 mid-points between the 10 examples as possible splits, and find the split that gives the highest information gain.

My Answer: C

> [!failure] Incorrect
> 
> C Incorrect

> [!check] Correct Answer
> 
> D.   Choose the 9 mid-points between the 10 examples as possible splits, and find the split that gives the highest information gain.
> 
> Correct. This is what is proposed in the lectures.


5. Which of these are commonly used criteria to decide to stop splitting? (Choose two.)
+ A. When the tree has reached a maximum depth
+ B. When a node is 50% one class and 50% another class (highest possible value of entropy)
+ C. When the information gain from additional splits is too large
+ D. When the number of examples in a node is below a threshold

My Answer: AD $\checkmark[ChatGPT]$

> [!check] Correct
> 
> + When the tree has reached a maximum depth
> + When the number of examples in a node is below a threshold

## Tree ensembles

___
***Using multiple decision trees***

> One of the weaknesses of using a single decision tree is that that decision tree can be highly sensitive to small changes in the data. $\Rightarrow$ a tree ensemble

**Trees are highly sensitive to small changes of the data**

![](./images/Pasted%20image%2020231003210801.png)

**Tree ensemble**

> run all decision trees

![](./images/Pasted%20image%2020231003210940.png)

___
***Sampling with replacement***

![](./images/Pasted%20image%2020231003211438.png)
![](./images/Pasted%20image%2020231003211741.png)

> the key building block for building an ensemble of trees.

___
***Random forest algorithm***

**Generating a tree sample**

![](./images/Pasted%20image%2020231003221938.png)

> It turns out that setting capital B to be **larger**, never hurts performance, but beyond a certain point, you end  up with diminishing returns and it **doesn't** actually get that much better when B is much larger than say 100 or so.

**Randomizing the feature choice**

At each node, when choosing a feature to use to split, if n features are available,pick a random subset of $k < n$ features and allow the algorithm to only choose from that subset of features.

> $n$ is large: $k=\sqrt{n}$

Random forest algorithm

> One way to think about why this is more robust to than a single decision tree is the something with replacement procedure causes the algorithm to explore a lot of small changes to the data already and it's training different decision trees and is averaging over all of those changes to the data that the something with replacement procedure causes.
> 
> And so this means that any little change further to the training set makes it less likely to have a huge impact on the overall output of the overall random forest algorithm. Because it's already explored and it's averaging over a lot of small changes to the training set.

___
***XGBoost***

<font  face="Times New Roman" color=green size=5><u>2023-10-04 15:37 $\downarrow$</u></font>

**Boosted trees intuition**

![](./images/Pasted%20image%2020231004154343.png)

> So rather than looking at all the training examples, we focus more attention on the subset of examples is **not yet doing well** on and get the new decision tree, the next decision tree reporting ensemble to try to do well on them.

**XGBoost (eXtreme Gradient Boosting)**

+ Open source implementation of boosted trees
+ Fast efficient implementation
+ Good choice of default splitting criteria and criteria for when to stop splitting
+ Built in regularization to prevent overfitting
+ Highly competitive algorithm for machine learning competitions (eg: Kaggle competitions)

**Using XGBoost**

1. Classification

```python
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

2. Regression

```python
from xgboost import XGBRegressor

model = XGBRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

___
***When to use decision trees***

**Decision Trees vs Neural Networks**

<u>Decision Trees and Tree ensembles</u>
+ Works well on tabular (**structured**) data
+ Not recommended for unstructured data (images, audio, text)
+ **Fast** ![](./images/Pasted%20image%2020231004155508.png)
+ Small decision trees may be human interpretable

<u>Neural Networks</u>
+ Works well on all types of data, including tabular (structured) and **unstructured** data
+ May be **slower** than a decision tree
+ Works with transfer learning
+ When building a system of multiple models working together, it might be easier to string together multiple neural networks

___
***Lab: Optional Lab: Tree Ensembles***

[Lab: Optional Lab: Tree Ensembles](C2W4/C2_W4_Lab_02_Tree_Ensemble.ipynb)

## Practice quiz: Tree ensembles

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/178705ec-6d16-47e3-9bdd-b8d5fb8e8e98image3.png?expiry=1698537600000&hmac=wQk0LHp4ccRG1CNqAhufN8kIV8DrxtpniQXUpAGAK_A)

1. For the random forest, how do you build each individual tree so that they are not all identical to each other?
+ A. Sample the training data without replacement
+ B. Sample the training data with replacement and select a random subset of features to build each tree
+ C. Train the algorithm multiple times on the same training set. This will naturally result in different trees.
+ D. If you are training B trees, train each one on 1/B of the training set, so each tree is trained on a distinct set of examples.

My Answer: B

> [!check] Correct
> 
> + A: Sample the training data with replacement and select a random subset of features to build each tree
> 
> Correct.  You can generate a training set that is unique for each individual tree by sampling the training data with replacement. The random forest algorithm further avoids identical trees by randomly selecting a subset of features when building the tree ensemble.

2. You are choosing between a decision tree and a neural network for a classification task where the input $x$ is a 100x100 resolution image. Which would you choose?
+ A. A neural network, because the input is unstructured data and neural networks typically work better with unstructured data.
+ B. A decision tree, because the input is structured data and decision trees typically work better with structured data.
+ C. A neural network, because the input is structured data and neural networks typically work better with structured data.
+ D. A decision tree, because the input is unstructured and decision trees typically work better with unstructured data.

My Answer: A

> [!check] Correct
> 
> Yes!

3. What does sampling with replacement refer to?
+ A. Drawing a sequence of examples where, when picking the next example, first remove all previously drawn examples from the set we are picking from.
+ B. lt refers to using a new sample of data that we use to permanently overwrite (that is, to replace) the original data.
+ C. Drawing a sequence of examples where, when picking the next example, first replacing all previously drawn examples into the set we are picking from.
+ D. It refers to a process of making an identical copy of the training set.

My Answer: D

> [!failure] Incorrect
> 
> D Incorrect

> [!check] Correct Answer
> 
> Drawing a sequence of examples where, when picking the next example, first replacing all previously drawn examples into the set we are picking from.

## Practice Lab: Decision Trees

[Practice Lab: Decision Trees](C2_W4/C2_W4_Decision_Tree_with_Markdown.ipynb)


## Conversations with Andrew (Optional

***Andrew Ng and Chris Manning on Natural Language Processing***

+ Chris, PhD: linguistics, NLP papers cited most
+ Andrew, Chris: triple majors when undergraduate
+ the biggest application of NLP is Web search

## Acknowledgments

<font  face="Times New Roman" color=green size=5><u>2023-10-04 17:25 $\Uparrow$</u></font>

Check Quiz Answer

<font  face="Times New Roman" color=green size=5><u>2023-10-27 20:49 $\Uparrow$</u></font>