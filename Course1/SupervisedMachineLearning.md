---
aliases: 
author: Li Yaozong
date: 2023-09-11
time: 2023-09-11 15:33
cover: https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://coursera-course-photos.s3.amazonaws.com/9c/90ae67ecdb4185a4ae79ec9a5ae0b6/Course-Logo--1.png
description: "Supervised Machine Learning: Regression and Classification"
source: Coursera
status: Completed
tags:
  - cs
  - MachineLearing
  - notes
url: https://www.coursera.org/learn/machine-learning/home/welcome
---

![cover](https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://coursera-course-photos.s3.amazonaws.com/9c/90ae67ecdb4185a4ae79ec9a5ae0b6/Course-Logo--1.png)

# Week1 Introduction to Machine Learning

Learning Objectives

- Define machine learning
- Define supervised learning
- Define unsupervised learning
- Write and run Python code in Jupyter Notebooks
- Define a regression model
- Implement and visualize a cost function
- Implement gradient descent
- Optimize a regression model using gradient descent

<font  face="Times New Roman" color=green size=5><u>2023-09-11 $\downarrow$ </u></font>

## Overview of Machine Learning

## Supervised vs. Unsupervised Machine Learning

***What is machine learning?***

> If Arthur Samuel's checkers-playing program had been allowed to play only 10 games (instead of tens of thousands games) against itself, how would this have affected its performance?
> 
> + Would have made it better
> + ==Would have made it worse== $\checkmark$

Machine learning algorithms

- Supervised learning
- Unsupervised learning
- Recommender systems
- Reinforcement learning

---
***Supervised learning***

<font  face="Times New Roman" color=green size=5><u>2023-09-12 $\downarrow$ </u></font>

Learns from being given "<font color=red>right answers</font>"

**Regression**: Housing price prediction
+ x: House size in feet<sup>2</sup>
+ y: Price in $ 1000's

**Classification**: Breast cancer detection
+ x: tumor size (diameter in cm)
+ Y: 0-benign, 1-malignant

class / category

predict categories

| Regression                       | Classification                   |
| -------------------------------- | -------------------------------- |
| Predict a number                 | predict categories               |
| infinitely many possible outputs | small number of possible outputs |

---
***Unsupervised learning***

Clustering
+ Google News
+ DNA microarray
+ Grouping customers

Unsupervised learning
+ Data only comes with inputs x, but not output labels y.
+ Algorithm has to find <font color="#00b0f0">structure</font> in the data.

> [!question] Quiz
> 
> Of the following examples, which would you address using an unsupervised learning algorithm?  (Check all that apply.)
> 
> + ==Given a set of news articles found on the web, group them into sets of articles about the same stories.== $\checkmark$
> + Given email labeled as spam/not spam, learn a spam filter.
> + ==Given a database of customer data, automatically discover market segments and group customers into different market segments.== $\checkmark$
> + Given a dataset of patients diagnosed as either having diabetes or not, learn to classify new patients as having diabetes or not.

---
***Jupyter Notebooks***

[Jupyter Notebooks](C1W1/C1_W1_Lab01_Python_Jupyter_Soln.ipynb)

## Practice Quiz: Supervised vs unsupervised learning




## Regression Model

***Linear regression model***

![](./images/Pasted%20image%2020230912195928.png)

**Terminology**
+ Training set: Data used to train the model
+ Notation
	- $x$: "input" variable / feature
	- $y$: "output" variable / "target" variable
	- $m$: number of training examples
	- $(x, y)$: single training example
	- $(x^{(i)}, y^{(i)})$: $i^{th}$ training example ($1^{st}, 2^{nd}, 3^{rd}$, ...)

![](./images/Pasted%20image%2020230912205512.png)

___
***Optionallab: Model representation***

[Model representation](C1W1/C1_W1_Lab02_Model_Representation_Soln.ipynb)

***
***Cost function formula***

Model: $f_{w,b}(x)=wx+b$
> $w,b$: parameters/coefficients/weights

Cost function: Squared error cost function

$$
\begin{align}
J(w,b)&=\frac{1}{2m}\sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2 \\
&=\frac{1}{2m}\sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2 \\
\end{align}
$$

> m=number of training examples

![](./images/Pasted%20image%2020230912221233.png)

> [!question] Quiz
> 
> The cost function used for linear regression is 
> 
> $$
> J(w,b)=\frac{1}{2m}\sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
> $$
> 
> Which of these are the parameters of the model that can be adjusted?
> + == $w$ and $b$== $\checkmark$
> + $f_{w,b}(x^{(i)})$
> + $w$ only, because we should choose $b$=0
> + $\hat{y}$
> 
> $w$ and $b$ are parameters of the model, adjusted as the model learns from the data. They’re also referred to as “**coefficients**” or “**weights**”

___
***Cost function intuition***

model:

$$
f_{w,b}(x)=wx+b
$$

parameters:

$$
w,b
$$

cost function:

$$
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
$$

goal:

$$
\underset{w,b}{\operatorname*{minimize}} J(w,b)
$$

simplified: $b=\varnothing$

![](./images/Pasted%20image%2020230912220541.png)
![](./images/Pasted%20image%2020230912221139.png)

> [!question] Quiz
> 
> When does the model fit the data relatively well, compared to other choices for parameter $w$?
> + ==When the cost $J$ is at or near a minimum.==$\checkmark$
> + When $f_w​(x)$ is at or near a minimum for all the values of x in the training set.
> + When $x$ is at or near a minimum.
> + When $w$ is close to zero.
> 
> > 正确  
> > 
> > When the cost is relatively small, closer to zero, it means the model fits the data better compared to other choices for w and b.

___
***Visualizing the cost function***

<font  face="Times New Roman" color=green size=5><u>2023-09-13 16:07 $\downarrow$</u></font>

contour plots: 等高线图

___
***Visualization examples***

![](./images/Pasted%20image%2020230913162907.png)

___
***Optionallab: Cost function***

[Cost function](C1W1/C1_W1_Lab03_Cost_function_Soln.ipynb)

## Practice Quiz: Regression Model

1.   For linear regression, the model is $f_{w,b}(x)=wx+b$.

Which of the following are the inputs, or features, that are fed into the model and with which the model is expected to make a prediction?

+ A. x
+ B. w and b.
+ C. m
+ D. (x,y)

My Answer: D ×

> [!failure] Incorrect
> 
> Although $x$ are the input features, $y$ are the labels. The model does not use the labels to make its predictions.

A. $x$

> [!check] Correct
> 
> The $x$, the input features, are fed into the model to generate a prediction $f_{w,b}​(x)$


2. For linear regression, if you find parameters $w$ and $b$ so that $J(w,b)$ is very close to zero, what can you conclude?

+ A. The selected values of the parameters $w$ and $b$ cause the algorithm to fit the training set really well.
+ B. This is never possible -- there must be a bug in the code.
+ C. The selected values of the parameters $w$ and $b$ cause the algorithm to fit the training set really poorly.

My Answer: A

> [!check] Correct
> 
> When the cost is small, this means that the model fits the training set well.


## Train the model with gradient descent

***Gradient descent***

the most advanced neural network models/Deep learning models

+ Have some function $J(w,b)$
+ Want $\underset{w,b}{\operatorname*{min}}(w,b)$
+ Outline: 
	- Start with xome $w,b$
	- Keep changing $w,b$ to reduce $J(w,b)$
	- Until we settle at or near a minimum

![](./images/Pasted%20image%2020230913170121.png)

___
***Implementing gradient descent***

Gradient descent algorithm

$w=w-\alpha\frac{\partial}{\partial w}J(w,b)$

> + $\alpha$: Learning rate
> + $\frac{\partial}{\partial w}$: Derivative

$b=b-\alpha \frac{\partial}{\partial b}J(w,b)$

Simultaneously update $w$ and $b$

**Correct**: Simultaneous update

+ $tmp\_w=w-\alpha\frac\partial{\partial w}J(w,b)$
+ $tmp\_b=b-\alpha\frac\partial{\partial b}J(w,b)$
+ $w=tmp\_w$
+ $b=tmp\_b$

<font color=red>Incorrect</font>

+ $tmp\_w=w-\alpha\frac\partial{\partial w}J(w,b)$
+ $w=tmp\_w$
+ $tmp\_b=b-\alpha\frac\partial{\partial b}J(w,b)$
+ $b=tmp\_b$


> [!question] Quiz
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/n7JrkYmhTLyya5GJoay8ww_d56fe65b44c84abe8f84e42eaf9223a1_w1l3ivq_4.png?expiry=1694736000000&hmac=-yq1d98Dnt0OLz4Xe5ZQWSLjaycccz2-0Ut_4rZvJyo)
> 
> Gradient descent is an algorithm for finding values of parameters w and b that minimize the cost function J. What does this update statement do? (Assume α is small.)
> 
> $w=w-\alpha\frac{\partial J(w,b)}{\partial w}$
> 
> + ==Checks whether $w$ is equal to $w-\alpha\frac{\partial J(w,b)}{\partial w}$== $\checkmark$
> + Updates parameter $w$ by a small amount
> 
> 正确
> 
> This updates the parameter by a small amount, in order to reduce the cost $J$.

___
***Gradient descent intuition***

Gradient descent algorithm

$$
\begin{align}
\text{repeat until convergence}: \{ \\
w=w-\alpha\frac{\partial}{\partial w}J(w,b) \\
b=b-\alpha\frac{\partial}{\partial b}J(w,b) \\
\}
\end{align}
$$

![](./images/Pasted%20image%2020230913174137.png)

> [!question] Quiz
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/RMPXvf4aS1-D173-GptfPA_43480abfcdf44ba98b75ddb917a1aca1_w1l3ivq_5.png?expiry=1694736000000&hmac=Jxu6XsNMCMZO7CY21Y-bdYdMSziK8N3hpLGv7eq7QxE)
> 
> Gradient descent is an algorithm for finding values of parameters w and b that minimize the cost function J.
> 
> $$
> \begin{align}
> \text{repeat until convergence}: \{ \\
> w=w-\alpha\frac{\partial}{\partial w}J(w,b) \\
> b=b-\alpha\frac{\partial}{\partial b}J(w,b) \\
> \}
> \end{align}
> $$
> 
> Assume the learning rate $α$ is a small positive number. When $\frac{\partial J(w,b)}{\partial w}$​ is a positive number (greater than zero) -- as in the example in the upper part of the slide shown above -- what happens to $w$ after one update step?
> 
> + w stays the same
> + It is not possible to tell if $w$ will increase or decrease.
> + $w$ increases
> + ==$w$ decreases.== $\checkmark$
> 
> 正确
> 
> The learning rate $α$ is always a positive number, so if you take $W$ minus a positive number, you end up with a new value for $W$ that is smaller. 
> 

___
***Learning rate***

![](./images/Pasted%20image%2020230913192947.png)

local minimum

![](./images/Pasted%20image%2020230913193454.png)

___
***Gradient descent for linear regression***

+ Linear regression model

$$f_{w,b}(x)=wx+b$$

+ Cost function

$$J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}-y^{(i)} \right)^2$$

+ Gradient descent algorithm

$$\begin{align}
\text{repeat until convergence } \{ \\
w=w-\alpha \frac{\partial}{\partial w} J(w,b) \\
b=b-\alpha \frac{\partial}{\partial b} J(w,b) \\
\}
\end{align}$$

> $$
> \frac{\partial}{\partial w} J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)}
> $$
> 
> $$
> \frac{\partial}{\partial b} J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(f_{w,b}\big(x^{(i)}\big)-y^{(i)}\big)
> $$

$$\Downarrow$$

$$
\begin{align}
\text{repeat until convergence }\{\\
w=w-\alpha\frac{1}{m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)}\\
b=b-\alpha\frac{1}{m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)})-y^{(i)})\\
\}
\end{align}
$$

![](./images/Pasted%20image%2020230913195711.png)

___
***Running gradient descent***

"Batch" gradient descent

"**Batch**": Each step of gradient descent uses all the training examples 

___
***Optional lab: Gradient descent***

[Gradient descent](C1W1/C1_W1_Lab04_Gradient_Descent_Soln.ipynb)

## Practice quiz: Train the model with gradient descent

1. Gradient descent is an algorithm for finding values of parameters w and b that minimize the cost function J.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/9d6af8aa-0910-478f-b535-192f5c901014image3.png?expiry=1698451200000&hmac=Eg48V4-tYqXcLeBnh8Y5eKUJ4v_1rdoQJKaB2SMBB78)

When $\frac{\partial J(w,b)}{\partial w}$ is a negative number (less than zero), what happens to $w$ after one update step?

+ A. $w$ decreases
+ B. $w$ increases.
+ C. $w$ stays the same
+ D. It is not possible to tell if $w$ will increase or decrease.

My Answer: D

> [!failure] Incorrect
> 
> Whether $w$ increases or decreases depends upon whether the derivative is positive or negative.

> Answer: $w$ increases

> [!check] Correct
> 
> The learning rate is always a positive number, so if you take W minus a negative number, you end up with a new value for W that is larger (more positive).


2. For linear regression, what is the update step for parameter b?

+ A. $$b=b-\alpha\frac{1}{m}\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})$$
+ B. $$b=b-\alpha\frac1m\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)}$$

My Answer: A

> [!check] Correct
> 
> The update step is $b=b-\alpha\frac{\partial J(w,b)}{\partial w}$ where $\frac{\partial J(w,b)}{\partial b}$​ can be computed with this expression: $\sum\limits_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})$

# Week2 Regression with multiple input variables

Learning Objectives

- Use vectorization to implement multiple linear regression
- Use feature scaling, feature engineering, and polynomial regression to improve model training
- Implement linear regression in code

## Multiple linear regression

***Multiple features***

Multiple features(variables)

| Size in feet<sup>2</sup> | Number of bedrooms | Number of floors | Age of home in years | Price in $1000's |
| ------------------------ | ------------------ | ---------------- | -------------------- | ---------------- |
| 2104                     | 5                  | 1                | 45                   | 460              |
| 1416                     | 3                  | 2                | 40                   | 232              |
| 1534                     | 3                  | 2                | 30                   | 315              |
| 852                      | 2                  | 1                | 36                   | 178              |
| ...                      | ...                | ...              | ...                  | ...              |

+ $x_j$ = $j^{th}$ feature
+ $n$ = number of features
+ $\vec{x}^{(i)}$ = features of $i^{th}$ training example
> $\vec{x}^{2}$=[1416 3 2 40]
+ $x_{j}^{(i)}$ = value of feature $j$ in $i^{th}$ training example
> $x_{3}^{(2)}$ = 2

Model:

Previously: $$f_{w,b}(x)=wx+b$$

$f_{w,b}(x)=w_{1}x_{1}+w_{2}x_{2}+\cdots+w_{n}x_{n}+b$

> $\vec{w}=[w_1\ w_2\ w_3\ ...\ w_n]$
> 
> $b$ is a number
> 
> $\vec{x}=[x_1\ x_2\ x_3\ ...\ x_n]$

$$f_{\overrightarrow{W},b}(\vec{X})=\vec{W}\cdot\vec{X}+b=w_{1}x_{1}+w_{2}x_{2}+w_{3}x_{3}+\cdots+w_{n}x_{n}+b$$

> [!question] Quiz
> 
> In the training set below, what is $x_1^{(4​)}$? Please type in the number below (this is an integer such as 123, no decimal points).
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/d-rtZ04RRgSq7WdOEYYEEg_33981fdca69e48538e967e4f3d7449a1_w2l1ivq_1.png?expiry=1694736000000&hmac=Snz2Zz9uHXBFVhBqHLiZBvEKlq4QJHx25YpgtmqZRII)
> 
> 852 $\checkmark$
> 
> $x_1^{(4​)}$ is the first feature (first column in the table) of the fourth training example (fourth row in the table).


___
***Vectorization***

Parameters and features
+ $\vec{w} = [w_1\ w_2\ w_3]$
+ $b$ is a number
+ $\vec{x} = [w_1\ w_2\ w_3]$

linear algebra: count from 1

```python
w = np.array([1.0,2.5,-3.3])
b = 4
x = np.array([10,20,30])
```

code: count from 0

Without Vectorization
```python
f = 0
for j in range(0,n):
	f = f + w[j] * x[j]
f = f + b
```

Vectorization
```python
f = np.dot(w,x)+b
```

![](./images/Pasted%20image%2020230913212323.png)

___
***Gradient descent for multiple linear regression***

<font  face="Times New Roman" color=green size=5><u>2023-09-14 18:46 $\downarrow$</u></font>

![](./images/Pasted%20image%2020230914184916.png)

![](./images/Pasted%20image%2020230914185146.png)

An alternative to gradient descent

**Normal equation**
+ Only for linear regression
+ Solve for $w,b$ without iterations

**Disadvantages**
+ Doesn't generalize to other learning algorithms.
+ Slow when number of features is large (> 10,000)

**What you need to know**
+ Normal equation method may be used in machine learning libraries that implement linear regression.
+ Gradient descent is the **recommended method** for finding parameters $w,b$

___
Optional Lab: multiple linear regression

[multiple linear regression](C1W2/C1_W2_Lab02_Multiple_Variable_Soln.ipynb)

## Practice quiz: Multiple linear regression

1. In the training set below, what is $x_4^{(3)}$? Please type in the number below (this is aninteger such as 123, no decimal points).
![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/aa52579a-b91d-4e22-834d-0cea942d1351image2.png?expiry=1694822400000&hmac=yqb5qFXqCQVkFBZ90B7i4eKBXMjr8xE4ICNnzhtXgTM)

My answer: 30

> [!check] Correct
> 
> Yes! $x_{4}^{(3)}$​ is the 4th feature (4th column in the table) of the 3rd training example (3rd row in the table).

2. Which of the following are potential benefits of vectorization? Please choose the bestoption.
+ A It makes your code run faster
+ B It can make your code shorter
+ C It allows your code to run more easily on parallel compute hardware
+ D All ofthe above 

My answer: D $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Correct! All of these are benefits of vectorization!

3. True/False? To make gradient descent converge about twice as fast, a technique that almost always works is to double the learning rate alpha.

My answer: False $\checkmark[ChatGPT]$

> False. (From ChatGPT)
> 
> Doubling the learning rate alpha in gradient descent does not necessarily make the algorithm converge twice as fast. In fact, increasing the learning rate too much can have adverse effects on the convergence of gradient descent.
> 
> The learning rate is a hyperparameter that determines the size of the steps taken during each iteration of gradient descent. If you increase the learning rate too much, it can lead to overshooting the minimum of the cost function, causing the algorithm to diverge rather than converge. On the other hand, if the learning rate is too small, it may lead to slow convergence.
> 
> Choosing the right learning rate is often a process of experimentation and fine-tuning. There are more sophisticated techniques, such as learning rate schedules and adaptive learning rate methods, that can help gradient descent converge faster without the need to blindly double the learning rate.

> [!check] Correct
> 
> Doubling the learning rate may result in a learning rate that is too large, and cause gradient descent to fail to find the optimal values for the parameters $w$ and $b$.

## Gradient descent in practice

***Feature scaling***

enable gradient descent to run much faster

![](./images/Pasted%20image%2020230914202449.png)

![](./images/Pasted%20image%2020230914202946.png)

Mean normalization

![](./images/Pasted%20image%2020230914203430.png)

Z-score normalization

standard deviation $\sigma$

![](./images/Pasted%20image%2020230914203916.png)

![](./images/Pasted%20image%2020230914204148.png)


> [!question] Quiz
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Xh5QR3GGSrmeUEdxhuq5Jg_6ad4808a655f46a4a7c21e866dfe2da1_Screen-Shot-2022-06-15-at-11.41.33-AM.png?expiry=1694822400000&hmac=DiDop-b6T1bhYo-Ke1leGu0EdrgkiJm1ofu9uZ9U5rI)
> 
> Which of the following is a valid step used during feature scaling? 
> 
> + <font color=red>Multiply</font> each value by the maximum value for that feature $×$
> + ==<font color=red>Divide</font> each value by the maximum value for that feature== $\checkmark$
> 
> 正确
> 
> By dividing all values by the maximum, the new maximum range of the rescaled features is now 1 (and all other rescaled values are less than 1).

___
***Checking gradient descent for convergence***

$\epsilon=10^{-3}$

![](./images/Pasted%20image%2020230914205524.png)

___
***Choosing the learning rate***

![](./images/Pasted%20image%2020230914212559.png)
![](./images/Pasted%20image%2020230914212753.png)

> [!question] Quiz
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/HJXHo_ccQ3qVx6P3HGN6Ww_8b5aa55079da4da29504bd23ca6cf5a1_Screen-Shot-2022-06-15-at-11.46.52-AM.png?expiry=1694822400000&hmac=1xr7XsXl3n4z2t-xiDXu1tyj042hORbaDe5yrEwRB68)
> You run gradient descent for 15 iterations with $α=0.3$ and compute $J(w)$ after each iteration. You find that the value of $J(w)$ increases over time.  How do you think you should adjust the learning rate $α$?
> + Keep running it for additional iterations
> + Try a larger value of $\alpha$ (say $α=1.0$).
> + Try running it for only 10 iterations so $J(w)$ doesn’t increase as much.
> + ==Try a smaller value of $α$ (say $α=0.1$).== $\checkmark$
> 
> 正确
> 
> Since the cost function is increasing, we know that gradient descent is diverging, so we need a lower learning rate.

___
***Optional Lab: Feature scaling and learning rate***


[Feature scaling and learning rate](C1W2/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb)

___
***Feature engineering***

<font  face="Times New Roman" color=green size=5><u>2023-09-15 19:55 $\downarrow$</u></font>

Feature engineering: Using <font color="#ffc000">intuition</font> to design <font color="#ffc000">new features</font>, by transforming or combining original features


> [!question] Quiz
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/DygYeH6_SnKoGHh-v8pydQ_50a8d6e730a349a2af49d92f98fd80a1_Screen-Shot-2022-06-15-at-11.56.00-AM.png?expiry=1694908800000&hmac=YoWi5CPz-5UacRWevt-QVOhPkm1dAoI9tLZjht9XVlA)
> 
> If you have measurements for the dimensions of a swimming pool (length, width, height), which of the following two would be a more useful engineered feature?
> 
> + ==length×width×height== $\checkmark$
> + length+width+height
> 
> 正确
> 
> The volume of the swimming pool could be a useful feature to use.  This is the more useful engineered feature of the two.

___
***Polynomial regression***

*多项式回归*

![](./images/Pasted%20image%2020230915201030.png)

___
***Optionallab: Feature engineering and Polynomial regression***

[Feature engineering and Polynomial regression](C1W2/C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb)

___
***Optionallab: Linear regression with scikit-learn***

[Linear regression with scikit-learn](C1W2/C1_W2_Lab05_Sklearn_GD_Soln.ipynb)

[scikit-learn](https://scikit-learn.org/stable/)

## Practice quiz: Gradient descent in practice

1. ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/91035b73-148b-4f43-be79-695519301becimage3.png?expiry=1694908800000&hmac=McXP9FEMlnzLtMNNPc0jO37ednV08JTa6ajpHX4qouo)

Which of the following is a valid step used during feature scaling?
+ Add the mean (average) from each value and and then divide by the (max - min).
+ Subtract the mean (average) from each value and then divide by the (max - min). ==My answer== $\checkmark[ChatGPT]$

> [!check] Correct
> 
> This is called mean normalization.

2. Suppose a friend ran gradient descent three separate times with three choices of the learning rate a and plotted the learning curves for each (cost J for each iteration).
![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/91035b73-148b-4f43-be79-695519301becimage4.png?expiry=1694908800000&hmac=92wVq1ACH4SJ7d_ZrscWsEJPMcUxJySFAYxLdKkXv6A)

For which case,A or B, was the learning rate a likely too large?
+ Neither Case A nor B
+ Both Cases A and B
+ case A only
+ case B only ==My answer== $\checkmark[ChatGPT]$

> [!check] Correct
> 
> The cost is increasing as training continues, which likely indicates that the learning rate alpha is too large.

3. Of the circumstances below, for which one is feature scaling particularly helpful?
+ Feature scaling is helpful when one feature is much larger (or smaller) than another feature. ==My answer== $\checkmark[ChatGPT]$
+ Feature scaling is helpful when all the features in the original data (before scaling is applied) range from 0 to 1.

> [!check] Correct
> 
> For example, the “house size” in square feet may be as high as 2,000, which is much larger than the feature “number of bedrooms” having a value between 1 and 5 for most houses in the modern era.


1. You are helping a grocery store predict its revenue, and have data on its items sold perweek, and price per item. What could be a useful engineered feature?
+ For each product, calculate the number ofitems sold **divided** by the price per item.
+ For each product, calculate the number of items sold **times** price per item. ==My answer== $\checkmark[ChatGPT]$

> [!check] Correct
> This feature can be interpreted as the revenue generated for each product.

2. True/False? With polynomial regression, the predicted values f_w,b(x) does notnecessarily have to be a straight line (or linear) function of the input feature x.
+ True  ==My answer== $\checkmark[ChatGPT]$
+ False

> [!check] Correct
> A polynomial function can be non-linear.  This can potentially help the model to fit the training data better.

> [!NOTE] ChatGPT
> True. With polynomial regression, the predicted values f(w, b, x) do not necessarily have to be a straight line or linear function of the input feature x. Polynomial regression allows for modeling relationships between the dependent variable (the predicted values) and the independent variable (the input feature) using polynomial functions of different degrees.
> 
> In polynomial regression, you can have equations like:
> 
> $f(w, b, x) = b + w_1 * x + w_2 * x^2 + w_3 * x^3 + ...$
> 
> Here, the polynomial terms ($x^2$, $x^3$, etc.) introduce curves and non-linearities into the relationship between the input feature x and the predicted values. The choice of polynomial degree determines how flexible the model is in capturing non-linear patterns in the data. Higher-degree polynomials can fit more complex, non-linear relationships between the variables. So, polynomial regression allows for more flexibility in modeling than simple linear regression, where the relationship is linear.

## Week 2 practice lab: Linear regression

[Linear regression](W2_2/C1_W2_Linear_Regression.ipynb)

# Week3 Classification

Learning Objectives

- Use logistic regression for binary classification
- Implement logistic regression for binary classification
- Address overfitting using regularization, to improve model performance

## Classification with logistic regression

***Motivations***

Classification

| Question                       | Answer"y" |
| ------------------------------ | --------- |
| Is this email spam?            | no   yes  |
| Is the transaction fraudulent? | no   yes  |
| Is the tumor malignant?        | no   yes  |

y can only be one of <font color="#00b0f0">two</font> values

"<font color="#00b0f0">binary</font> classification"

class=category

+ "negative class": false/0
+ "positive class": true/1

![](./images/Pasted%20image%2020230915210347.png)

> [!question] Quiz
> 
> Which of the following is an example of a classification task?
> 
> + ==Decide if an animal is a cat or not a cat.== $\checkmark$
> + Estimate the weight of a cat based on its height.
> 
> Correct
> 
> Correct: This is an example of _binary classification_ where there are two possible classes (True/False or Yes/No or 1/0).

___
***Optional lab: Classification***

%% + http://t.csdn.cn/NVWAY
+ VSCode-MachineLearning.ipynb %%

___
***Logistic regression***

<font  face="Times New Roman" color=green size=5><u>2023-09-18 14:54 $\downarrow$</u></font>

![](./images/Pasted%20image%2020230918143717.png)
![](./images/Pasted%20image%2020230918143955.png)

Interpretation of logistic regression output
$$
f_{\vec{w},b}(\vec{x})=\frac{1}{1+e^{-(\vec{w}\cdot\vec{x}+b)}}
$$

"probability" that class is 1

Example:
+ $x$ is "tumor size"
+ $y$ is 0 (not malignant)
+ or 1 (malignant)

+ $f_{\vec{\mathrm{w}},b}(\vec{\mathrm{x}})=0.7$
+ 70% chance that $y$ is 1

$f_{\vec{\mathrm{w}},b}(\vec{\mathrm{x}})=P(\mathbf{y}=1|\vec{\mathrm{x}};\vec{\mathrm{w}},b)$
+ Probability that y is 1
+ given input $\vec{x}$ , parameters $\vec{w},b$

P(y=0)+P(y=1)=1

> [!question] Quiz
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/yJ-DMkC1T8efgzJAtb_HFA_3a3b7a7266ce488db279c09724a8fdf1_Screen-Shot-2022-06-11-at-12.40.39-PM.png?expiry=1695168000000&hmac=oRpYZFYb8RH5Pm_Li3e3b4Jxvlj9DojQRoscFXTNBxk)
> Recall the sigmoid function is $g(z)=\frac1{1+e^{-z}}$
> 
> If $z$ is a large negative number then:
> 
> + $g(z)$ is near negative  one (-1)
> + ==$g(z)$ is near zero== $\checkmark$
> 
> Correct
> Say $z$=-100. $e^{-z}$ is then $e^{100}$, a really big positive number. So, $g(z)=\frac1{1+\text{a big positive number}}$ or about $0$

___
***Optional lab: Sigmoid function and logistic regression***

[Sigmoid function and logistic regression](C1W3/C1_W3_Lab02_Sigmoid_function_Soln.ipynb)

___
***Decision boundary***

![](./images/Pasted%20image%2020230918184525.png)
![](./images/Pasted%20image%2020230918185050.png)
![](./images/Pasted%20image%2020230918185302.png)
![](./images/Pasted%20image%2020230918185456.png)

> [!question] Quiz
> 
> Let’s say you are creating a tumor detection algorithm. Your algorithm will be used to flag potential tumors for future inspection by a specialist. What value should you use for a threshold?
> 
> + High, say a threshold of 0.9? $\times$
> + ==Low, say a threshold of 0.2?== $\checkmark$
> 
> Correct
> 
> **Correct**: You would not want to miss a potential tumor, so you will want a low threshold. A specialist will review the output of the algorithm which reduces the possibility of a ‘false positive’. The key point of this question is to note that the threshold value does not need to be 0.5.

___
***Optional lab: Decision boundary***

[Decision boundary](C1W3/C1_W3_Lab03_Decision_Boundary_Soln.ipynb)


## Practice quiz: Classification with logistic regression

1. Which is an example of a classification task?
+ Based on the size of each tumor, determine if each tumor is malignant (cancerous)or not.
+ Based on a patient's blood pressure, determine how much blood pressuremedication (a dosage measured in milligrams) the patient should be prescribed.
+ Based on a patient's age and blood pressure, determine how much blood pressuremedication (measured in milligrams) the patient should be prescribed.

My Answer: A $\checkmark[ChatGPT]$

> [!check] Correct
> 
> This task predicts one of two classes, malignant or not malignant.

2. Recall the sigmoid function is $g(z)=\frac{1}{1+e^{-z}}$
![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/d60aeff4-f74f-459c-b70d-9c06c64458d7image3.png?expiry=1695168000000&hmac=O_GDG7DtW4K0gbCEphHfpxX5MJVsRHsVgVKxjQJTd-4)
lf z is a large positive number, then:
+ $g(z)$ is near negative one (-1)
+ $g(z)$ is near one (1)
+ $g(z)$ will be near 0.5
+ $g(z)$ will be near zero (0)

My Answer: B $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Say $z=+100$. So $e^{-z}$ is then $e^{-100}$,  a really small positive number. So $g(z)=\frac{1}{1+\mathrm{a~small~positive~number}}$ which is close to $1$

3. A cat photo classification model predicts lif it's a cat,and 0 if it's not a cat For a particular photograph, the logistic regression model outputs $g(z)$ (a number between 0 and 1). Which of these would be a reasonable criteria to decide whether to predict if it's a cat?
![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/d60aeff4-f74f-459c-b70d-9c06c64458d7image2.png?expiry=1695168000000&hmac=qV1VEyB8OHh8jPp7auOHNTCN5GhCZdWFSbiEzOMT0iE)

+ Predict it is a cat if g(z) < 0.5
+ Predict it is a cat if g(z) = 0.5
+ Predict it is a cat if g(z) >= 0.5
+ Predict it is a cat if g(z) < 0.7

My Answer: C $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Think of g(z) as the probability that the photo is of a cat. When this number is at or above the threshold of 0.5, predict that it is a cat.

4. True/False? No matter what features you use (including if you use polynomial features), the decision boundary learned by logistic regression will be a linear decision boundary.

My Answer: False $\checkmark[ChatGPT]$

> [!check] Correct
> 
> The decision boundary can also be non-linear, as described in the lectures.
## Cost function for logistic regression

***Cost function for logistic regression***

![](./images/Pasted%20image%2020230918205529.png)

**Squared error cost**
$$
J(\vec{w},b)=\dfrac{1}{m}\sum_{i=1}^{m}\dfrac{1}{2}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^{2}
$$

![](./images/Pasted%20image%2020230918210212.png)

**Logistic loss function**

$$L\big(f_{\vec{w},b}\big(\vec{x}^{(i)}\big),y^{(i)}\big)=\begin{cases}\quad-\log\big(f_{\vec{w},b}\big(\vec{x}^{(i)}\big)\big)&\text{if }y^{(i)}=1\\-\log\big(1-f_{\vec{w},b}\big(\vec{x}^{(i)}\big)\big)&\text{if }y^{(i)}=0\end{cases}$$

![](./images/Pasted%20image%2020230918210848.png)
![](./images/Pasted%20image%2020230918211051.png)

**Cost**

![](./images/Pasted%20image%2020230918211209.png)


> [!question] Quiz
> 
> Why is the squared error cost not used in logistic regression?
> 
> + ==The non-linear nature of the model results in a “wiggly”, non-convex cost function with many potential local minima.== $\checkmark$
> + The mean squared error is used for logistic regression.
> 
> Correct
> 
> If using the mean squared error for logistic regression, the cost function is "non-convex", so it's more difficult for gradient descent to find an optimal value for the parameters w and b.

___
***Optional lab: Logistic loss***

[Logistic loss](C1W3/C1_W3_Lab04_LogisticLoss_Soln.ipynb)

___
***Simplified Cost Function for Logistic Regression***

![](./images/Pasted%20image%2020230918212938.png)
![](./images/Pasted%20image%2020230918213117.png)

___
***Optional lab: Cost function for logistic regression***

[Cost function for logistic regression](C1W3/C1_W3_Lab05_Cost_Function_Soln.ipynb)

## Practice quiz: Cost function for logistic regression

<font  face="Times New Roman" color=green size=5><u>2023-09-19 16:20 $\downarrow$</u></font>

1. in this lecture series,"cost" and "loss" have distinct meanings. Which one applies to a single training example?
+ Loss
+ Cost
+ Both Loss and Cost
+ Neither Loss nor Cost
![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/30800c78-5d39-4239-a323-c65b911d1bfcimage2.png?expiry=1695254400000&hmac=qu4tDzbWcsOdmXZutmcQ393XLIsRFLOjXs4X7LIdHHo)

My Answer: A $\checkmark[ChatGPT]$

> The term "Loss" typically applies to a single training example. It represents a measure of the error or the difference between the predicted output and the actual target output for that specific example. The loss is used to assess how well the model is performing on individual data points.
> 
> On the other hand, "Cost" or "Total Cost" usually refers to the cumulative loss over a dataset or a batch of examples. It sums up the losses for multiple training examples and is often used in the context of optimizing a machine learning model through techniques like gradient descent.
> 
> So, for a single training example, the answer is "Loss."

> [!check] Correct
> 
> In these lectures, loss is calculated on a single training example. It is worth noting that this definition is not universal. Other lecture series may have a different definition.

2. For the simplified loss function, if the label $y^{(i)} = 0$, then what does this expression simplify to?
![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/30800c78-5d39-4239-a323-c65b911d1bfcimage3.png?expiry=1695254400000&hmac=oJ3uPZVuUErPSJoVb8rJ4Fj3FD5jRyH0Bnaw_UiR_qo)
+ $-\log(1-f_{\vec{w},b}(X^{(i)}))$
+ $\log(f_{\vec{w},b}(X^{(i)}))$
+ $-\log(1-f_{\vec{w},b}(X^{(i)}))-\log(1-f_{\vec{w},b}(X^{(i)}))$
+ $\log(1-f_{\vec{w},b}(X^{(i)}))+\log(1-f_{\vec{w},b}(X^{(i)}))$

My Answer: A $\checkmark[ChatGPT]$

> [!check] Correct
> 
> When $y^{(I)}=0$, the first term reduces to zero.


## Gradient descent for logistic regression

***Gradient Descent Implementation***

**Training logistic regression**

Find $\vec{w},b$

Fiven new $\vec{x}$, output $f_{\vec{w},b}(\vec{X})=\frac{1}{1+e^{-(\vec{w}\cdot\vec{x}+b)}}$

$P(y=1|\vec{x};\vec{w},b)$

**Gradient Descent**

![](./images/Pasted%20image%2020230919161149.png)
![](./images/Pasted%20image%2020230919161418.png)

___
***Optional lab: Gradient descent for logistic regression***

[Gradient descent for logistic regression](C1W3/C1_W3_Lab06_Gradient_Descent_Soln.ipynb)

___
***Optional lab: Logistic regression with scikit-learn***

[Logistic regression with scikit-learn](C1W3/C1_W3_Lab07_Scikit_Learn_Soln.ipynb)

## Practice quiz: Gradient descent for logistic regression

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Tv99olQhR7K_faJUIbeyDQ_61ca3d07977d46b4aec7eae43554bca1_Screen-Shot-2022-06-29-at-8.41.24-PM.png?expiry=1695254400000&hmac=ZPfSjAE6nAO_14MgiXneUkhBGuiR8YiyhRONkoKz43w)
Which of the following two statements is a more accurate statement about gradient descent for logistic regression?
+ The update steps look like the update steps for linear regression, but the definition of $f_{\vec{w},b}(X^{(i)})$ is different.
+ The update steps are identical to the update steps for linear regression.

My Answer: A $\checkmark[ChatGPT]$

> [!check] Correct
> 
> For logistic regression, $f_{\vec{w},b}(X^{(i)})$ is the sigmoid function instead of a straight line.

## The problem of overfitting

***The problem of overfitting***

Regression example

![](./images/Pasted%20image%2020230919165909.png)

Classification

![](./images/Pasted%20image%2020230919170220.png)

> [!question] Quiz
> 
> Our goal when creating a model is to be able to use the model to predict outcomes correctly for **new examples**. A model which does this is said to **generalize** well. 
> 
> When a model fits the training data well but does not work well with new examples that are not in the training set, this is an example of:
> 
> + ==Overfitting (high variance)== $\checkmark$
> + None of the above
> + A model that generalizes well (neither high variance nor high bias)
> + Underfitting (high bias)
> 
> 正确
> 
> This is when the model does not generalize well to new examples.

___
***Addressing overfitting***

1. Collect more training examples

![](./images/Pasted%20image%2020230919170948.png)

2. Select features to include/exclude

![](./images/Pasted%20image%2020230919171403.png)

3. Regularization

> + **shrink** the values of parameters without necessarily demanding that the parameter is set to **exactly 0**.
> + keep **all features** but prevents the features from having a **overly large effect** (overfitting)

![](./images/Pasted%20image%2020230919171932.png)


> [!Summary] Addressing overfitting
> options
> 1. Collect more data
> 2. Select features
> 	- Feature selection
> 3. Reduce size of parameters
> 	- "Regularization"

> [!question] Quiz
> 
> Applying regularization, increasing the number of training examples, or selecting a subset of the most relevant features are methods for…
> 
> + ==Addressing overfitting (high variance)== $\checkmark$
> + Addressing underfitting (high bias)
> 
> 正确
> 
> These methods can help the model generalize better to new examples that are not in the training set.

___
***Optional lab: Overfitting***

[Overfitting](C1W3/C1_W3_Lab08_Overfitting_Soln.ipynb)

___
***Cost function with regularization***

![](./images/Pasted%20image%2020230919173238.png)

penalize: 惩罚

$\lambda$ : a regularization parameter

$+\frac{\lambda}{2m}b^2$: makes very little difference in practice

![](./images/Pasted%20image%2020230919174004.png)

+ $\lambda$: too small$\rightarrow$overfit
+ $\lambda$: too large$\rightarrow$underfit

![](./images/Pasted%20image%2020230919174856.png)

> [!question] Quiz
> 
> For a model that includes the regularization parameter $λ$ (lambda), increasing $λ$ will tend to…
> 
> + Increases the size of the parameters $w_1,w_2,...,w_n$
> + ==Decrease the size of parameters $w_1,w_2,...,w_n$== $\checkmark$
> + Increase the size of parameter $b$.
> + Decrease the size of the parameter $b$.
> 
> 正确
> 
> Increasing the regularization parameter ***lambda*** reduces overfitting by reducing the size of the parameters.  For some parameters that are near zero, this reduces the effect of the associated features.

___
***Regularized linear regression***

![](./images/Pasted%20image%2020230919195056.png)
![](./images/Pasted%20image%2020230919200014.png)

How to get the derivative term (optional)

![](./images/Pasted%20image%2020230919204629.png)

> [!question] Quiz
> 
> Recall the gradient descent algorithm utilizes the gradient calculation:
> 
> $$\begin{aligned}
> \text{repeat until convergence: \{} \\
> w_{j}=w_{j}-\alpha\Bigg[\frac{1}{m}\sum_{i=1}^{m}(f_{\mathbf{w},b}(\mathbf{x}^{(i)})-y^{(i)})x_{j}^{(i)}+\frac{\lambda}{m}w_{j}\Bigg]& \mathrm{for~j=1..n}  \\
> b=b-\alpha\frac{1}{m}\sum_{i=0}^{m-1}(f_{\mathbf{w},b}(\mathbf{x}^{(i)})-y^{(i)}) \\
> \}
> \end{aligned}$$
> 
> Where each iteration performs simultaneous updates on $w_j$​ for all $j$.
> 
> In lecture, this was rearranged to emphasize the impact of regularization:
> 
> $$w_j=w_j-\alpha\left[\frac{1}{m}\sum_{i=1}^m(f_{\mathbf{w},b}(\mathbf{x}^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}w_j\right]\quad\text{for j}=1..\text{n}$$
> 
> is rearranged to be:
> 
> $$w_j=w_j\underbrace{{\left(1-\alpha\frac\lambda m\right)}}_{NewPart}-\underbrace{\alpha\frac1m\sum_{i=1}^m(f_{\mathbf{w},b}(\mathbf{x}^{(i)})-y^{(i)})x_j^{(i)}}_{OriginalPart}\quad\mathrm{for~j=1..n}$$
> 
> Assuming $α$, the learning rate, is a small number like 0.001, $λ$ is 1, and $m=50$, what is the effect of the 'new part' on updating $w_j$​?
> + ==The new part decreases the value of $w_j$​ each iteration by a little bit.== $\checkmark$
> 
> + The new part increases the value of $w_j$ each iteration by a little bit.
> 
> + The new parts impact varies each iteration.
> 
> 正确
> 
> **Correct:** the new term decreases $w_j$​ each iteration.

___
***Regularized logistic regression***

![](./images/Pasted%20image%2020230919214335.png)
![](./images/Pasted%20image%2020230919214454.png)

> [!question] Quiz
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Db_hva8mTr-_4b2vJi6_Pg_4a4c3dd4f61c46b4827391bd6d724cf1_Screen-Shot-2022-06-11-at-7.12.34-PM.png?expiry=1695254400000&hmac=YX2p1BujBaFRFHoUgCCRFUUoA0DRBPBBEmp6I3-FWpk)
> For regularized **logistic** regression, how do the gradient descent update steps compare to the steps for linear regression?
> 
> + ==They look very similar, but the $f(x)$ is not the same.== $\checkmark$
> + They are identical
> 
> 正确
> 
> For logistic regression, $f(x)$ is the sigmoid (logistic) function, whereas for linear regression, $f(x)$ is a linear function.

___
***Optional lab: Regularization***

[Regularization](C1W3/C1_W3_Lab09_Regularization_Soln.ipynb)

## Practice quiz: The problem of overfitting

1. Which of the following can address overfitting?
+ Select a subset of the more relevant features.
+ Remove a random set oftraining examples
+ Collect more training data
+ Apply regularization

My Answer: A, C, D $\checkmark[ChatGPT]$

> [!check] Correct
> 
> + A If the model trains on the more relevant features, and not on the less useful features, it may generalize better to new examples.
> + C If the model trains on more data, it may generalize better to new examples.
> + D Regularization is used to reduce overfitting.

2. You fit logistic regression with polynomial features to a dataset, and your model looks like this

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/335acb2e-4819-4d1f-920c-1b1d682eeb37image2.png?expiry=1695254400000&hmac=2cwBzC3wyKYjv87IEGAX2_QbHv7Nk2hEd2V2F19XRRg)

What would you conclude? (Pick one)
+ The model has high variance (overfit). Thus, adding data is, by itself, unlikely to help much.
+ The model has high bias (underfit). Thus, adding data is likely to help
+ The model has high variance (overfit). Thus, adding data is likely to help
+ The model has high bias (underfit). Thus, adding data is, by itself, unlikely to help much.

My Answer: C $\checkmark[ChatGPT]$

> [!check] Correct
> 
> The model has high variance (it overfits the training data). Adding data (more training examples) can help.

3. Suppose you have a regularized linear regression model, lf you increase the regularization parameter $\lambda$, what do you expect to happen to the parameters $w_1, w_2, ..., w_n$?

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/ujk9ltT9RAC5PZbU_YQALQ_7bc73e8e2f3a48cf800d1a504f55ecf1_Screen-Shot-2022-06-14-at-3.38.02-PM.png?expiry=1695254400000&hmac=W_VhMnTrsqsY55VnZtuY62Ht56ibA7-eomw1YiYMBN0)

+ This will reduce the size of the parameters $w_1, w_2, ..., w_n$
+ This will increase the size of the parameters $w_1, w_2, ..., w_n$

My Answer: A $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Regularization reduces overfitting by reducing the size of the parameters $w_1​,w_2​,...w_n$​.

## Week 3 practice lab: logistic regression

[practice lab: logistic regression](W3_2/C1_W3_Logistic_Regression.ipynb)

## Conversations with Andrew (Optional)

***Andrew Ng and Fei-Fei Li on Human-Centered Al***

<font  face="Times New Roman" color=green size=5><u>2023-09-20 13:56 $\downarrow$</u></font>

+ Feifei Li: Stanford 2009
+ Physics $\rightarrow$ AI
+ STEM kid
> STEM is **an abbreviation for science, technology, engineering, and mathematics**. The term is used to describe both education and careers in those fields. STEM was first introduced in 2001 by the U.S. National Science Foundation.
+ passion for asking big question
+ Roger Penrose *Emperor's New Mind*
+ understanding of natural objects and natural things
+ Feifei Li, During PhD... the most important challenge in AI machine learning is the lack of generalizability
+ the Caltech 101dataset, of 101 object categories, and about 30,000 pictures.
+ Suddenly the availability of data was a new thing.
+ the right **North Star problem** and the data that drives it.
+ better regulation
+ human centered
+ that is not a regulatory policy, it's more an incentive policy to build and rejuvenate ecosystems.

## Acknowledgments

<font  face="Times New Roman" color=green size=5><u>2023-09-20 15:53 $\Uparrow$</u></font>

Check Quiz Answer

<font  face="Times New Roman" color=green size=5><u>2023-10-26 17:50 $\Uparrow$</u></font>
