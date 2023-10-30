---
aliases: 
author: Li Yaozong
date: 2023-10-04
time: 2023-10-04 17:26
cover: https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://coursera-course-photos.s3.amazonaws.com/9d/94b73964644f83be145d935af64acc/Course-Logo--3.png
description: Unsupervised Learning, Recommenders, Reinforcement Learning
source: Coursera
status: Completed
tags:
  - cs
  - MachineLearing
  - notes
url: https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/home/welcome
---

![cover](https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://coursera-course-photos.s3.amazonaws.com/9d/94b73964644f83be145d935af64acc/Course-Logo--3.png)

# Week1 Unsupervised Learning

Learning Objectives

- Implement the k-means clustering algorithm
- Implement the k-means optimization objective
- Initialize the k-means algorithm
- Choose the number of clusters for the k-means algorithm
- Implement an anomaly detection system
- Decide when to use supervised learning vs. anomaly detection
- Implement the centroid update function in k-means
- Implement the function that finds the closest centroids to each point in k-means

## Welcome to the course!

***Welcome***

**Beyond Supervised Learning**

+ Unsupervised Learning
	- Clustering
	- Anomaly detection
+ Recommender Systems
+ Reinforcement Learning

## Clustering

___
***What is clustering?***

**Supervised learning**

![](./images/Pasted%20image%2020231004195756.png)

**Unsupervised learning**

![](./images/Pasted%20image%2020231004195934.png)

**Applications of clustering**

![](./images/Pasted%20image%2020231004200144.png)

___
***K-means intuition***

Step 1: Assign each point to its closest centroid

![](./images/Pasted%20image%2020231004201531.png)

Step 2: Recompute the centroids

![](./images/Pasted%20image%2020231004201702.png)

Step 1

![](./images/Pasted%20image%2020231004201910.png)

Step 2

![](./images/Pasted%20image%2020231004201952.png)

Repeat Step 1, Step 2

![](./images/Pasted%20image%2020231004202148.png)

___
***K-means algorithm***

![](./images/Pasted%20image%2020231004202900.png)

L2 norm: $\parallel x^{(i)}-\mu_k\parallel$

![](./images/Pasted%20image%2020231004203141.png)

![](./images/Pasted%20image%2020231004203326.png)

+ k = k-1 (more common)
+ randomly reinitialize that cluster centroid

**K-means for clusters that are not well separated**

![](./images/Pasted%20image%2020231004203823.png)

___
***Optimization objective***

**K-means optimization objective**

+ $c^{(i)}$ = index of cluster $(1, 2, ..., K)$ to which example $x^{(i)}$ is currently assigned
+ $\mu _k$ = cluster centroid $k$
+ $\mu _{c^{(i)}}$ = cluster centroid of cluster to which example $x^{(i)}$ has been assigned

Cost function (/distortion function)

$$
\begin{aligned}J\big(c^{(1)},...,c^{(m)},\mu_{1},...,\mu_{K}\big)&=\frac{1}{m}\sum_{i=1}^{m}\big\Vert x^{(i)}-\mu_{c^{(i)}}\big\Vert^{2}\end{aligned}
$$

![](./images/Pasted%20image%2020231004211025.png)

> And it turns out that what the K-means algorithm is doing is trying to find assignments of points of clusters centroid as well as find locations of clusters centroid that minimizes the squared distance.

![](./images/Pasted%20image%2020231004211616.png)

**Cost function for K-means**

$$
J\big(c^{(1)},\ldots,c^{(m)},\mu_{1},\ldots,\mu_{K}\big)=\frac{1}{m}\sum_{i=1}^{m}\big\Vert x^{(i)}-\mu_{c^{(i)}}\big\Vert^{2}
$$

![](./images/Pasted%20image%2020231004212133.png)

**Moving the centroid**

![](./images/Pasted%20image%2020231004212458.png)

___
***Initializing K-means***

**K-means algorithm**

![](./images/Pasted%20image%2020231004213917.png)

**Random initialization**

![](./images/Pasted%20image%2020231004214319.png)
![](./images/Pasted%20image%2020231004215548.png)
![](./images/Pasted%20image%2020231004215805.png)

___
***Choosing the number of clusters***

**What is the right value of K?**

![](./images/Pasted%20image%2020231004220056.png)

**Choosing the value of K**

Elbow method *(Andrew Ng does not use)*

![](./images/Pasted%20image%2020231004220430.png)

Often, you want to get clusters for some later (downstream) purpose.
Evaluate K-means based on how well it performs on that later purpose.

![](./images/Pasted%20image%2020231004220740.png)

## Practice Quiz: Clustering

<font  face="Times New Roman" color=green size=5><u>2023-10-05 $\downarrow$ </u></font>

1. Which of these best describes unsupervised learning?

+ A. A form of machine learning that finds patterns using unlabeled data (x).
+ B. A form of machine learning that finds patterns without using a cost function.
+ C. A form of machine learning that finds patterns using labeled data (x, y)
+ D. A form of machine learning that finds patterns in data using only labels (y) butwithout anyinputs (x).

My Answer: A

> [!check] Correct
> 
> Unsupervised learning uses unlabeled data. The training examples do not have targets or labels "y". Recall the T-shirt example. The data was height and weight but no target size.

2. Which of these statements are true about K-means? Check all that apply.

+ A. The number of cluster assignment variables $c^{(i)}$ is equal to the number of trainingexamples.
+ B. The number of cluster centroids $\mu_k$ is equal to the number of examples.
+ C. lf each example x is a vector of 5 numbers, then each cluster centroid $\mu_k$ is also going to be a vector of 5 numbers.
+ D. If you are running K-means with $K = 3$ clusters, then each $c^{(i)}$ should be 1, 2, or 3

My Answer: CD

> [!warning] Warning
> 
> You didn’t select all the correct answers

> [!check] Correct Answer
> 
> + The number of cluster assignment variables $c^{(i)}$ is equal to the number of trainingexamples.
> 
> > $c^{(i)}$ describes which centroid example$(i)$ is assigned to.
> 
> + lf each example x is a vector of 5 numbers, then each cluster centroid $\mu_k$ is also going to be a vector of 5 numbers.
> 
> > The dimension of $μ_k$​ matches the dimension of the examples.
> 
> + If you are running K-means with $K = 3$ clusters, then each $c^{(i)}$ should be 1, 2, or 3
> 
> > $c^{(i)}$ describes which centroid example($i$) is assigned to. If $K=3$, then $c^{(i)}$ would be one of 1,2 or 3 assuming counting starts at 1.

3. You run K-means 100 times with different initializations. How should you pick from the 100 resulting solutions?

+ A. Pick the last one (i.e, the 100th random initialization) because K-means always improves over time
+ B. Pick randomly -- that was the point of random initialization.
+ C. Pick the one with the lowest cost $J$
+ D. Average all 100 solutions together.

My Answer: C

> [!check] Correct
> 
> K-means can arrive at different solutions depending on initialization. After running repeated trials, choose the solution with the lowest cost.

4. You run K-means and compute the value of the cost function $J(c^{(1)}.., ... , c^{(m)}, \mu_1, ... , \mu_K)$ after each iteration. Which of these statements should be true?

+ A. Because K-means tries to maximize cost, the cost is always greater than or equal to the cost in the previous iteration.
+ B. There is no cost function for the K-means algorithm.
+ C. The cost will either decrease or stay the same after each iteration.
+ D. The cost can be greater or smaller than the cost in the previous iteration, but it decreases in the long run.

My Answer: C

> [!check] Correct
> 
> The cost never increases. K-means always converges.

5. In K-means, the elbow method is a method to

+ A. Choose the maximum number of examples for each cluster
+ B. Choose the best random initialization
+ C. Choose the best number of samples in the dataset
+ D. Choose the number of clusters $K$

My Answer: D

> [!check] Correct
> 
> The elbow method plots a graph between the number of clusters K and the cost function. The ‘bend’ in the cost curve can suggest a natural value for K. Note that this feature may not exist or be significant in some data sets.

## Practice Lab 1

[Programming Assignment: k-means](C3_W1_lab1/C3_W1_KMeans_Assignment.ipynb)


## Anomaly detection

___
***Finding unusual events***

**Anomaly detection example**

![](./images/Pasted%20image%2020231005190318.png)

**Density estimation**

![](./images/Pasted%20image%2020231005190702.png)

**Anomaly detection example**

![](./images/Pasted%20image%2020231005191843.png)

___
***Gaussian (normal) distribution***

> bell-shaped curve

![](./images/Pasted%20image%2020231005192442.png)

**Gaussian distribution example**

![](./images/Pasted%20image%2020231005192800.png)

**Parameter estimation**

![](./images/Pasted%20image%2020231005193214.png)

___
***Anomaly detection algorithm***

**Density estimation**

![](./images/Pasted%20image%2020231005194043.png)

> what that means is we're going to estimate, the mean of the feature $x_1$ and also the variance of feature $x_1$ and that will be $\mu_1$ and $\sigma_1$.

**Anomaly detection algorithm**

![](./images/Pasted%20image%2020231005194453.png)

**Anomaly detection algorithm example**

![](./images/Pasted%20image%2020231005194854.png)

___
***Developing and evaluating an anomaly detection system***

**The importance of real-number evaluation**

When developing a learning algorithm (choosing features, etc.), making decisions is much easier if we have a way of evaluating our learning algorithm.

![](./images/Pasted%20image%2020231005203850.png)

**Aircraft engines monitoring example**

![](./images/Pasted%20image%2020231005205455.png)

 > The downside of this **alternative** here is that after you've tuned your algorithm, you don't have a fair way to tell how well this will actually do on future examples because you don't have the test set. But when your dataset is small, especially when the number of anomalies you have, your dataset is small, this might be the best alternative you have.

**Algorithm evaluation**

+ Fit model $p(x)$ on training set $x(1),x(2),...,x(m)$
+ On a cross validation/test example $x$, predict

$$y=\begin{cases}1&ifp(x)<\varepsilon(anomaly) \\
0&ifp(x)\geq\varepsilon(normal)\end{cases}$$


+ Possible evaluation metrics:
	- True positive, false positive, false negative, true negative
	- Precision/Recall
	- F<sub>1</sub>-score
+ Use cross validation set to choose parameter $\epsilon$

___
***Anomaly detection vs. supervised learning***

![](./images/Pasted%20image%2020231005211306.png)
![](./images/Pasted%20image%2020231005211601.png)

___
***Choosing what features to use***

**Non-gaussian features**

![](./images/Pasted%20image%2020231005212214.png)

> It turns out a **larger** value of $C$, will end up transforming this distribution **less**.

> Reminder: whatever transformation you apply to the training set, please remember to apply the same transformation to your **cross validation and test set** data as well.

**Error analysis for anomaly detection**

![](./images/Pasted%20image%2020231005213135.png)

**Monitoring computers in a data center**

![](./images/Pasted%20image%2020231005213408.png)

## Practice quiz: Anomaly detection

1. You are building a system to detect if computers in a data center are malfunctioning. You have 10,000 data points of computers functioning well, and no data from computers malfunctioning. What type ofalgorithm should you use?

+ A. Anomaly detection
+ B. Supervised learning

My Answer: A

> [!check] Correct
> 
> Creating an anomaly detection model does not require labeled data.

2. You are building a system to detect if computers in a data center are malfunctioning. You have 10,000 data points of computers functioning well, and 10,000 data points of computers malfunctioning. What type of algorithm should you use?

+ A. Anomaly detection
+ B. Supervised learning

My Answer: B

> [!check] Correct
> 
> You have a sufficient number of anomalous examples to build a supervised learning model.

3. Say you have 5,000 examples of normal airplane engines, and 15 examples of anomalous engines. How would you use the 15 examples of anomalous engines to evaluate your anomaly detection algorithm?

+ A. Put the data of anomalous engines (together with some normal engines) in the cross-validation and/or test sets to measure if the learned model can correctly detect anomalous engines.
+ B. Because you have data of both normal and anomalous engines, don't use anomaly detection. Use supervised learning instead.
+ C. Use it during training by fitting one Gaussian model to the normal engines, and a different Gaussian model to the anomalous engines.
+ D. You cannot evaluate an anomaly detection algorithm because it is an unsupervised learning algorithm.

My Answer: A

> [!check] Correct
> 
> Anomalous examples are used to evaluate rather than train the model.

4. Anomaly detection flags a new input $x$ as an anomaly if $p(x)<\epsilon$. lf we reduce the value of $\epsilon$, what happens?

+ A. The algorithm is more likely to classify new examples as an anomaly.
+ B. The algorithm is less likely to classify new examples as an anomaly.
+ C. The algorithm is more likely to classify some examples as an anomaly, and less likely to classify some examples as an anomaly. lt depends on the example $x$.
+ D. The algorithm will automatically choose parameters $\mu$ and $\sigma$ to decrease $p(x)$ and compensate.

My Answer: B

> [!check] Correct
> 
> When ϵ is reduced, the probability of an event being classified as an anomaly is reduced.

5. You are monitoring the temperature and vibration intensity on newly manufactured aircraft engines. You have measured 100 engines and fit the Gaussian model described in the video lectures to the data. The 100 examples and the resulting distributions are shown in the figure below.

The measurements on the latest engine vou are testing have a temperature of 17.5 and a vibration intensity of 48. These are shown in magenta on the figure below. What is the probability of an engine having these two measurements?

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/0b4675ef-89e7-487f-a8a8-3e089a81a817image2.png?expiry=1696636800000&hmac=S5V25guzF2IMPZs-kl_LJHAWgDMuZnvBNf2c0mt8Eag)

+ A. 17.5+48=65.5
+ B. 0.0738\*0.02288=0.00169
+ C. 0.0738+0.02288=0.0966
+ D. 17.5\*48=840

My Answer: B

> [!check] Correct
> 
> According to the model described in lecture, p(A, B) = p(A) * p(B). .

## Practice Lab 2

[Programming Assignment: Anomaly Detection](C3_W1_lab2/C3_W1_Anomaly_Detection.ipynb)

# Week2 Recommender systems

Learning Objectives

- Implement collaborative filtering recommender systems in TensorFlow
- Implement deep learning content based filtering using a neural network in TensorFlow
- Understand ethical considerations in building recommender systems

## Collaborative filtering

Collaborative filtering: 协同过滤

___
***Making recommendations***

<font  face="Times New Roman" color=green size=5><u>2023-10-05 22:05 $\downarrow$</u></font>

**Predicting movie ratings**

![](./images/Pasted%20image%2020231005221206.png)

___
***Using per-item features***

<font  face="Times New Roman" color=green size=5><u>2023-10-06 15:19 $\downarrow$</u></font>

**What if we have features of the movies?**

![](./images/Pasted%20image%2020231006152742.png)

**Cost function**

Notation:

+ $r(i, j) = 1$ if user $j$ has rated movie $i$ ($0$ otherwise)
+ $y^{(i,j)}$ = rating given by user $j$ on movie $i$ (if defined)
+ $w^{(j)}, b^{(j)}$ = parameters for user $j$
+ $x^{(i)}$ = feature vector for movie $i$

For user $j$ and movie $i$, predict rating: $w^{(j)}\cdot x^{(i)} + b^{(j)}$

+ $m^{(j)}$ = no. of movies rated by user $j$
+ To learn $w^{(j)}$, $b^{(j)}$

$$
\underset{w^{(j)},b^{(j)}}{\text{min}}J(w^{(j)},b^{(j)})=\frac{1}{2m^{(j)}}\sum_{i:r(i,j)=1}\bigl(w^{(j)}\cdot x^{(i)}+b^{(j)}-y^{(i,j)}\bigr)^{2}+\frac{\lambda}{2\xcancel{m^{(j)}}}\sum_{k=1}^{n}\left(w_{k}^{(j)}\right)^{2}
$$

Regularization term

$$\frac{\lambda}{2m^{(j)}}\sum_{k=1}^{n}\left(w_{k}^{(j)}\right)^{2}$$

+ $n$: number of features, same as the number of numbers in $w^{(j)}$
+ $\xcancel{m^{(j)}}$: $m^{(j)}$ is just a **constant** in this expression

![](./images/Pasted%20image%2020231006160402.png)

___
***Collaborative filtering algorithm***

**Problem motivation**

![](./images/Pasted%20image%2020231006161313.png)

**Cost function**

![](./images/Pasted%20image%2020231006161927.png)

**Collaborative filtering**

![](./images/Pasted%20image%2020231006162556.png)

**Gradient Descent**

![](./images/Pasted%20image%2020231006162844.png)

___
***Binary labels: favs, likes and clicks***

**Binary labels**

![](./images/Pasted%20image%2020231006163729.png)

**Example applications**

1. Did user $j$ purchase an item after being shown? <font color=blue>1/0/?</font>
2. Did user $j$ fav/like an item? <font color=blue>1/0/?</font>
3. Did user $j$ spend at least 30sec with an item? <font color=blue>1/0/?</font>
4. Did user $j$ click on an item? <font color=blue>1/0/?</font>

Meaning of ratings:

+ <font color=blue>1</font> - engaged after being shown item
+ <font color=blue>0</font> - did not engage after being shown item
+ <font color=blue>?</font> - item not yet shown

**From regression to binary classification**

![](./images/Pasted%20image%2020231006164422.png)

**Cost function for binary application**

![](./images/Pasted%20image%2020231006165206.png)

## Practice quiz: Collaborative filtering

1. You have the following table of movie ratings:

| Movie            | Elissa | Zach | Barry | Terry |
| ---------------- | ------ | ---- | ----- | ----- |
| Football Forever | 5      | 4    | 3     | ?     |
| Pies, Pies, Pies | 1      | ?    | 5     | 4     |
| Linear Algebra   | 4      | 5    | ?     | 1     |

Refer to the table above for question 1 and 2. Assume numbering starts at 1 for this quiz, so the rating for *Football Forever* by *Elissa* is at (1,1)

What is the value of $n_u$

My Answer: 4

> [!check] Correct
> 
> This is the number of users. $n_m$​ is the number of movies/items and is 3 in this table.

2. What is the value of $r(2,2)$

My Answer: 0

> [!check] Correct
> 
> $r(i,j)$ is a 1 if the movie has a rating and 0 if it does not. In the table above, a question mark indicates there is no rating.

3. In which of the following situations will a collaborative filtering system be the most appropriate learning algorithm (compared to linear or logistic regression)?

+ A. You manage an online bookstore and you have the book ratings from many users. You want to learn to predict the expected sales volume (number of books sold) as a function of the average rating of a book.
+ B. You subscribe to an online video streaming service, and are not satisfied with their movie suggestions. You download all your viewing for the last 10 years and rate each item. You assign each item a genre. Using your ratings and genre assignment, you learn to predict how you will rate new movies based on the genre.
+ C. You run an online bookstore and collect the ratings of many users. You want to use this to identify what books are "similar" to each other (i.e., if a user likes a certain book, what are other books that they might also like?)
+ D. You're an artist and hand-paint portraits for your clients. Each client gets a different portrait (of themselves) and gives you 1-5 star rating feedback, and each client purchases at most 1 portrait, You'd like to predict what rating your next customer will give you.

My Answer: C

> [!check] Correct
> 
> You can find "similar" books by learning feature values using collaborative filtering.

4. For recommender systems with binary labels $y$, which of these are reasonable ways for defining when $y$ should be $1$ for a given user $j$ and item $i$? (Check all that apply.)

+ A. $y$ is $1$ if user $j$ has been shown item $i$ by the recommendation enginey
+ B. $y$ is $1$ if user $j$ has not yet been shown item $i$ by the recommendation enginey 
+ C. $y$ is $1$ if user $j$ fav/likes/clicks on item $i$ (after being shown the item)
+ D. $y$ is $1$ if user $j$ purchases item $i$ (after being shown the item)

My Answer: CD

> [!check] Correct
> 
> + C. $y$ is $1$ if user $j$ fav/likes/clicks on item $i$ (after being shown the item)
> 
> > fav/likes/clicks on an item shows a user's interest in that item. It also shows that an item is interesting to a user.
> + $y$ is $1$ if user $j$ purchases item $i$ (after being shown the item)
> 
> > Purchasing an item shows a user's preference for that item. It also shows that an item is preferred by a user.

## Recommender systems implementation detail

___
***Mean normalization***

**Users who have not rated any movies**

![](./images/Pasted%20image%2020231006215815.png)

**Mean normalization**

![](./images/Pasted%20image%2020231006220703.png)

> It turns out that by normalizing the mean of the different movies ratings to be zero, the optimization algorithm for the recommended system will also run just a little bit faster. But it does make the algorithm behave much better for users who have rated no movies or very small numbers of movies. And the predictions will become more reasonable.

___
***TensorFlow implementation of collaborative filtering***

<font  face="Times New Roman" color=green size=5><u>2023-10-07 15:57 $\downarrow$</u></font>

> TensorFlow can **automatically** figure out for you what are the derivatives of the cost function.
> 
> This is a very powerful feature of TensorFlow called **Auto Diff**.
> 
> And some other machine learning packages like **pytorch** also support Auto Diff.

**Derivatives in ML**

![](./images/Pasted%20image%2020231007160051.png)

**Custom Training Loop**

![](./images/Pasted%20image%2020231007160706.png)

**Implementation in TensorFlow**

![](./images/Pasted%20image%2020231007161536.png)

___
***Finding related items***

![](./images/Pasted%20image%2020231007162304.png)

**Limitations of Collaborative Filtering**

Cold start problem. How to

+ rank **new** items that few users have rated?
+ show something **reasonable** to new users who have rated few items?

Use side information about items or users:

+ Item: Genre, movie stars, studio, ....
+ User: Demographics (age, gender, location), expressed preferences，...

## Practicelab 1

[Programming Assignment: Collaborative Filtering Recommender Systems](C3_W2_lab1/C3_W2_Collaborative_RecSys_Assignment.ipynb)

## Practice quiz: Recommender systems implementation

1. Lecture described using 'mean normalization' to do feature scaling of the ratings. What equation below best describes this algorithm?

+ A. 

$$\begin{aligned}
y_{norm}(i,j)& \begin{aligned}=y(i,j)-\mu_i\quad&\text{where}\end{aligned}  \\
\mu_{i}& =\frac1{\sum_jr(i,j)}\sum_{j:r(i,j)=1}y(i,j) 
\end{aligned}$$

+ B. 

$$\begin{aligned}
y_{norm}(i,j)& =\frac{y(i,j)-\mu_i}{max_i-min_i}\quad\text{where}  \\
\mu_{i}& =\frac1{\sum_jr(i,j)}\sum_{j:r(i,j)=1}y(i,j) 
\end{aligned}$$

+ C. 

$$\begin{aligned}
y_{norm}(i,j)& =\frac{y(i,j)-\mu_i}{\sigma_i}\quad\text{where}  \\
\mu_{i}& =\frac1{\sum_jr(i,j)}\sum_{j:r(i,j)=1}y(i,j)  \\
\sigma_{i}^{2}& =\frac1{\sum_jr(i,j)}\sum_{j:r(i,j)=1}(y(i,j)-\mu_j)^2 
\end{aligned}$$

My Answer: B $\checkmark[ChatGPT]$

> [!failure] Incorrect
> 
> This is a mean normalization algorithm as described in Course 1 and could be used, but not quite what was described in lecture. The division by $max_i​−min_i​$ will reduce the range of the ratings but they are already constrained to small values.

> [!check] Correct Answer
> 
> $$\begin{aligned}
> y_{norm}(i,j)& =y(i,j)-\mu_i\quad\text{where}  
> \\
> \mu_{i}& =\frac1{\sum_jr(i,j)}\sum_{j:r(i,j)=1}y(i,j) 
> \end{aligned}$$
> 
> This is the mean normalization algorithm described in lecture. This will result in a zero average value on a per-row basis.

2. The implementation of collaborative filtering utilized a custom training loop in TensorFlow. ls it true that TensorFlow always requires a custom training loop?

+ A. Yes. TensorFlow gains flexibility by providing the user primitive operations they can combinein many ways.
+ B. No: TensorFlow provides simplified training operations for some applications.

My Answer: B $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Recall in Course 2, you were able to build a neural network using a ‘model’, ‘compile’, ‘fit’, sequence which managed the training for you. A custom training loop was utilized in this situation because training $w$, $b$, and $x$ does not fit the standard layer paradigm of TensorFlow's neural network flow. There are alternate solutions such as custom layers, however, it is useful in this course to introduce you to this powerful feature of TensorFlow.

3. Once a model is trained, the 'distance' between features vectors gives an indication ofhow similar items are.
The squared distance between the two vectors $x^{(k)}$ and $x^{(i)}$ is:

$$distance=\left\|\mathbf{x^{(k)}}-\mathbf{x^{(i)}}\right\|^2=\sum_{l=1}^n(x_l^{(k)}-x_l^{(i)})^2$$

Using the table below, find the closest item to the movie "pies, Pies, Pies"

| Movie               | User 1 | ... | User n | $x_0$ | $x_1$ | $x_2$ |
| ------------------- | ------ | --- | ------ | ----- | ----- | ----- |
| Pastries for Supper |        |     |        | 2.0   | 2.0   | 1.0   |
| Pies, Pies, Pies    |        |     |        | 2.0   | 3.0   | 4.0   |
| Pies and You        |        |     |        | 5.0   | 3.0   | 4.0   |

+ A. Pies and You
+ B. Pastries for Supper

My Answer: A $\checkmark[ChatGPT]$

> [!check] Correct
> 
> The distance from ‘Pies, Pies, Pies’ is 9 + 0 + 0 = 9.

4. Which of these is an example of the cold start problem? (Check all that apply.)

+ A. A recommendation system takes so long to train that users get bored and leave.
+ B. A recommendation system is unable to give accurate rating predictions for a new product that no users have rated.
+ C. A recommendation system is so computationally expensive that it causes your computer CPU to heat up, causing your computer to need to be cooled down and restarted.
+ D. A recommendation system is unable to give accurate rating predictions for a new user that has rated few products.

My Answer: BD $\checkmark[ChatGPT]$

> [!check] Correct
> 
> + A recommendation system is unable to give accurate rating predictions for a new product that no users have rated.
> 
> > A recommendation system uses product feedback to fit the prediction model.
> 
> + A recommendation system is unable to give accurate rating predictions for a new user that has rated few products.
> 
> > A recommendation system uses user feedback to fit the prediction model.


## Content-based filtering

___
***Collaborative filtering vs Content-based filtering***

**Collaborative filtering vs Content-based filtering**

Collaborative filtering:

Recommend items to you based on <u>ratings of users who gave similar ratings as you</u>

Content-based filtering:

Recommend items to you based on <u>features of user and item to find good match</u>

> In other words, it requires having some features of each user, as well as some features of each item and it uses those features to try to decide which items and users might be a good match for each other.

+ <font color="#00b0f0">$r(i,j)=1$</font> if user $j$ has rated item $i$
+ <font color="#00b0f0">$y^{(i,j)}$</font> rating given by user $j$ on item $i$ (if defined)

**Examples of user and item features**

![](./images/Pasted%20image%2020231007171030.png)

**Content-based filtering: Learning to match**

![](./images/Pasted%20image%2020231007171711.png)

___
***Deep learning for content-based filtering***

**Neural network architecture**

![](./images/Pasted%20image%2020231007173614.png)

$\Downarrow$ **draw them together in a single diagram**

![](./images/Pasted%20image%2020231007174129.png)

**Learned user and item vectors:**

![](./images/Pasted%20image%2020231007174355.png)

___
***Recommending from a large catalogue***

**How to efficiently find recommendation from a large set of items?**

![](./images/Pasted%20image%2020231007175235.png)

**Two steps: Retrieval & Ranking**

<u>Retrieval:</u>

+ Generate large list of plausible item candidates
	- e.g.
	- (1) For each of the last 10 movies watched by the user, find 10 most similar movies 
	> <font color="#00b0f0">$\left\|\mathbf{v_{m}^{(k)}}-\mathbf{v_{m}^{(i)}}\right\|^2$</font>
	- (2) For most viewed 3 genres, find the top 10 movies
	- (3) Top 20 movies in the country
+ Combine retrieved items into list, removing duplicates and items already watched/purchased

<u>Ranking:</u>

![](./images/Pasted%20image%2020231007180320.png)

**Retrieval step**

+ Retrieving more items results in better performance, but slower recommendations.
+ To analyse/optimize the trade-off, carry out **offline experiments** to see if retrieving additional items results in more relevant recommendations (i.e., $p^{(y(i,j)}) = 1$ of items displayed to user are higher)

___
***Ethical use of recommender systems***

**What is the goal of the recommender system?**

Recommend:

+ Movies most likely to be rated 5 stars by user
+ Products most likely to be purchased
+ Ads most likely to be clicked on
+ Products generating the largest profit
+ Video leading to maximum watch time

**Ethical considerations with recommender systems**

![](./images/Pasted%20image%2020231007203953.png)

**Other problematic cases:**

+ Maximizing user engagement (e.g. watch time) has led to large social media/video sharing sites to amplify conspiracy theories and hate/toxicity
- Amelioration : Filter out problematic content such as hate speech, fraud, scams and violent content

+ Can a ranking system maximize your profit rather than users' welfare be presented in a transparent way?
- Amelioration : Be transparent with users

___
***TensorFlow implementation of content-based filtering***

![](./images/Pasted%20image%2020231007205052.png)

## Practice Quiz: Content-based filtering

1. Vector $x_u$ and vector $x_m$ must be of the same dimension, where $x_u$ is the input features vector for a user (age, gender, etc.) $x_m$ is the input features vector for a movie (year, genre, etc.) True or false?

My Answer: False $\checkmark[ChatGPT]$

> [!check] Correct
> 
> These vectors can be different dimensions.

2. lf we find that two movies, $i$ and $k$, have vectors $v_m^{(i)}$ and $v_m^{(k)}$ that are similar to each other (i.e., $\left\| v_m^{(i)}-v_m^{(k)}\right\|$ is small), then which of the following is likely to be true? Pick the best answer.

+ A. The two movies are similar to each other and will be liked by similar users.
+ B. We should recommend to users one of these two movies, but not both.
+ C. A user that has watched one of these two movies has probably watched the other as well.
+ D. The two movies are very dissimilar.

My Answer: A $\checkmark[ChatGPT]$

> [!check] Correct
> 
> Similar movies generate similar $v_m$​’s.

3. Which of the following neural network configurations are valid for a content based filtering application? Please note carefully the dimensions of the neural network indicated in the diagram. Check all the options that apply:

+ A. ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/62d86c86-30aa-41a3-a88b-bad03438f032image4.png?expiry=1696809600000&hmac=fpLat8rVptLBw0eFjwpe1skOQRnM1dTg2cDNFydROTI)Both the user and the item networks have the same architecture
+ B. ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/62d86c86-30aa-41a3-a88b-bad03438f032image5.png?expiry=1696809600000&hmac=yYt6ID2cHnAt0-6hRPIYzHBuylxzsptbGSeq4BfidMI)The user vector $v_u$ is 32 dimensional, and the item vector $v_m$ is 64 dimensional
+ C. ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/62d86c86-30aa-41a3-a88b-bad03438f032image3.png?expiry=1696809600000&hmac=OACJ9BM_6BG4SYaR0XzmaOTmAEh6hCUVJlYQhqEQ6so)The user and the item networks have different architectures
+ D. ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/62d86c86-30aa-41a3-a88b-bad03438f032image2.png?expiry=1696809600000&hmac=BlU09zygb-91tUuzT7MiP9k8CoqfkPSyJe7tWi5Wevs)The user and item networks have 64 dimensional $v_u$ and $v_m$ vector respectively

My Answer: ACD $\checkmark[ChatGPT]$

> [!check] Correct
> 
> + A. User and item networks can be the same or different sizes.
> + C. User and item networks can be the same or different sizes.
> + D. Feature vectors can be any size so long as $v_u$​ and $v_m$​ are the same size.

4. You have built a recommendation system to retrieve musical pieces from a large database of music, and have an algorithm that uses separate retrieval and ranking steps. If you modify the algorithm to add more musical pieces to the retrieved list (i.e., the retrieval step returns more items), which of these are likely to happen? Check all that apply.

+ A. The system's response time might decrease (i.e., users get recommendations more quickly)
+ B. The quality of recommendations made to users should stay the same or improve.
+ C. The quality of recommendations made to users should stay the same or worsen.
+ D. The system's response time might increase (i.e., users have to wait longer to get recommendations)

My Answer: BD $?$

> [!check] Correct
> 
> + The quality of recommendations made to users should stay the same or improve.
> > A larger retrieval list gives the ranking system more options to choose from which should maintain or improve recommendations.
> + The system’s response time might increase (i.e., users have to wait longer to get recommendations)
> > A larger retrieval list may take longer to process which may _increase_ response time.

5. To speed up the response time of your recommendation system, you can pre-compute the vectors $v_m$ for all the items you might recommend. This can be done even before a user logs in to your website and even before you know the $x_u$ or $v_u$ vector. True/False?

My Answer: True $\checkmark[ChatGPT]$

> [!check] Correct
> 
> The output of the item/movie neural network, $v_m$​ is not dependent on the user network when making predictions. Precomputing the results speeds up the prediction process.

## Practice lab 2

[Programming Assignment: Deep Learning for Content-Based Filtering](C3_W2_lab2/C3_W2_RecSysNN_Assignment.ipynb)

## Principal Component Analysis

主成分分析 (principal components analysis, 简称PCA)

___
***Reducing the number of features (optional)***

<font  face="Times New Roman" color=green size=5><u>2023-10-09 21:13 $\downarrow$</u></font>

> Principal Component Analysis: This is an algorithm that is commonly used for visualization.

**Car measurements**

![](./images/Pasted%20image%2020231009211719.png)

![](./images/Pasted%20image%2020231009212025.png)

**Size**

![](./images/Pasted%20image%2020231009212630.png)

> In practice, PCA is usually used to reduce a very large number of features, say 10, 20, 50, even thousands of features, down to maybe two or three features so that you can visualize the data in a two-dimensional or in a three-dimensional plot.

**From 3D to 2D**

![](./images/Pasted%20image%2020231009212820.png)

50 $\rightarrow$ 2

![](./images/Pasted%20image%2020231009213002.png)

**Data visualization**

![](./images/Pasted%20image%2020231009213214.png)

___
***PCA algorithm (optional)***

![](./images/Pasted%20image%2020231009214348.png)

**Choose an axis**

![](./images/Pasted%20image%2020231009214632.png)
![](./images/Pasted%20image%2020231009214751.png)
![](./images/Pasted%20image%2020231009214951.png)

![](./images/Pasted%20image%2020231009215139.png)

**Coordinate on the new axis**

![](./images/Pasted%20image%2020231009215427.png)

**More principal components**

![](./images/Pasted%20image%2020231009215607.png)

**PCA is not linear regression**

![](./images/Pasted%20image%2020231009220003.png)

> When linear regression is used to predict a target output Y and PCA is trying to take a lot of features and treat them all equally and reduce the number of axis needed to represent the data well.

![](./images/Pasted%20image%2020231009220058.png)

**Approximation to the original data**

![](./images/Pasted%20image%2020231009220426.png)

___
***PCA in code (optional)***

**PCA in scikit-learn**

![](./images/Pasted%20image%2020231009220832.png)

**Example**

![](./images/Pasted%20image%2020231009221143.png)

![](./images/Pasted%20image%2020231009221512.png)

> 2D $\rightarrow$ 2D: This isn't that useful for visualization but it might help us understand better how PCA and how they code for PCA works.

**Applications of PCA**

![](./images/Pasted%20image%2020231009221808.png)

___
***Lab: PCA and data visualization (optional)***

[PCA and data visualization (optional)](C3_W2_pca_op/C3_W2_Lab01_PCA_Visualization_Examples.ipynb)



# Week3 Reinforcement learning

Learning Objectives

- Understand key terms such as return, state, action, and policy as it applies to reinforcement learning
- Understand the Bellman equations
- Understand the state-action value function
- Understand continuous state spaces
- Build a deep Q-learning network

## Reinforcement learning introduction

___
***What is Reinforcement Learning?***

<font  face="Times New Roman" color=green size=5><u>2023-10-12 14:46 $\downarrow$</u></font>

**Autonomous Helicopter**

![](./images/Pasted%20image%2020231012144844.png)

How to fly it?

**Reinforcement Learning**

![](./images/Pasted%20image%2020231012145354.png)

**Robotic Dog Example**

![](./images/Pasted%20image%2020231012145505.png)

**Application**

+ Controlling robots
+ Factory optimization
+ Financial (stock) trading
+ Playing games (including video games)

> And the key idea is rather than you needing to tell the algorithm what is the right output y for every single input, all you have to do instead is specify **a reward function** that tells it when it's doing well and when it's doing poorly. And it's the job of the algorithm to **automatically** figure out how to choose good actions.

___
***Mars rover example***

![](./images/Pasted%20image%2020231012150452.png)

___
***The Return in reinforcement learning***

**Return**

![](./images/Pasted%20image%2020231012153735.png)

> discount factor: $\gamma$
> 
> In many reinforcement learning algorithms, a common choice for the discount factor will be a number pretty close to 1, like 0.9, or 0.99, or even 0.999.

**Example of Return**

![](./images/Pasted%20image%2020231012154225.png)

___
***Making decisions: Policies in reinforcement learning***

> a policy in reinforcement learning algorithm

**Policy**

![](./images/Pasted%20image%2020231012154624.png)

A policy is a function $\pi(s)=a$ mapping from states to actions, that tells you waht action $a$ to take in a given state $s$.

**The goal of reinforcement learning**

![](./images/Pasted%20image%2020231012154948.png)

Find a policy $\pi$ that tells you what action ($a = \pi(s)$) to take in every state ($s$) so as to maximize the return.

___
***Review of key concepts***

![](./images/Pasted%20image%2020231012155444.png)

**Markov Decision Process(MDP)**

![](./images/Pasted%20image%2020231012155648.png)

## Practice quiz: Reinforcement learning introduction

1. You are using reinforcement learning to control a four legged robot. The position of the robot would be its

+ A. state
+ B. action
+ C. return
+ D. reward

My Answer: A

> [!check] Correct
> 
> state

2. You are controlling a Mars rover. You will be very very happy if it gets to state 1 (significant scientific discovery), slightly happy if it gets to state 2 (small scientific discovery), and unhappy if it gets to state 3 (rover is permanently damaged). To reflect this, choose a reward function so that:

+ A. R(1)> R(2)> R(3), where R(1), R(2) and R(3) are positive.
+ B. R(1) < R(2) < R3), where R(1) and R(2) are negative and R(3) is positive.
+ C. R(1) > R(2)> R(3), where R(1), R(2) and R(3) are negative.
+ D. R(1) > R(2)> R(3), where R(1) and R(2) are positive and R(3) is negative.

My Answer: D

> [!check] Correct
> 
> R(1) > R(2)> R(3), where R(1) and R(2) are positive and R(3) is negative.

3. You are using reinforcement learning to fly a helicopter, Using a discount factor of 0.75, your helicopter starts in some state and receives rewards -100 on the first step, -100 on the second step, and 1000 on the third and final step (where it has reached a terminal state). What is the return?

+ A. $-0.25*100-0.252*100 +0.25^3*1000$
+ B. $-0.75*100-0.75^2*100 +0.75^3*1000$
+ C. $-100-0.25*100+0.25^2*1000$
+ D. $-100-0.75*100 +0.75^2*1000$

My Answer: B

> [!failure] Incorrect
> 
> Remember the first reward is not discounted.

> [!check] Correct Answer
> 
> -100 - 0.75*100 + 0.75^2*1000

4. Given the rewards and actions below, compute the return from state 3 with a discount factor of $\gamma =0.25$.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/ab8c69e8-08f0-4ad4-a4f6-cb67e091a5fbimage2.png?expiry=1697241600000&hmac=fjIwDb_Yqo7pnEzqpiPKgBKy-It3tw6QB6yddIqeqCg)

+ A. 0
+ B. 6.25
+ C. 25
+ D. 0.39

My Answer: B

> [!check] Correct
> 
> If starting from state 3, the rewards are in states 3, 2, and 1. The return is 0+(0.25)×0+(0.25)<sup>2</sup>×100=6.25.


## State-action value function

___
***State-action value function definition***

**State-action value function(Q-function)**

![](./images/Pasted%20image%2020231012161955.png)

**Picking actions**

![](./images/Pasted%20image%2020231012162638.png)

The best possible return from state $s$ is $\underset{a}{max}\ Q(s,a)$.

The best possible action in state $s$ is the action $a$ that gives $\underset{a}{max}\ Q(s,a)$.

$Q^{*}$: Optimal $Q$ function

___
***State-action value function example***

In this Jupyter notebook, you can modify the mars rover example to see hoe the values of Q(s,a) will change depending on the rewards and discount factor changing.

```python
import numpy as np
from utils import *

# DO not modify
num_states = 6
num_actions = 2

terminal_left_reward = 100
terminal_right_reward = 40
each_step_reward = 0

# discount factor
gamma=0.9

# probability of going in the wrong direction
misstep_prob = 0

generate_visualization(terminal_left_reward,terminal_right_reward,each_step_reward,gamma,misstep_prob)
```

___
***State-action value function (optional lab)***

[State-action value function (optional lab)](<C3_W3_lab1_op/State-action value function example.ipynb>)

___
***Bellman Equation***

![](./images/Pasted%20image%2020231012201114.png)

> $R(s)$: immediate reward

![](./images/Pasted%20image%2020231012201637.png)

**Explanation of Bellman Equation**

$Q(s,a)$ = return if you

+ start in state $s$.
+ take action $a$ (once).
+ then behave optimally after that.

The best possible return from state $s\prime$ is $\underset{a\prime}{\text{max}}Q(s\prime,a\prime)$

$$
Q(s,a) = R(s)+\gamma\ \underset{a\prime}{\text{max}}Q(s\prime,a\prime)
$$

![](./images/Pasted%20image%2020231012202809.png)

![](./images/Pasted%20image%2020231012203619.png)

___
***Random(stochastic) environment(Optional)***

**Stochastic Environment**

![](./images/Pasted%20image%2020231012204157.png)

**Expected Return**

![](./images/Pasted%20image%2020231012204626.png)

> The job of reinforcement learning algorithm is to choose a policy Pi to maximize the average or the expected sum of discounted rewards.

Goal of Reinforcement Learning:
Choose a policy $\pi(s) = a$ that will tell us what action $a$ to take in state $s$ so as to maximize the expected return.

![](./images/Pasted%20image%2020231012204842.png)


## Quiz: State-action value function

1. Which of the following accurately describes the state-action value function $Q(s,a)$?

+ A. It is the return if you start from state $s$, take action $a$ (once), then behave optimally after that.
+ B. It is the return if you start from state $s$ and repeatedly take action $a$.
+ C. It is the return if you start from state $s$ and behave optimally.
+ D. It is the immediate reward if you start from state $s$ and take action $a$ (once).

My Answer: A

> [!check] Correct
> 
>   It is the return if you start from state $s$, take action $a$ (once), then behave optimally after that.

2. You are controlling a robot that has 3 actions: $\leftarrow$(left), $\rightarrow$ (right) and STOP. From a given state $s$, you have computed $Q(s,\leftarrow)=-10$ , $Q(s,\rightarrow)=-20$ , $Q(s, STOP)=0$ .
What is the optimal action to take in state $s$?

+ A. STOP
+ B. $\leftarrow$(left)
+ C. $\rightarrow$(right)
+ D. Impossible to tell

My Answer: D

> [!failure] Incorrect
> 
> Actually, since there are only 3 possible actions and we have their values, we can determine the optimal action among these three.

> [!check] Correct Answer
> 
> > STOP
> 
> Yes, because this has the greatest value.

3. For this problem, $\gamma = 0.25$. The diagram below shows the return and the optimal action from each state. Please compute $Q(5,\leftarrow)$.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/3cbf737d-d132-4a1f-8e14-8865ff330e28image3.png?expiry=1697241600000&hmac=3J1TyharIM4aTsmeMmI0GnszOOh6mPBYvmbYgjW7kIc)

+ A. 0.625
+ B. 0.391
+ C. 1.25
+ D. 2.5

My Answer: A

> [!check] Correct
> 
>   Yes, we get $0$ reward in state 5. Then $0∗0.25$ discounted reward in state 4, since we moved left for our action. Now we behave optimally starting from state 4 onwards. So, we move right to state 5 from state 4 and receive $0∗0.25^2$ discounted reward. Finally, we move right in state 5 to state 6 to receive a discounted reward of $40∗0.25^3$. Adding these together we get $0.625$.

## Continuous state spaces

___
***Example of continuous state space applications***

**Discrete vs Continuous State**

![](./images/Pasted%20image%2020231012210635.png)

**Autonomous Helicopter**

![](./images/Pasted%20image%2020231012210857.png)

___
***Lunar lander***

![](./images/Pasted%20image%2020231012211829.png)

**Reward Function**

+ Getting to landing pad: 100 - 140
+ Additional reward for moving toward/away from pad.
+ Crash:-100
+ Soft landing: +100
+ Leg grounded: +10
+ Fire main engine: -0.3
+ Fire side thruster: -0.03

**Lunar Lander Problem**

![](./images/Pasted%20image%2020231012212234.png)

___
***Learning the state-value function***

<font  face="Times New Roman" color=green size=5><u>2023-10-15 19:58 $\downarrow$</u></font>

**Deep Reinforcement Learning**

![](./images/Pasted%20image%2020231015200424.png)

**Bellman Equation**

![](./images/Pasted%20image%2020231015201138.png)

**Learning Algorithm**

![](./images/Pasted%20image%2020231015201642.png)

> The algorithm you just saw is sometimes called the DQN algorithm which stands for Deep Q-Network because you're using deep learning and neural network to train a model to learn the Q functions.

___
***Algorithm refinement: Improved neural network architecture***

**Deep Reinforcement Learning**

![](./images/Pasted%20image%2020231015204921.png)
![](./images/Pasted%20image%2020231015205118.png)
___
***Algorithm refinement: $\epsilon$ -greedy policy***

**Learning Algorithm**

![](./images/Pasted%20image%2020231015205412.png)

**How to choose actions while still learning?**

![](./images/Pasted%20image%2020231015210719.png)

> Why do we want to occasionally pick an action randomly? Well, here's why. Suppose there's some strange reason that Q(s,a) was initialized randomly so that the learning algorithm thinks that firing the main thruster is never a good idea. Maybe the neural network parameters were initialized so that Q(s,main) is always very low. If that's the case, then the neural network, because it's trying to pick the action a that maximizes Q(s,a), it will never ever try firing the main thruster. Because it never ever tries firing the main thruster, it will never learn that firing the main thruster is actually sometimes a good idea. Because of the random initialization, if the neural network somehow initially gets stuck in this mind that some things are bad idea, just by chance, then option 1, it means that it will never try out those actions and discover that maybe is actually a good idea to take that action, like fire the main thrusters sometimes. Under option 2 on every step, we have some small probability of trying out different actions so that the neural network can learn to overcome its own possible preconceptions about what might be a bad idea that turns out not to be the case.

___
***Algorithm refinement: Mini-batch and soft updates (optional)***

**How to choose actions while still learning?**

![](./images/Pasted%20image%2020231015211922.png)

**Mini-batch**

![](./images/Pasted%20image%2020231015212122.png)
![](./images/Pasted%20image%2020231015212414.png)

**Learning Algorithm**

![](./images/Pasted%20image%2020231015212629.png)

**Soft Update**

![](./images/Pasted%20image%2020231015212901.png)

___
***The state of reinforcement learning***

**Limitations of Reinforcement Learning**

+ Much easier to get to work in a simulation than a real robot!
+ Far fewer applications than supervised and unsupervised learning
+ But ... exciting research direction with potential for future applications.

## Quiz: Continuous state spaces

1. The Lunar Lander is a continuous state Markov Decision Process (MDP) because:

+ A. The state has multiple numbers rather than only a single number (such as position in the $x$-direction)
+ B. The state-action value $Q(s, a)$ function outputs continuous valued numbers
+ C. The state contains numbers such as position and velocity that are continuous valued.
+ D. The reward contains numbers that are continuous valued

My Answer: C

> [!check] Correct
> 
> The state contains numbers such as position and velocity that are continuous valued.

2. In the learning algorithm described in the videos, we repeatedly create an artificial training set to which we apply supervised learning where the input $x = (s, a)$ and the target, constructed using Bellman's equations, is $y$ = ?

+ A. $y = \underset{a\prime}{max} Q(s\prime, a\prime)$ where $s\prime$ is the state you get to after taking action $a$ in state $s$
+ B. $y = R(s\prime)$ where $s\prime$ is the state you get to after taking action $a$ in state $s$
+ C. $y= R(s)$
+ D. $y = R(s) + \gamma\ \underset{a\prime}{max}\ Q(s\prime, a\prime)$ where $s\prime$ is the state you get to after taking action $a$ in state $s$

My Answer: D

> [!check] Correct
> 
> $y = R(s) + \gamma\ \underset{a\prime}{max}\ Q(s\prime, a\prime)$ where $s\prime$ is the state you get to after taking action $a$ in state $s$

3. You have reached the final practice quiz of this class! What does that mean? (Please check all the answers, because all of them are correct!)

+ You deserve to celebrate!
+ What an accomplishment -- you made it!
+ Andrew sends his heartfelt congratulations to you!
+ The DeepLearning.AI and Stanford Online teams would like to give you a round of applause!

## Practice Lab: Reinforcement Learning

[Programming Assignment: Reinforcement Learning](C3_W3_lab2/C3_W3_A1_Assignment.ipynb)


## Summary and thank you

**Courses**

+ Supervised Machine Learning: Regression and Classification
	- Linear regression, logistic regression, gradient descent
+ Advanced Learning Algorithms
	- Neural networks, decision trees, advice for ML
+ Unsupervised Learning, Recommenders, Reinforcement Learning
	- Clustering, anomaly detection, collaborative filtering, content-based filtering,reinforcement learning

## Conversations with Andrew (Optional)

***Andrew Ng and Chelsea Finn on AI and Robotics***



## Acknowledgments

<font  face="Times New Roman" color=green size=5><u>2023-10-15 22:12 $\Uparrow$</u></font>

Check Quiz Answer

<font  face="Times New Roman" color=green size=5><u>2023-10-28 17:31 $\Uparrow$</u></font>