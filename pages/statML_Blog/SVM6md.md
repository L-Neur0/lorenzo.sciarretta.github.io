---
layout: default
permalink: /blog6
---
# Support Vector Machine

An other approach to classification is the **Support Vector Machine**
(SVM). It is a powerful and flexible class of supervised algorithms for
both classification and regression.

## Optimal Separating Hyperplane

::: definition
In a p-dimensional space, a **hyperplane** is a flat affine subspace of
dimension $p-1$. In general the equation of a hyperplane is:
$$\sum_{i=1}^{p} \beta_i x_i = 0$$ The vector of
$\beta = (\beta_1, ..., \beta_p)$ is called the **Normal Vector** and is
orthogonal to the surface of the hyperplane.
:::

We can use the hyperplane to separate the classes in a classification
problem for example. A smart way to do that is to find the hyperplane
that maximizes the margin between the two classes. This is called
**maximal margin hyperplane**. So it is a constraint optimization
problem, we want to maximize the margin $M$.

## Suppot Vector Classifier

Data cannot be always split linearly. This is often the case, unless
$N<p$. Sometimes the data are separable, but noisy. This can led to a
poor solution for the maximal margin classifier. This is the case of the
SVM model called *Soft Margin Classifier* or *Support Vector
Classifier*.\
\
In this case the *Support Vector Classifier* maximizes a **soft margin**
and minimize classification errors.\
\
It is the solution to the optimization problem:
$$\max_{\beta_0, \beta_1, \ldots, \beta_p} M \tag{5.0}$$
$$\beta_0, \beta_1, \ldots, \beta_p, \epsilon_1, \ldots, \epsilon_n, M \tag{5.1}$$

subject to

$$\sum_{j=1}^p \beta_j^2 = 1,  \tag{5.2}$$

$$y_i \left( \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} \right) \geq M (1 - \epsilon_i), \tag{5.3}$$

$$\epsilon_i \geq 0,
\sum_{i=1}^n \epsilon_i \leq C, \tag{5.4}$$

where $C$ is a nonnegative tuning parameter, called *regularization
parameter* or *Budget*. $M$ is the width of the margin; we seek to make
this quantity as large as possible. In (5.3),
$\epsilon_1, \ldots, \epsilon_n$ are *slack variables*, orthogonal to
the surface of the hyperplane., that allow individual observations to be
on the wrong side of the margin or the hyperplane; we will explain them
in greater detail momentarily. C is treated as a tuning parameter that
is generally chosen via cross-validation.\
\
**Remark:** C here is considered as a *hyperparameter* that we decide
and that constraints the margin. C bounds the sum of the $\epsilon_i$'s,
and so it determines the number and severity of the violations to the
margin (and to the hyperplane) that we will tolerate. For $C > 0$ no
more than C observations can be on the wrong side of the hyperplane,
because if an observation is on the wrong side of the hyperplane then
$\epsilon_i > 1$.\
. C also controls the bias-variance tradeoff, a small C, so a small
margin, leds to low *bias* but high variance, and viceversa.\
\
In other words, margin maximization can be interpreted as
regularization.

#### Standardizing the Variables

\
In Support vector machine models the variables needs to be standardized.
This is because the support vector classifier is not scale invariant. In
all the calculations we are using the euclidean distance!

## Support Vector Machine

The support vector classifier is a natural approach for classification
in the two-class setting, if the boundary between the two classes is
linear. Often, it happens that this boundry is not linear. The solution
is *feature expansion*, we can enlarge the features space using
quadratic or cubic functions for the predictors.\
\
For example we can fit a SVC using instead of $p$ predictors we can use
$2p$ predictors using : $$X_1, X_1^2, X_2, X_2^2 \cdots X_p, X_2^p$$.
Then the optimization problem becomes:

$$\max_{\beta_0, \beta_1, \ldots, \beta_p} M$$ subject to:
$$y_i(\beta_0 \sum_{j=1}^{p} \beta_j x_{ij} + \sum_{j=1}^{p} \beta_{j2} x_{ij}^2) \geq M(1 - \epsilon_i)$$
$$\epsilon_i \geq 0, \sum_{i=1}^{n} \epsilon_i \leq C, \quad \sum_{j=1}^{p} \sum_{k=1}^{2} \beta_{jk}^2 = 1$$\
\
This is not the only way to enlarge the feature space. The support
vector machine (SVM) is an extension of the support vector support
classifier that results from enlarging the feature space in a specific
way, vector using *kernels*.\
\
The kernel approach that we describe here is simply an efficient
computational approach for enacting this idea. The solution to the
(5.0)--(5.4) optimization problem can be expressed in terms of the
*inner products* of pairs of observations. Thus, the inner product of
two observations, vectors of $p$ components, $x_i, x_{i'}$ is ;
$$\langle x_i, x_{i'} \rangle = \sum_{j=1}^{p} x_{ij} x_{i'j} \tag{5.5}$$\
Moreover, we can represent the SVC as:
$$f(x) = \beta_0 + \sum_{i=1}^{n} \alpha_i \langle x, x_i \rangle$$
where the $\alpha_i$ are the $n$ solutions to the optimization problem.
To estimate them we need the $\binom{n}{2}$ inner products
$\langle x, x_{i} \rangle$ between all pairs of new point $x$ and $x_i$
training points. However, it turns out that $\alpha_i$ is non zero only
for the support vectors in the solution, the points that lie on the
margin or on the wrong side of the margin.\
\
We suppose that $\cal S$ is the set of indices of the support points,
the *Support Set*. Then we can rewrite (5.5) as:
$$f(x) = \beta_0 + \sum_{i \in \cal S} \alpha_i \langle x, x_i \rangle$$\
\
In other words, in representing the linear classifier $f(x)$, and in
computing its coefficients, all we need are inner products. This is
where the kernel comes in. The kernel is a function that computes a
$generalization$ of the inner product between two vectors:
$$K(x_i, x_{i'})$$ and K, the $Kernel$ function we can use is e.g. the
**Polynomial Kernel**:
$$K(x_i, x_{i'}) = (1 + \sum_{j=1}^{p} x_{ij} x_{i'j})^d \tag{5.6}$$\
where $d$ is the degree of the polynomial. It computes the inner product
needed for $d$ dimensional polynomials, $\binom{p+d}{d}$ basis
functions.\
\
The solution is in the form of:
$$f(x) = \beta_0 + \sum_{i \in \cal S} \hat \alpha_i K(x, x_i)$$\
\
Another choice is the **Radial Kernel**:
$$K(x_i, x_{i'}) = \exp \left( - \gamma \sum_{j=1}^{p} (x_{ij} - x_{i'j})^2 \right) \tag{5.7}$$\
where $\gamma$ is a tuning parameter and remember that
$\sum_{j=1}^{p} (x_{ij} - x_{i'j})^2$ = $\|x_i - x_{i'}\|$. If $\gamma$
is large, the kernel will have a small variance and the decision
boundary will be very wiggly. If $\gamma$ is small, the kernel will have
a large variance and the decision boundary will be smoother. The *Radial
Kernel* works pretty well in high dimensional spaces, it controls the
variance by squashing down most of the dimensions. If a given
observation $x^*$ is far from the trainig data $x_i$, in terms of
Euclidean distance then the $\sum_{j=1}^{p} (x_{ij} - x_{i'j})^2$ will
be large, and so the kernel will be small. This means that in $f(x)$ the
observation $x^*$ will have a very small weight. This means that the
*Radial Kernel* is a very local method, it is very sensitive to the
distance between the observations.\
\
**Remark:** The choice of the kernel is for computational advantages.

### Support Vector Machine for More than Two Classes

We have two methods to extend the SVM to more than two classes:

1.  **One-versus-One** (OvO): We fit a binary classifier for each pair
    of classes. We have $\binom{K}{2}$ classifiers. We classify a new
    observation by majority vote.

2.  **One-versus-All** (OvA): We fit a binary classifier for each class
    versus the rest. We have $K$ 2-class classifiers. We classify a new
    observation by the classifier that gives the highest score.

We choose OVO if the number $K$ of classes is small, and OvA if the
number of classes is large.

### Relationship to Logistic Regression

It tuns out that we can rewrite the (5.0) - (5.4) optimization problem
for fitting the $f(x) = \beta_0 + beta_1X_1+ ... + \beta_p X_p$ as:
$$\min_{\beta_0, \beta_1, \ldots, \beta_p} \left( \sum_{i=1}^{n} \max(0, 1 - y_i f(x_i)) + \lambda \sum_{j=1}^{p} \beta_j^2 \right)$$

where $\lambda$ is a tuning parameter for the violation of the margin.
The equation is known as the\
**loss + penalty**, where the loss is called the *hinge loss*, similar
to the loss in logistic regression .

#### Summary 

-   When class are (nearly) separable, SVM does better than LR. So does
    LDA.

-   When not, LR (with ridge penalty) and SVM are very similar.

-   If you wish to estimate probabilities, LR is the choice.

-   For nonlinear boundaries, kernel SVMs are popular. Can use kernels
    with LR and LDA as well, but computations are more expensive.