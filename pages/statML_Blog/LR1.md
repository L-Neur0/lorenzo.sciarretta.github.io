---
layout: default
permalink: /blog1
---
# Linear Regression

## Linear Regression {#sec:linear_regression}

Given a predictor vector $X$ and a dipendent variable $Y$, the linear
regression model is defined as: $$Y = \beta_0 + \beta_1 X + \epsilon$$
where $\epsilon$ is the error term. The goal is to estimate the
coefficients $\beta_0$ and $\beta_1$ that minimize the sum of squared
residuals: $$RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

where we indicate with $e_i = y_i - \hat{y}_i$ the **residuals error**.
The least squares estimates of $\beta_0$ and $\beta_1$ are, using the
**Ordinary Least Square (OLS)** method
$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$
$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

### Multiple Linear Regression

We define **Multiple Linear Regression** as the case in which we have
$p$ predictors $X_1, X_2, \dots, X_p$ and a dependent variable $Y$. The
model is defined as:
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon$$
where of course the $X_i$ are vectors of observations.

### Assesing the model

In order to asses the model we can use several statistics, we are here
going to itemize the most important four:

1.  ::: definition
    We define as the most important metric for assesing machine learning
    technciques as the **Root Mean Squared Error (RMSE)**:
    $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
    Which measures the OVERALL accuracy of the model.
    :::

2.  ::: definition
    The **Residual Standard Error** is similar to the RMSE but it is
    normalized by the degrees of freedom of the model, where $p$ is the
    number of predictors (mostly used with multiple linear regression):
    $$RSE = \sqrt{\frac{1}{n-p-1} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
    :::

3.  ::: definition
    The **R-squared** statistic, $R^2 \in \{0,1\}$, or the *coefficient
    of determination*, is defined as: $$R^2 = \frac{TSS - RSS}{TSS}$$
    $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y_i})^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
    where $TSS = \sum_{i=1}^{n} (y_i - \bar{y})^2$ is the **Total Sum of
    Squares** and $RSS$ is the **Residual Sum of Squares**. The $R^{2}$
    statistic measures the proportion of variation (variance) in the
    data used in the model.
    :::

4.  ::: definition
    The last statistic used by data scientists is the $t-statistic$ of a
    $\beta$ coefficent, is the ratio of the estimate of the coefficient
    and its standard error:
    $$t_{\beta} = \frac{\hat{\beta}}{SE(\hat{\beta})}$$ **Remark:** The
    t-statistic is the inverse proportionally to the p-value, it is its
    mirror image. The higher the t-statistic, the lower the p-value, the
    more significative is a predictor. That is used to as a tool to
    decide which variables keep as predictors in the model (**Variable
    Selection**).
    :::

### Prediction Intervals vs Confidence Intervals

::: definition
A **Prediction Interval** quantifies the uncertainty in individual
predictions, of a single value (so it s large usually!).
:::

::: definition
A **Confidence Interval** quantifies the uncertainty around regression
coefficients (mean or other **statistics**).
:::

Remember that we can compute *empirically* the C.I. using the bootstrap
method (see bootstrap).

### Categorical variables

We know that linear regression is used to predict quantitative values,
and so to train the model we need quantitative variables (numbers). But
can we use also *categorical variables*? Which are variables on a
limited number of discrete values (such as qualitative variables)?\
\
For example if we have a binary variable (yes/no belongs to something),
or a variable that is a label for classification of something. How can
we handle this in the linear regression?\
\
We can of course, the only thing that we have to do is that since
Regression requires numerical inputs, so factor(categorical) variables
need to be recoded in a \"numerical way\" to be used in the model.\
\
The most common approach is to convert a variable into a set of binary
**dummy** variables. For example a variable yes/no becomes the dummy
variable 0/1, or we can use several encoding schemes (one-hot encoding,
etc.).

### Correlated Predictors

In data science, the most important use of regression is to predict some
dependent (outcome) variable. In some cases, however, gaining insight
from the equation itself to understand the nature of the relationship
between the predictors and the outcome can be of value.\
\
We now analyze the case of multiple regression, where we can heve more
than one predictor. In this case, the predictor variables are often
correlated with each other.\
\
When do we notice this? When we fit a model and get **negative
coefficients** for some predictors! Where however we expect positive
ones (e.g. numebrs of bedrooms and Living Sqft for value of a house)!
Having correlated predictors can make it difficult to interpret the sign
and value of regression coefficients (and can inflate the standard error
of the estimates).\
\
Correlated variables are only one issue with interpreting regression
coefficients.

#### Multicollinearity

Multicollinearity is the extreme expression of collinearity, it is the
condition where two variables can be expressed as a linear combination
of the other (prefect Multicollinearity).\
This can happen if we include the same predictor twice, by error. It is
a singular problem for regression.

### Confounding Variables

A confounding variable is am external variable that is correlated with
both the dependent variable (y) and one or more independent variables,
the predictors, in a way that may create a spurious or misleading
association.\
Confounders can introduce bias in the model's predictions and make it
harder to establish a clear cause-and-effect relationship between input
features and the target.\
\
For example, suppose we want to predict the risk of a disease based on
certain health metrics like cholesterol level. If age is a confounding
variable (affecting both cholesterol level and disease risk), our model
might incorrectly interpret cholesterol alone as the risk driver when,
in fact, age might be the underlying factor influencing both.

### Polynomial Regression

The relationship between the response and a predictor variable is not
necessarily linear.\
\
Polynomial regression consists of adding polynomial terms to the
regression equation. For example, a quadratic regression model would
have the form: $$Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \epsilon$$ This
is often tested empirically, by adding polynomial terms to the model and
checking if the model improves. This could happen because the
relationship between the predictors and the response maybe be not that
linear, but more polynomial.
