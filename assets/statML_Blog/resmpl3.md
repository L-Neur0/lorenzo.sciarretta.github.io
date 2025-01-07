
### Cross-Validation {#sec:cross_validation}

#### Validation Set approach

We focus on the *test error* of the model and we decide to split
randomly the data into two parts: a **training set** and a **test
set**.\
We fit the model on the training set and then we evaluate the model on
the test set.\
\
The resulting validation hold-out set set error rate---typically
assessed using MSE in the case of a quantitative response---provides an
estimate of the test error rate.\
\
**Problems:**

-   The MSE is very variable, because it depends on the random split of
    the data. **High Bias**.

-   We use only few of the available observation to train the model\...

#### K-Fold Cross-Validation

K-Fold Cross-Validation is a resampling method that addresses the 2
problems of the validation set approach. The idea is that we divide the
dataset into $K$ folds, group, and that we train the model on $K-1$
folds and evaluate it on the remaining fold. This is done $K$ times, one
time for each fold, computing for each time the corresponding $MSE_k$
for that evaluation fold.\
\
This will result in a MSE for each fold, and we can compute the average
of the MSEs to get the **Cross-Validation Error Rate**.
$$CV_{(K)} = \frac{1}{K} \sum_{k=1}^{K} MSE_k$$\
\
We then have a special case of the K-Fold Cross-Validation, the
**Leave-One-Out Cross-Validation** (LOOCV), where we set $K = n$. this
means that we leave one obsevation out, and we train the model on the
remaining $n-1$ observations, and we evaluate it on the left-out
observation. This is done for each observation (n), and we compute the
average of the MSEs to get the **LOOCV Error Rate**.
$$CV_{(n)} = \frac{1}{n} \sum_{k=1}^{n} MSE_k$$\
\
**Advantages LOOCV:**

-   No randomness because we test the model on all observations.

-   Lower Bias because we use all the data to train the model. (But High
    Variance)

**Advantages K-Fold:**

-   Faster (computationally) than LOOCV.

-   Higher Bias.

-   Lower Variance because we use group of training sets that are less
    correlated between each other than the ones mostly equal, used in
    LOOCV.

#### Bootstrapping

::: definition
**(Bootstrap)** The bootstrap is a resampling technique used to estimate
statistics (such as mean, variance, CI, SE) on a population by
resempling the data with replacement.
:::

How and why use it?\
We use bootstrap when we do not have access to the population and we
mimick the population by resampling the data. We use it to create
**Bootstrap samples** that are used to estimate the statistics of a
population (i.e. statistics on different samples).\
\
Suppose that we have a sample of observations, and we want to get some
estimation on statistics of the population o those observations. We
should have more samples of from the same population and then average
the results on those samples. But we do not have access to the
population, so we can use the bootstrap to create more samples from the
same sample. we do that by **resampling with replacement** the data,
creating new bootstrap samples of the same size of the orinal sample.
This way, we can have multiple samples that appears to be drawn from the
same population, but actually are just resampled from the original
sample, but if we resample with replacement, it is the same as getting
them from that population.\

#### Bootstrap Estimate of the SE

For example, we create $R$ bootstrap samples of size $n$ from the
original sample of size $n$, and we compute the statistic of interest on
each of the $R$ samples, and then we average the results to get the
final estimation of the statistic. And if we do that for example with
the sample mean $\bar{x}$, we can get the **Bootstrap estimate of the
SE** of the sample mean, which is the standard deviation of the
bootstrap samples.
$$SE_{\bar{x}} = \sqrt{\frac{1}{R-1} \sum_{r=1}^{R} (\bar{x}^*_r - \bar{x}^*)^2}$$
where $\bar{x}^* = \frac{1}{R} \sum_{i=1}^{R} \bar{x_i}^*$ is the mean
of the bootstrap samples means. This means that the SD of the bootstrap
samples means is the Bootstrap estimate of the SE of the sample mean of
the original sample.\

::: definition
**(Standard Error)** Standard error is the measure of the variability of
the sample mean, i.e. how much the sample mean changes if we take
different samples from the same population. This is relative to the
sample size: $$SE = \frac{\sigma}{\sqrt{n}}$$ where
$\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$ is the
population standard deviation.
:::

#### Bootstrap Confidence Intervals

We can also use the bootstrap to estimate the confidence intervals of a
statistic. We just have to compute the statistic of interest on the
bootstrap samples, and then we can compute the confidence intervals by
just **sorting in ascending order the bootstrap statistics** computed
and then find the quantiles that we want for our Confidence Interval.

#### Limitations of the Bootstrap

-   The bootstrap cannot be used with time series data. Becaese each
    data is dependent time and so there is an order.

-   The bootstrap is not useful for computing the **Prediction Error**
    of a model, it is not a resampling method that is used to evaluate
    the model.