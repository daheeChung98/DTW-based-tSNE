# DTW-based-tSNE
High-dimensional trajectory data visualization with DTW based t-SNE

## t-SNE

### Stochastic Neighbor Embedding

The similarity of datapoint $x_i$ is the conditional probability, $p_{j_i}$, that $x_i$ would pick $x_j$ as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at $x_i$.

For nearby datapoints $x_i$ and $x_j$, $p_{j|i}$ is relatively high, whereas for widely separated datapoints, $p_{j|i}$ will be almost infinitesimal.

Mathematically, the conditonal probability $p_{j|i}$ is given by

$$P_{j|i} = \frac{exp(-\|x_i-x_j\|^2/2\sigma_i^2)}{\sum_{k \ne i} exp(-\|x_i-x_k\|^2/2\sigma_i^2)}$$,

where $\sigma_i$ is the variance of the Gaussian that is centered on datapoint $x_i$.

For the low-dimensional counterparts $y_i$ and $y_j$ of the which we denote by $q_{j|i}. Hence, we model the similarity of map point $y_j$ to map point $y_i$ by

$$q_{j|i} = \frac{exp(-\|y_i-y_j\|^2)}{\sum_{k \ne i} exp(-\|y_i-y_k\|^2)}$$.

Since we are only interested in modeling pairwise similarities, we set $p_{i|i}=q_{i|i}=0$.

If the map points $y_i$ and $y_j$ correctly model the similarity between the high-dimensional datapoints $x_i$ and $x_j$, the conditional probabilities $p_{j|i}$ and $q_{j|i}$ will be equal. Motivated by this observation, SNE aims to find a low-dimensional data representation that minimizes the mismatch between $p_{j|i}$ and $q_{j|i}$.

SNE minimizes the sum of Kullback-Leibler divergences over all datapoints using a gradient descent method. The cost function $C$ is given by

$$C = \sum\limits_i KL(P_i\|Q_i)=\sum\limits_i\sum\limits_jp_{j|i}\log\frac{p_{j|i}}{q_{j|i}}$$,

in which $P_i$ represents the conditional probability distribution over all other datapoints given datapoint $x_i$, and $Q_i$ represents the conditioanl probability distribution over all other map points given map point $y_i$.
