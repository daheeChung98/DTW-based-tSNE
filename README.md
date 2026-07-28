# DTW-based-tSNE
High-dimensional trajectory data visualization with DTW based t-SNE

## t-SNE

### Stochastic Neighbor Embedding

The similarity of datapoint $x_i$ is the conditional probability, $p_{j_i}$, that $x_i$ would pick $x_j$ as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at $x_i$.

For nearby datapoints $x_i$ and $x_j$, $p_{j|i}$ is relatively high, whereas for widely separated datapoints, $p_{j|i}$ will be almost infinitesimal.

Mathematically, the conditonal probability $p_{j|i}$ is given by

$$
P_{j|i} = \frac{\exp(-\|x_i-x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i-x_k\|^2 / 2\sigma_i^2)}
$$

where $\sigma_i$ is the variance of the Gaussian that is centered on datapoint $x_i$.

For the low-dimensional counterparts $y_i$ and $y_j$ of the which we denote by $q_{j|i}$. Hence, we model the similarity of map point $y_j$ to map point $y_i$ by

$$q_{j|i} = \frac{exp(-\|y_i-y_j\|^2)}{\sum_{k \ne i} exp(-\|y_i-y_k\|^2)}$$

Since we are only interested in modeling pairwise similarities, we set $p_{i|i}=q_{i|i}=0$.

If the map points $y_i$ and $y_j$ correctly model the similarity between the high-dimensional datapoints $x_i$ and $x_j$, the conditional probabilities $p_{j|i}$ and $q_{j|i}$ will be equal. Motivated by this observation, SNE aims to find a low-dimensional data representation that minimizes the mismatch between $p_{j|i}$ and $q_{j|i}$.

SNE minimizes the sum of Kullback-Leibler divergences over all datapoints using a gradient descent method. The cost function $C$ is given by

$$C = \sum\limits_i KL(P_i\|Q_i)=\sum\limits_i\sum\limits_jp_{j|i}\log\frac{p_{j|i}}{q_{j|i}}$$

in which $P_i$ represents the conditional probability distribution over all other datapoints given datapoint $x_i$, and $Q_i$ represents the conditioanl probability distribution over all other map points given map point $y_i$.

### Perplexity

The variance $\sigma_i$ of the Gaussian that is optimal for all datapoints in the data set because the density of the data is likely to vary. In dense regoins, a smaller value of $\sigma_i$ is usually more appropriate than in sparser regions. This distribution has an entropy which increases as $\sigma_i$ increases.

SNE performs a binary search for the value of $\sigma_i$ that produces a $P_i$ with a fixed perplexity that is specified by the user. The perplexity increases monotonically with the variance $\sigma_i$.

$$
Perp(P_i)=2^{H(P_i)}
$$

where $H(P_i)$ is the Shannon entropy of $P_i$ measured in bits

$$
H(P_i)=-\sum\limits_j p_{j|i}\log_2 p_{j|i}
$$

The perplexity can be interpreted as a smooth measure of the effective number of neighbors. The performance of SNE is fairly robust to changes in the perplexity, and typical values are between 5 and 50.

### Crowding Problem

In the original high-dimensional space, SNE models local neighborhood relationships as probabilities using a Gaussian kernel.

However, when a Gaussian kernel is also used in the low-dimensional embedding space, its thin-tailed distribution causes the similarity between distant points to decrease very rapidly toward zero.

As a result, the low-dimensional space cannot accommodate all pairwise relationships while preserving the original distances. Consequently, points that should remain relatively far apart are forced to crowd together near the center of the embedding. This phenomenon is known as the crowding problem.

To address this issue, t-SNE replaces the Gaussian distribution in the low-dimensional space with a Student's t-distribution (with one degree of freedom).

Because the Student's t-distribution has much heavier tails, the similarity between distant points decreases more gradually rather than collapsing to zero. This allows dissimilar points to remain farther apart in the embedding space.

Consequently, t-SNE produces a more spread-out embedding that better preserves the neighborhood structure of the original high-dimensional data while also maintaining the global arrangement more effectively than SNE.

### t-Distributed Stochastic Neighbor Embedding (t-SNE)

Although SNE constructs reasonably good visualizations, it is hampered by a cost function that is difficult to optimize and by a problem we refer to as the "crowding problem". In this paper, authors presented a new technique called "t-Distributes Stochastic Neighbor Embedding (t-SNE)" that aims to alleviate these problems. The cost function used by t-SNE differs from the one used by SNE in two ways:

- It uses a symmetrized version of the SNE cost function with simpler gradients that was briefly introduced by Cook et al. (2007)
- It uses a Student-t distribution rather than a Gaussian to compute the similarity between two points in the low-dimensional space.

t-SNE employs a heavy-tailed distribution in the low-dimensional space to alleviate both the crowding problem and the optimization problems to SNE.

Using this distribution, the joint probabilities $q_{ij}$ are defined as

$$
q_{ij}=\frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum\limits_{k \neq l} (1+\|y_k-y_l\|^2)^{-1}}
$$

---

## Similarity Measure

t-SNE first computes pairwise Euclidean distances in the high-dimensional space, converts those distances into neighborhood probabilities using a Gaussian kernel, and then learns a low-dimensional embedding whose Student's t-distribution–based similarities closely match the original probability distribution by minimizing the KL divergence.

Let's take a closer look at these probabilities.

$$
p_{ij} = \frac{\exp(-\|x_i - x_j\|^2/2\sigma^2)}{\sum\limits_{k \neq l}\exp(-\|x_k - x_l\|^2/2\sigma^2)}
$$

t-SNE first computes pairwise Euclidean distances in the high-dimensional space, converts those distances into neighborhood probabilities using a Gaussian kernel, and then learns a low-dimensional embedding whose Student's t-distribution based similarities closely match the original probability distribution by minimizing the KL divergence.

However, we soon encountered a fundamental limitation.

All of these probability distributions $p_{ij}$ are built upon pairwise Euclidean distances ($\|x_i - x_j\|^2$). This works well when each observation is represented as a static feature vector. But what if each observation is a trajectory?

A trajectory is fundamentally different from a static data point. It is a sequence that evolves over time, and therefore its similarity cannot always be captured by pointwise Euclidean distance. Two trajectories may describe nearly identical motion patterns while appearing far apart under Euclidean distance because of temporal misalignment. This led us to a fundamental question:

If Euclidean distance is no longer appropriate, what similarity measure should be used instead?

Answering this question became the starting point of our research.

### Euclidean Distance

### Fréchet Distance

### Dynamic Time Warping (DTW)

Dynamic Time Warping (DTW) is a similarity measure designed for sequential or trajectory data. 
Unlike Euclidean distance, which compares observations at the same time index, 
DTW allows the time axis to be stretched or compressed so that similar patterns occurring at different speeds can still be aligned.

This makes DTW particularly useful when two trajectories have similar shapes but differ in timing or sampling frequency.

**Algorithm**

1. Compute the pairwise distance matrix between two trajectories.
2. Initialize the first row and first column of the cumulative cost matrix.
3. For each remaining cell, add the current distance to the minimum cumulative cost from the left, upper, or diagonal neighbor.
4. Continue until reaching the bottom-right corner of the matrix.
5. The cumulative cost at the final cell is the DTW distance, and the optimal alignment can be obtained by backtracking through the minimum-cost path.

Simply put, each cell can be filled with
```
D(i, j) = d(i, j) + min{D(i-1, j), D(i, j-1), D(i-1, j-1)}
```

### Multidimensional DTW

**Independent DTW**

**Dependent DTW**
