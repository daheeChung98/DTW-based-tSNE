# DTW-based-tSNE
High-dimensional trajectory data visualization with DTW based t-SNE

## t-SNE

### Stochastic Neighbor Embedding
$$P_{j|i} = \frac{exp(-\|x_i-x_j\|^2/2\sigma_i^2)}{\sum_{k \ne i} exp(-\|x_i-x_k|\|^2/2\sigma_i^2)}$$
