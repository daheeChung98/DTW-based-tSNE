# DTW-based-tSNE
High-dimensional trajectory data visualization with DTW based t-SNE

## t-SNE

### Stochastic Neighbor Embedding
$$P_{j|i} = \frac{exp(-\|x_i-x_j\|^2/2\sigma_i^2)}{\sum_{k \ne i} exp(-\|x_i-x_k\|^2/2\sigma_i^2)}$$
$$q_{j|i} = \frac{exp(-\|y_i-y_j\|^2)}{\sum_{k \ne i} exp(-\|y_i-y_k\|^2)}$$
$$C = \sum\limits_i KL(P_i\|Q_i)=\sum\limits_i\sum\limits_jp_{j|i}\log\frac{p_{j|i}}{q_{j|i}}$$
