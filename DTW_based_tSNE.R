#DTWD

DTW_D <- function(P, Q) {
  # Sequence length (each variable has same length, but each subject has different length)
  M <- nrow(P) # data set P,Q has only numeric variables
  N <- nrow(Q)
  # Number of variables
  V <- ncol(P) # or ncol(Q) since, each subject has same variables
  # Cost Matrix
  D <- matrix(0, M, N)
  # Initial Cell
  # D[1, 1]
  D[1, 1] <- 0
  D[1, 1] <- sum((P[1, ]-Q[1, ])^2)
  # D[1, ]
  for (l in 2:N) {
    distD <- 0
    distD <- sum((P[1, ]-Q[l, ])^2)
    D[1, l] <- distD + min(D[0, l - 1],
                           D[0, l],
                           D[1, l - 1])
  }
  # D[ ,1]
  for (k in 2:M) {
    distD <- 0
    distD <- sum((P[k, ]-Q[1, ])^2)
    D[k, 1] <- distD + min(D[k - 1, 0],
                           D[k - 1, 1],
                           D[k, 0])
  }
  # Cell
  for (k in 2:M) {
    for (l in 2:N) {
      distD <- 0
      distD <- sum((P[k, ]-Q[l, ])^2)
      D[k, l] <- distD + min(D[k - 1, l - 1],
                             D[k - 1, l],
                             D[k, l - 1])
    }
  }
  return(D[M, N]) # dependent DTW
}

# Distance matrix (upper triangular)
n <- length(sub0)
distDTWD <- matrix(0, n, n)
for (i in 1:(n - 1)) {  
  for (j in (i + 1):n) {  
    dtwd <- DTW_D(sub0[[i]][,-1], sub0[[j]][,-1]) # fullnm except
    distDTWD[i, j] <- dtwd
  }
}

distDTWD[lower.tri(distDTWD)] <- t(distDTWD)[lower.tri(distDTWD)]

# Packages
library(dtw)
library(proxy)
library(Rtsne)
library(dplyr)
library(ggplot2)
library(caret)
library(permute)

# DTW based t-SNE
tsne <- Rtsne(distDTW, is_distance = T,perplexity = 277, dims = 2)
