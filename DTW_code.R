UNIdtw <- function(P, Q){
  M <- length(P)
  N <- length(Q)
  # Cost Matrix
  D <- matrix(0, M + 1, N + 1)
  # Initial Cell
  D[2:(M + 1), 1] <- Inf
  D[1, 2:(N + 1)] <- Inf
  # Cell
  for (i in 2:(M + 1)) {
    for (j in 2:(N + 1)) {
      distE <- sqrt((P[i - 1]-Q[j - 1])^2)
      D[i, j] <- distE + min(D[i - 1, j],
                             D[i - 1, j - 1],
                             D[i, j - 1])
    }
  }
  return(D[M + 1, N + 1]) # DTW
}
# example
Q = c(1,4,2,6,5,3)
P = c(2,6,3,1)
UNIdtw(P, Q)

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

