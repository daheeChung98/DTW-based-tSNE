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
