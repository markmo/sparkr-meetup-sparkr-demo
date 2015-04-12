# Mirror matrix (left-right)
mirror.matrix <- function(x) {
  xx <- as.data.frame(x);
  xx <- rev(xx);
  xx <- as.matrix(xx);
  xx;
}

createImage <- function(x) {
  mirror.matrix(matrix(x, nrow=28))
}