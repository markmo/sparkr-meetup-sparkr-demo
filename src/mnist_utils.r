d <- 784 # dimension of digits
NTrain <- 60000
NTest <- 10000

D <- 900 # number of random features
gamma <- 4e-4

# create a matrix - one row per digit
readMatrix <- function(split, part, d) {
  matList <- sapply(part, function(line) {
    as.numeric(strsplit(line, ",", fixed=TRUE)[[1]])
  }, simplify=FALSE)

  # unlist options: do not recurse beyond first level in matList, drop names
  mat <- Matrix(ncol=d+1, data=unlist(matList, F, F),
                sparse=T, byrow=T)
  mat
}

randomFeatures <- function(mat, label, b, W) {
  ones <- rep(1, nrow(mat))
  features <- cos(
    mat %*% t(W) + (matrix(ncol=1, data=ones) %*% t(b)))
  onesMat <- Matrix(ones)
  featuresPlus <- cBind(features, onesMat)
  labels <- matrix(nrow=nrow(mat), ncol=10, data=-1)
  for (i in 1:nrow(mat)) {
    labels[i, label[i]] <- 1
  }
  list(label=labels, features=featuresPlus)
}

predictLabels <- function(testData, W, b, NTest, C) {
  testData <- test[,-1]
  # contstruct the feature maps for all examples from this digit
  featuresTest <- cos(testData %*% t(W) +
                        (matrix(ncol=1, data=rep(1, NTest)) %*% t(b)))
  featuresTest <- cBind(
    featuresTest, Matrix(rep(1, NTest)))

  # extract the one vs. all assignment
  results <- featuresTest %*% C
  labelsGot <- apply(results, 1, which.max)
  labelsGot - 1
}