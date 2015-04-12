# Demo Code Walkthrough

Install SparkR

<pre><code>
library(devtools)
install_github("amplab-extras/SparkR-pkg", subdir="pkg")
</code></pre>

Start hadoop.

Test if working:

<pre><code>
hdfs dfs -ls /
</code></pre>

Start R:

<pre><code>
R
</code></pre>

To install SparkR:

<pre><code>
library(devtools)
install_github("amplab-extras/SparkR-pkg", subdir="pkg")
</code></pre>

[Documentation](http://amplab-extras.github.io/SparkR-pkg/rdocs/1.2/index.html)

<pre><code>
library(SparkR)
Sys.setenv(SPARK_MEM="4g")
Sys.setenv(spark.executor.memory="4g")
</code></pre>

Create the SparkContext

<pre><code>
sc <- sparkR.init(master="local[*]")
</code></pre>

Don't run spark-shell in another shell otherwise "java.net.BindException: Address already in use"

<pre><code>
sc

lines <- textFile(sc, "hdfs://localhost:9000/shakespeare.txt")
take(lines, 2)

words <- flatMap(lines, function(line) { strsplit(as.character(line), " ")[[1]] })
wordCount <- lapply(words, function(word) { list(word, 1) })
counts <- reduceByKey(wordCount, "+", 2L)
output <- collect(counts)

df <- data.frame(matrix(unlist(output), ncol=2, byrow=T))
colnames(df) <- c("word", "freq")

# word and freq come in as factors
df$word <- as.character(df$word)
df$freq <- as.integer(levels(df$freq))[df$freq]
sorted <- df[order(-df$freq),]
</code></pre>

## MNIST Demo

<pre><code>
setwd("/Users/markmo/src/Tutorials/SparkR/demo/src")

options(java.parameters="-Xmx4g")
library(SparkR)
library(Matrix)

# machine learning library
library(caret)

source("mnist_utils.r")
source("image_utils.r")

# local machine, 8 cores
sc <- sparkR.init("local[8]")

# training file
# the label is the first column
file <- textFile(sc,
                 "../data/train-mnist-dense-with-labels.data", 8)

# random features
# D is number of random features: 900; d is number of pixels: 784
W <- gamma * matrix(nrow=D, ncol=d, data=rnorm(D*d))
b <- 2 * pi * matrix(nrow=D, ncol=1, data=runif(D))

# make above variables available on all workers
# Broadcast variables allow the programmer to keep a read-only variable
# cached on each machine rather than shipping a copy of it with tasks.
# They can be used, for example, to give every node a copy of a large
# input dataset in an efficient manner. Spark also attempts to distribute
# broadcast variables using efficient broadcast algorithms to reduce
# communication cost.
wBroadcast <- broadcast(sc, W)
bBroadcast <- broadcast(sc, b)

# make following package available on all workers
includePackage(sc, Matrix)

# Parse file to create a distributed matrix
# the file is split across machines and a matrix created for each split
# Note that the readMatrix function is automatically serialized
# and available on each worker
numericLines <- lapplyPartitionsWithIndex(file,
                                          function(split, part) {
                                            readMatrix(split, part, d)
                                          })

# RDD[(Features, Labels)]
featureLabels <- cache(lapplyPartition(
  numericLines,
  function(part) {
    label <- part[,1]
    mat <- part[,-1]
    randomFeatures(mat, label, value(bBroadcast),
                   value(wBroadcast)) # vars captured in closure
  }))

##### build up normal equation

ATA <- Reduce("+", collect(
  lapplyPartition(featureLabels,
                  function(part) {
                    t(part$features) %*% part$features
                  }), flatten=F))

ATb <- Reduce("+", collect(
  lapplyPartition(featureLabels,
                  function(part) {
                    t(part$features) %*% part$label
                  }), flatten=F))


##### solve the equation locally
# This generic function solves the equation ‘a %*% x = b’ for ‘x’,
#     where ‘b’ can be either a vector or a matrix.
C <- solve(ATA, ATb)

##### read test set, calculate test error
test <- Matrix(as.matrix(read.csv(
  "../data/test-mnist-dense-with-labels.data",
  header=F), sparse=T))
testLabels <- matrix(ncol=1, test[,1]) - 1

labelsGot <- predictLabels(test, W, b, NTest, C)
err <- sum(testLabels != labelsGot) / nrow(testLabels) * 100.0
cat("Test error is ", err, "%\n")

##### Visualization: plot the confusion matrix
# confusionMatrix is from the caret package
image(t(log(1 + confusionMatrix(labelsGot, testLabels)$table)),
      axes=F, col=gray((128:0)/128))
axis(1, at=seq(0, 1.09, 0.11), labels=seq(0, 9, 1))
axis(2, at=seq(0, 1.09, 0.11), labels=seq(0, 9, 1))

# Note the darker areas outside the diagonal where the machine learning
# algorithm is getting more confused

##### Visualization: plot a random misclassified image
misclassfiedImg <- sample(which(testLabels != labelsGot),
                          size=1, replace=F)
image(createImage(test[misclassfiedImg,][-1]))
cat("Expected: ", testLabels[misclassfiedImg], " Got: ",
    labelsGot[misclassfiedImg], "\n")
</code></pre>

## Developer release - Predicting Customer Behaviour

3 datasets:

* Transactions
* Demographic Info Per Customer
* DM Treatment Sample

How do we decide who to send the offer to?

Use R’s glm method to train a logistic regression model on the treatment sample

To install the latest developer release:

<pre><code>
library(devtools)
install_github("amplab-extras/SparkR-pkg", ref="sparkr-sql", subdir="pkg")

setwd("/Users/markmo/src/Tutorials/SparkR/demo/src/")
library(SparkR)

sc <- sparkR.init(master="local[*]")
sqlCtx <- sparkRSQL.init(sc)

df <- jsonFile(sqlCtx, "../data/people.json")
head(df)
avg <- select(df, avg(df$age))
head(avg)

txnsRaw <- loadDF(sqlCtx, paste(getwd(), "/../data/Customer_Transactions.parquet", sep = ""), "parquet")
demo <- withColumnRenamed(loadDF(sqlCtx, paste(getwd(), "/../data/Customer_Demographics.parquet", sep = ""), "parquet"),
                          "cust_id", "ID")
sample <- loadDF(sqlCtx, paste(getwd(), "/../data/DM_Sample.parquet", sep = ""), "parquet")

printSchema(txnsRaw)
printSchema(demo)
printSchema(sample)

head(txnsRaw)
</code></pre>

Aggregate Transaction Data

<pre><code>
perCustomer <- agg(groupBy(txnsRaw, "cust_id"),
                   txns = countDistinct(txnsRaw$day_num),
                   spend = sum(txnsRaw$extended_price))

head(perCustomer)
</code></pre>

Bring In Demographic Data

<pre><code>
joinToDemo <- select(join(perCustomer, demo, perCustomer$cust_id == demo$ID),
                     demo$"*",
                     perCustomer$txns, 
                     perCustomer$spend)
head(joinToDemo)
explain(joinToDemo)
</code></pre>

Split Into Train/Test and convert to R data.frames

<pre><code>
trainDF <- select(join(joinToDemo, sample, joinToDemo$ID == sample$cust_id),
                  joinToDemo$"*",
                  alias(cast(sample$respondYes, "double"), "respondYes"))
head(trainDF)

estDF <- select(filter(join(joinToDemo, sample, joinToDemo$ID == sample$cust_id, "left_outer"),
                       "cust_id IS NULL"),
                joinToDemo$"*")
head(estDF)

printSchema(estDF)

persist(estDF, "MEMORY_ONLY")

count(estDF)
</code></pre>

# Convert to R data.frames

<pre><code>
train <- collect(trainDF) ; train$ID <- NULL

est <- collect(estDF)

class(est)
names(est)
summary(est)
</code></pre>

Estimate logit model, create custom scoring function, score customers

<pre><code>
theModel <- glm(respondYes ~ ., "binomial", train)

summary(theModel)

predictWithID <- function(modObj, data, idField) {
  scoringData <- data[, !names(data) %in% as.character(idField)]
  scores <- predict(modObj, scoringData, type = "response", se.fit = TRUE)
  idScores <- data.frame(ID = data[as.character(idField)], Score = scores$fit)
  idScores[order( -idScores$Score), ]
}

testScores <- predictWithID(theModel, est, "ID")

head(testScores, 25)
</code></pre>
