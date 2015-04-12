setwd("/Users/markmo/src/Tutorials/SparkR/demo/src/")

library(SparkR)

Sys.setenv(SPARK_MEM="4g")
Sys.setenv(spark.executor.memory="4g")

sc <- sparkR.init(master="local[*]")

lines <- textFile(sc, "../data/speeches-test-set.txt")

splits <- cache(lapplyPartition(lines, function(line) { strsplit(as.character(line), "\\t") }))

k <- 5

shinglesByDoc <- lapply(splits, function(split) {
    s = c()
    for (i in 1:(nchar(split[[2]]) - k)) {
        s <- c(s, substr(split[[2]], i, i + k))
    }
    unique(s)
})

# this is a way to get the set of all unique shingles:
# 1) emit a (shingle, 1) tuple
# 2) reduceByKey (the total count is unused)
#
s1 <- flatMap(shinglesByDoc, function(shingles) { lapply(shingles, function(shingle) { list(shingle, 1) }) })
s2 <- reduceByKey(s1, "+", 4L)

# The superset of all possible k-shingles
allShingles <- collect(lapply(s2, function(shingle) { shingle[[1]][[1]] }))
allShinglesBroadcast <- broadcast(sc, allShingles)

characteristics <- lapply(shinglesByDoc, function(shingles) {
    lapply(value(allShinglesBroadcast), function(s) {
        s %in% shingles
        })
    })

# Build characteristic matrix
mat <- matrix(unlist(collect(characteristics)), ncol=5)
presidents <- lapply(splits, function(split) { split[[1]] })
names <- lapply(presidents, function(president) { paste(strsplit(as.character(president), " ")[[1]][1:2], collapse=" ") })
df <- data.frame(mat)
colnames(df) <- collect(names)

# Calculate the Jaccard similarity from the characteristic matrix

jaccard <- function(vec1, vec2) {
  numer = sum(vec1 & vec2)
  denom = sum(vec1 | vec2)
  return(numer/denom)
}

sim_12 <- jaccard(df[,1], df[,2])
sim_23 <- jaccard(df[,2], df[,3])
sim_13 <- jaccard(df[,1], df[,3])