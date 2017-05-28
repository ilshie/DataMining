length <- 1000
dim <- 3
sim <- matrix(rep(0,dim*length),ncol=dim)
for (i in 1:500) {
    sim[i,1] <- rnorm(1,-3,1)
    sim[i,2] <- rnorm(1,3,1)
    sim[i,3] <- rnorm(1,2,1)
    sim[i+500,1] <- rnorm(1,3,1)
    sim[i+500,2] <- rnorm(1,-3,1)
    sim[i+500,3] <- rnorm(1,-2,1)
}
write.csv(sim,"testdata.txt")
