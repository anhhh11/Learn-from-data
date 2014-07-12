set.seed(1000)
x <- seq(-1,1,0.001)
y <- sin(pi*x)
org <- data.frame(x1=x,y1=y)
fit <- lm(y~x-1)
num_sample_data <- 1000
g_neg <- 0
var <- rep(0,num_sample_data)
for ( i in seq(1,num_sample_data)){
  x1 <- runif(2,-1,1)
  y1 <- sin(pi*x1)
  fit_d <- lm(y1~x1-1)
  g_neg <- g_neg + fit_d$coef
  p <- predict(object=fit_d,org)
  var[i] <- fit_d$coef
}
g_neg <- g_neg / num_sample_data
g_x_neg <- g_neg*x
for (i in seq(1,num_sample_data)){
  var[i] <- mean((var[i]*x-g_x_neg)^2) 
}
var <- mean(var)
bias <- mean((g_x_neg - y)^2)
g_neg
var
bias
