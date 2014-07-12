x <- seq(-1,1,0.001)
y <- sin(pi*x)
org <- data.frame(x1=x,y1=y)
fit <- lm(y~x)
num_sample_data <- 1000
g_neg <- 0
var <- list()#rep(0,num_sample_data)
for ( i in seq(1,num_sample_data)){
  x1 <- runif(2,-1,1)
  y1 <- sin(pi*x1)
  fit_d <- lm(y1~x1)
  g_neg <- g_neg + fit_d$coef
  p <- predict(object=fit_d,org)
  var[[i]] <- fit_d$coef
}
g_neg <- g_neg / num_sample_data
g_x_neg <- g_neg[1] + g_neg[2]*x
ret_var = c()
for (i in seq(1,num_sample_data)){
  ret_var[i] <- mean((var[[i]][2]*x+var[[i]][1]-g_x_neg)^2) 
}
var <- mean(ret_var)
bias <- mean((g_x_neg - y)^2)
g_neg
var
bias

# With H: y = ax
# fit <- lm(y~x-1)
# g_neg
# x1 
# 1.412343 
# 
# > var
# [1] 0.2320538
# 
# > bias
# [1] 0.266644


# With H: y = b
# > g_neg
# (Intercept) 
# 0.005898376 
# 
# > var
# [1] 0.08613906
# 
# > bias
# [1] 0.4997849

# With H: y = ax + b
# > g_neg
# (Intercept)           x1 
# -0.009098882  0.751622699 
# 
# > var
# [1] 1.736339
# 
# > bias
# [1] 0.2100751
