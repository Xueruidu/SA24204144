## ----eval=FALSE---------------------------------------------------------------
# invF <- function(u,sigma){
#   x <- sqrt(  -2*sigma^2 * log(1-u) )
#   return(x)
# }
# N <- 1000
# set.seed(1234)
# U <- runif(N,0,1)
# X <- rep(0,N)

## ----eval=FALSE---------------------------------------------------------------
# sigma <- 1
# X <- invF(U,sigma)
# hist(X,probability = TRUE,main = expression(frac(x, sigma^2) * e^{-x^2 / (2 * sigma^2)},sigma==1),breaks = 10,ylim=c(0,0.65))
# y <- seq(0,4,0.01)
# lines(y,y/sigma^2*exp(-y^2/(2*sigma^2)),col="red")

## ----eval=FALSE---------------------------------------------------------------
# sigma <- 10
# X <- invF(U,sigma)
# hist(X,probability = TRUE,main = expression(frac(x, sigma^2) * e^{-x^2 / (2 * sigma^2)},sigma==10),breaks = 10,ylim=c(0,0.07))
# y <- seq(0,40,0.01)
# lines(y,y/sigma^2*exp(-y^2/(2*sigma^2)),col="red")

## ----eval=FALSE---------------------------------------------------------------
# sigma <- 50
# X <- invF(U,sigma)
# hist(X,probability = TRUE,main = expression(frac(x, sigma^2) * e^{-x^2 / (2 * sigma^2)},sigma==50),breaks = 10,ylim=c(0,0.013))
# y <- seq(0,200,0.01)
# lines(y,y/sigma^2*exp(-y^2/(2*sigma^2)),col="red")

## ----eval=FALSE---------------------------------------------------------------
# sigma <- 1000
# X <- invF(U,sigma)
# hist(X,probability = TRUE,main = expression(frac(x, sigma^2) * e^{-x^2 / (2 * sigma^2)},sigma=1000),breaks = 10,ylim=c(0,0.0006))
# y <- seq(0,4000,0.01)
# lines(y,y/sigma^2*exp(-y^2/(2*sigma^2)),col="red")

## ----eval=FALSE---------------------------------------------------------------
# N <- 1000
# set.seed(1234)
# X1 <- rnorm(N,0,1)
# X2 <- rnorm(N,3,1)
# p1 <- 0.75
# p2 <- 1-p1
# mixf <- function(x,p1){
#   return( p1 / sqrt(2*pi) * exp(- x^2/2) + (1-p1) / sqrt(2*pi) * exp (- (x-3)^2/2) )
# }
# set.seed(123)
# p <- sample(c(0,1),N,replace = TRUE,prob = c(p1,p2))
# X <- p*X1 + p*X2
# hist(X,breaks = 50,probability = TRUE,main = paste0("p1=",p1),xlim = c(-2,5))
# y <- seq(-1,4,0.01)
# lines(y,mixf(y,p1),col="red")

## ----eval=FALSE---------------------------------------------------------------
# for(p1 in (0:8)/8){
#   p2 <- 1-p1
#   set.seed(123)
#   p <- sample(c(0,1),N,replace = TRUE,prob = c(p1,p2))
#   X <- p*X1 + p*X2
#   hist(X,breaks = 50,probability = TRUE,main = paste0("p1=",p1),xlim = c(-2,5))
#   lines(y,mixf(y,p1),col="red")
# }

## ----eval=FALSE---------------------------------------------------------------
# total <- 1000
# t <- 10
# lambda <- 1
# alpha <- 1
# beta <- 0.5
# X10 <- numeric(total)
# for(time in 1:total){
#   set.seed(1234*time)
#   N <- rpois(1,lambda*t)
#   Y <- rgamma(N,shape = alpha,scale = beta)
#   X10[time] <- sum(Y)
# }
# print(paste0("when lambda=",lambda,",alpha=",alpha,",beta=",beta))
# print(paste0("the theoretical values:E[X(10)]=",lambda*t*alpha*beta,",Var(x(10))=",lambda*t*(alpha^2*beta^2+alpha*beta^2),";"))
# print(paste0("the empirical values:E[X(10)]=",mean(X10),",Var(x(10))=",var(X10),";"))

## ----eval=FALSE---------------------------------------------------------------
# total <- 1000
# t <- 10
# lambda <- 1
# alpha <- 2
# beta <- 0.8
# X10 <- numeric(total)
# for(time in 1:total){
#   set.seed(1234*time)
#   N <- rpois(1,lambda*t)
#   Y <- rgamma(N,shape = alpha,scale = beta)
#   X10[time] <- sum(Y)
# }
# print(paste0("when lambda=",lambda,",alpha=",alpha,",beta=",beta))
# print(paste0("the theoretical values:E[X(10)]=",lambda*t*alpha*beta,",Var(x(10))=",lambda*t*(alpha^2*beta^2+alpha*beta^2),";"))
# print(paste0("the empirical values:E[X(10)]=",mean(X10),",Var(x(10))=",var(X10),";"))

## ----eval=FALSE---------------------------------------------------------------
# total <- 1000
# t <- 10
# lambda <- 1
# alpha <- 5
# beta <- 2
# X10 <- numeric(total)
# for(time in 1:total){
#   set.seed(1234*time)
#   N <- rpois(1,lambda*t)
#   Y <- rgamma(N,shape = alpha,scale = beta)
#   X10[time] <- sum(Y)
# }
# print(paste0("when lambda=",lambda,",alpha=",alpha,",beta=",beta))
# print(paste0("the theoretical values:E[X(10)]=",lambda*t*alpha*beta,",Var(x(10))=",lambda*t*(alpha^2*beta^2+alpha*beta^2),";"))
# print(paste0("the empirical values:E[X(10)]=",mean(X10),",Var(x(10))=",var(X10),";"))

## ----eval=FALSE---------------------------------------------------------------
# total <- 1000
# t <- 10
# lambda <- 3
# alpha <- 1
# beta <- 0.5
# X10 <- numeric(total)
# for(time in 1:total){
#   set.seed(1234*time)
#   N <- rpois(1,lambda*t)
#   Y <- rgamma(N,shape = alpha,scale = beta)
#   X10[time] <- sum(Y)
# }
# print(paste0("when lambda=",lambda,",alpha=",alpha,",beta=",beta))
# print(paste0("the theoretical values:E[X(10)]=",lambda*t*alpha*beta,",Var(x(10))=",lambda*t*(alpha^2*beta^2+alpha*beta^2),";"))
# print(paste0("the empirical values:E[X(10)]=",mean(X10),",Var(x(10))=",var(X10),";"))

## ----eval=FALSE---------------------------------------------------------------
# total <- 1000
# t <- 10
# lambda <- 3
# alpha <- 2
# beta <- 0.8
# X10 <- numeric(total)
# for(time in 1:total){
#   set.seed(1234*time)
#   N <- rpois(1,lambda*t)
#   Y <- rgamma(N,shape = alpha,scale = beta)
#   X10[time] <- sum(Y)
# }
# print(paste0("when lambda=",lambda,",alpha=",alpha,",beta=",beta))
# print(paste0("the theoretical values:E[X(10)]=",lambda*t*alpha*beta,",Var(x(10))=",lambda*t*(alpha^2*beta^2+alpha*beta^2),";"))
# print(paste0("the empirical values:E[X(10)]=",mean(X10),",Var(x(10))=",var(X10),";"))

## ----eval=FALSE---------------------------------------------------------------
# total <- 1000
# t <- 10
# lambda <- 3
# alpha <- 5
# beta <- 2
# X10 <- numeric(total)
# for(time in 1:total){
#   set.seed(1234*time)
#   N <- rpois(1,lambda*t)
#   Y <- rgamma(N,shape = alpha,scale = beta)
#   X10[time] <- sum(Y)
# }
# print(paste0("when lambda=",lambda,",alpha=",alpha,",beta=",beta))
# print(paste0("the theoretical values:E[X(10)]=",lambda*t*alpha*beta,",Var(x(10))=",lambda*t*(alpha^2*beta^2+alpha*beta^2),";"))
# print(paste0("the empirical values:E[X(10)]=",mean(X10),",Var(x(10))=",var(X10),";"))

## ----eval=FALSE---------------------------------------------------------------
# N <- 1000
# alpha <- 3
# beta <- 3
# Fhat <- function(x){
#   set.seed(1234)
#   U <- runif(N,0,x)
#   gU <- x * 1/beta(alpha,beta)*U^(alpha-1)*(1-U)^(beta-1)
#   return(mean(gU))
# }

## ----eval=FALSE---------------------------------------------------------------
# for(x in (1:9)/10){
#   print(paste0("when x = ",x,",the real F(x):",pbeta(x,alpha,beta),",the estimated F(x):",Fhat(x),"."))
# }

## ----eval=FALSE---------------------------------------------------------------
# invF <- function(u,sigma){
#   x <- sqrt(  -2*sigma^2 * log(1-u) )
#   return(x)
# }
# N <- 1000
# set.seed(1234)
# U <- runif(N,0,1)
# X1 <- rep(0,N/2)
# X2 <- rep(0,N/2)
# X <- rep(0,N)

## ----eval=FALSE---------------------------------------------------------------
# sigma <- 1
# X1 <- invF(U[1:(N/2)],sigma)
# X2 <- invF(U[(N/2+1):N],sigma)
# AX <- invF(1-U[1:(N/2)],sigma)
# svar1 <- sd((X1+X2)/2)^2 * N / (N-1)
# svar2 <- sd((X1+AX)/2)^2 * N /(N-1)
# print(paste0("The variance reducts ",(1-svar2/svar1)*100,"%."))

## ----eval=FALSE---------------------------------------------------------------
# g <- function(x){
#   result <- x^2/sqrt(2*pi) * exp(-x^2/2)
#   return(result)
# }
# f1 <- function(x){
#   result <- sqrt(2/pi) * exp(-(x-1)^2/2)
#   return(result)
# }
# f2 <- function(x,lambda){
#   result <- lambda * exp(-lambda*(x-1))
# }

## ----eval=FALSE---------------------------------------------------------------
# N <- 10000
# set.seed(1234)
# X <- abs(rnorm(N,0,1)) + 1
# mean(g(X)/f1(X))
# 
# lambda <- 1 ##parameter of the exponential distribution
# set.seed(1234)
# Y <- rexp(N,lambda)+1
# mean(g(Y)/f2(Y,lambda))

## ----eval=FALSE---------------------------------------------------------------
# quick_sort<-function(x){
#   num<-length(x)
#   if(num==0||num==1){return(x)
#   }else{
#     a<-x[1]
#     y<-x[-1]
#     lower<-y[y<a]
#     upper<-y[y>=a]
#     return(c(quick_sort(lower),a,quick_sort(upper)))}#??????
# }

## ----eval=FALSE---------------------------------------------------------------
# all <- 100
# time1 <- numeric(all)
# time2 <- numeric(all)
# time4 <- numeric(all)
# time6 <- numeric(all)
# time8 <- numeric(all)
# for(i in 1:all){
#   test1 <- sample(1:1e4)
#   test2 <- sample(1:(2*1e4))
#   test4 <- sample(1:(4*1e4))
#   test6 <- sample(1:(6*1e4))
#   test8 <- sample(1:(8*1e4))
#   time1[i] <- system.time(quick_sort(test1))[1]
#   time2[i] <- system.time(quick_sort(test2))[1]
#   time4[i] <- system.time(quick_sort(test4))[1]
#   time6[i] <- system.time(quick_sort(test6))[1]
#   time8[i] <- system.time(quick_sort(test8))[1]
# }
# time <- c(mean(time1),mean(time2),mean(time4),mean(time6),mean(time8))
# time

## ----eval=FALSE---------------------------------------------------------------
# n <- c(1,2,4,6,8)*1e4
# true <- n*log(n)
# lm(time~true)
# plot(true,time)
# abline(coef(lm(time~true))[1],coef(lm(time~true))[2],col="red")

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())
# n <- 1000
# N <- 1000
# 
# skew <- numeric(N)
# for(i in 1:N){
#   set.seed(i)
#   X <- rnorm(n,0,1)
#   m2 <- mean((X-mean(X))^2)
#   m3 <- mean((X-mean(X))^3)
#   skew[i] <- m3/(m2^(3/2))
# }
# hist(skew,probability = TRUE,xlim=c(-0.3,0.3),ylim = c(0,5),breaks = 12)
# y <- seq(-0.3,0.3,0.01)
# lines(y,1/(sqrt(2*pi*6/n))*exp(-y^2/(2*6/n)),col="red")

## ----eval=FALSE---------------------------------------------------------------
# q <- c(0.025,0.05,0.95,0.975)
# quantile(skew,probs=q)
# qnorm(q,0,sqrt(6/n))

## ----eval=FALSE---------------------------------------------------------------
# smean <- mean(skew)
# svar <- 1/(N-1)*sum((skew-mean(skew))^2)
# fq <- dnorm(quantile(skew,probs=c(0.025,0.05,0.95,0.975)),mean = smean,sd=sqrt(svar))
# varq <- q*(1-q)/(N*fq^2)
# varq

## ----eval=FALSE---------------------------------------------------------------
# rm(list = ls())

## ----eval=FALSE---------------------------------------------------------------
# n <- 1000
# mean <- c(0,0)
# alpha <- 0.05
# cor <- 0.1
# sigma1 <- 1
# sigma2 <- 2
# sigma <- matrix(c(sigma1^2,cor*sigma1*sigma2,cor*sigma1*sigma2,sigma2^2),nrow = 2)
# N <- 1000
# reject1 <- numeric(N)
# reject2 <- numeric(N)
# reject3 <- numeric(N)
# for(i in 1:N){
#   set.seed(i)
#   data <- rmvnorm(n,mean = mean,sigma = sigma) ##data generation
#   p1 <- cor.test(data[,1],data[,2],alternative = "two.sided",method = c("pearson"))$p.value
#   p2 <- cor.test(data[,1],data[,2],alternative = "two.sided",method = c("spearman"))$p.value
#   p3 <- cor.test(data[,1],data[,2],alternative = "two.sided",method = c("kendall"))$p.value
#   reject1[i] <- as.numeric(p1<alpha) ##rejection
#   reject2[i] <- as.numeric(p2<alpha)
#   reject3[i] <- as.numeric(p3<alpha)
# }

## ----eval=FALSE---------------------------------------------------------------
# sum(reject1)/N ##power
# sum(reject2)/N
# sum(reject3)/N

## ----eval=FALSE---------------------------------------------------------------
# rm(list = ls())

## ----eval=FALSE---------------------------------------------------------------
# generate_bivariate_exponential <- function(n, lambda1, lambda2, correlation) {
#   u1 <- rexp(n, rate = lambda1)  # first variable
#   u2 <- rexp(n, rate = lambda2)  # second variable
# 
#   # adjust the second variable through correlation
#   v <- correlation * u1 + sqrt(1 - correlation^2) * u2
# 
#   return(data.frame(X = u1, Y = v))
# }

## ----eval=FALSE---------------------------------------------------------------
# n <- 1000
# alpha <- 0.05
# lambda1 <- 1
# lambda2 <- 2
# correlation <- 0.05
# N <- 1000
# reject1 <- numeric(N)
# reject2 <- numeric(N)
# reject3 <- numeric(N)
# for(i in 1:N){
#   set.seed(i)
#   data <- generate_bivariate_exponential(n, lambda1, lambda2, correlation) #data generation
#   p1 <- cor.test(data$X,data$Y,alternative = "two.sided",method = c("pearson"))$p.value
#   p2 <- cor.test(data$X,data$Y,alternative = "two.sided",method = c("spearman"))$p.value
#   p3 <- cor.test(data$X,data$Y,alternative = "two.sided",method = c("kendall"))$p.value
#   reject1[i] <- as.numeric(p1<alpha) #rejection
#   reject2[i] <- as.numeric(p2<alpha)
#   reject3[i] <- as.numeric(p3<alpha)
# }

## ----eval=FALSE---------------------------------------------------------------
# sum(reject1)/N ##power
# sum(reject2)/N
# sum(reject3)/N

## ----eval=FALSE---------------------------------------------------------------
# N <- 1000
# n_null <- 950
# n_alt <- N - n_null
# alpha <- 0.1
# m <- 10000
# 
# results <- matrix(0, nrow = 3, ncol = 2)
# colnames(results) <- c("Bonferroni correction", "B-H correction")
# rownames(results) <- c("FWER", "FDR", "TPR")
# 
# for (i in 1:m) {
# 
#   set.seed(i)
#   p_null <- runif(n_null,0,1)
#   p_alt <- rbeta(n_alt, 0.1, 1)
#   p_values <- c(p_null, p_alt)
# 
#   bonferroni_p <- p.adjust(p_values, method = "bonferroni")
#   bh_p <- p.adjust(p_values, method = "BH")
# 
#   bonferroni_rejects <- which(bonferroni_p < alpha)
#   fwer_bon <- sum(bonferroni_rejects %in% 1:n_null) > 0
#   fdr_bon <- sum(bonferroni_rejects %in% 1:n_null) / max(length(bonferroni_rejects), 1)
#   tpr_bon <- sum(bonferroni_rejects %in% (n_null + 1):N) / n_alt
# 
#   bh_rejects <- which(bh_p < alpha)
#   fwer_bh <- sum(bh_rejects %in% 1:n_null) > 0  # If at least one null hypothesis is incorrectly rejected, it is recorded as a false positive event.
#   fdr_bh <- sum(bh_rejects %in% 1:n_null) / max(length(bh_rejects), 1) # Calculate how many of the rejected hypotheses are null hypotheses.
#   tpr_bh <- sum(bh_rejects %in% (n_null + 1):N) / n_alt # Calculate the number of correctly rejected alternative hypotheses divided by the total number of true alternative hypotheses.
# 
#   results[1, 1] <- results[1, 1] + fwer_bon
#   results[2, 1] <- results[2, 1] + fdr_bon
#   results[3, 1] <- results[3, 1] + tpr_bon
# 
#   results[1, 2] <- results[1, 2] + fwer_bh
#   results[2, 2] <- results[2, 2] + fdr_bh
#   results[3, 2] <- results[3, 2] + tpr_bh
# }
# 
# results <- results / m
# print(round(results, 4))

## ----eval=FALSE---------------------------------------------------------------
# aircondit_times <- c(3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487)
# sample_mean <- mean(aircondit_times)
# lambda_mle <- 1 / sample_mean
# lambda_mle_boot <- function(data, indices) {
#   sample_data <- data[indices]
#   return(1 / mean(sample_data))
# }
# boot_results <- boot(data = aircondit_times, statistic = lambda_mle_boot, R = 1000)
# print(boot_results)
# lambda_bias <- mean(boot_results$t) - lambda_mle
# lambda_se <- sd(boot_results$t)
# 
# paste0("MLE of lambda:", lambda_mle, "\n")
# paste0("Bias estimate:", lambda_bias, "\n")
# paste0("Standard error estimate:", lambda_se, "\n")
# 

## ----eval=FALSE---------------------------------------------------------------
# mu_boot <- function(data, indices) {
#   sample_data <- data[indices]
#   return(mean(sample_data))
# }
# boot_results <- boot(data = aircondit_times, statistic = mu_boot, R = 1000)
# 
# normal_ci <- boot.ci(boot_results, type = "norm")
# basic_ci <- boot.ci(boot_results, type = "basic")
# percentile_ci <- boot.ci(boot_results, type = "perc")
# bca_ci <- boot.ci(boot_results, type = "bca")
# 
# paste0("Normal 95% CI:", normal_ci$normal[2:3], "\n")
# paste0("Basic 95% CI:", basic_ci$basic[4:5], "\n")
# paste0("Percentile 95% CI:", percentile_ci$percent[4:5], "\n")
# paste0("BCa 95% CI:", bca_ci$bca[4:5], "\n")
# 

## ----eval=FALSE---------------------------------------------------------------
# lambda_true <- 2
# sizes <- c(5, 10, 20)
# B <- 1000
# m <- 1000
# 
# results <- data.frame(sample_size = integer(),
#                       mean_bootstrap_bias = numeric(),
#                       bootstrap_se = numeric(),
#                       theoretical_bias = numeric(),
#                       theoretical_se = numeric())
# 
# for (n in sizes) {
#   mean_bias <- numeric(m)
#   bootstrap_se <- numeric(m)
# 
#   for (i in 1:m) {
#     set.seed(i)
#     sample_data <- rexp(n, rate = lambda_true)
#     boot_results <- boot(data = sample_data, statistic = lambda_mle_boot, R = B)
#     mean_bias[i] <- mean(boot_results$t) - (1 / mean(sample_data))
#     bootstrap_se[i] <- sd(boot_results$t)
#   }
# 
#   mean_bootstrap_bias_avg <- mean(mean_bias)
#   bootstrap_se_avg <- mean(bootstrap_se)
# 
#   theoretical_bias <- lambda_true / (n - 1)
#   theoretical_se <- (lambda_true * n) / ((n - 1) * sqrt(n - 2))
# 
#   results <- rbind(results,
#                     data.frame(sample_size = n,
#                                mean_bootstrap_bias = mean_bootstrap_bias_avg,
#                                bootstrap_se = bootstrap_se_avg,
#                                theoretical_bias = theoretical_bias,
#                                theoretical_se = theoretical_se))
# }
# print(results)
# 

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())
# scor
# 
# 
# cov_scor <- cov(scor)
# lambda <- eigen(cov_scor)$values
# theta_hat <- lambda [1] / sum(lambda )
# n <- nrow(scor)
# theta_jackknife <- numeric(n)
# 
# for (i in 1:n) {
#   cov_jack <- cov(scor[-i, ])
#   lambda_jack <- eigen(cov_jack)$values
#   theta_jackknife[i] <- lambda_jack[1] / sum(lambda_jack)
# }
# 
# bias_jack <- (n - 1) * (mean(theta_jackknife) - theta_hat)
# se_jack <- sqrt((n - 1) * mean((theta_jackknife - theta_hat)^2))
# 
# round(c(original=theta_hat,bias.jack=bias_jack,
# se.jack=se_jack),5)
# 
# 

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())
# attach(ironslag)
# a <- seq(10, 40, .1)
# L1 <- lm(magnetic ~ chemical)
# plot(chemical, magnetic, main="Linear", pch=16)
# yhat1 <- L1$coef[1] + L1$coef[2] * a
# lines(a, yhat1, lwd=2)
# L2 <- lm(magnetic ~ chemical + I(chemical^2))
# plot(chemical, magnetic, main="Quadratic", pch=16)
# yhat2 <- L2$coef[1] + L2$coef[2] * a + L2$coef[3] * a^2
# lines(a, yhat2, lwd=2)
# L3 <- lm(log(magnetic) ~ chemical)
# plot(chemical, magnetic, main="Exponential", pch=16)
# logyhat3 <- L3$coef[1] + L3$coef[2] * a
# yhat3 <- exp(logyhat3)
# lines(a, yhat3, lwd=2)
# L4 <- lm(magnetic ~ chemical + I(chemical^2) + I(chemical^3))
# plot(chemical, magnetic, main="Cubic", pch=16)
# yhat4 <- L4$coef[1] + L4$coef[2] * a + L4$coef[3] * a^2 + L4$coef[4] * a^3
# lines(a, yhat4, lwd=2)

## ----eval=FALSE---------------------------------------------------------------
# n <- length(magnetic)
# e1 <- e2 <- e3 <- e4 <- numeric(n)
# for(k in 1:n) {
#   y <- magnetic[-k]
#   x <- chemical[-k]
#   J1 <- lm(y ~ x)
#   yhat1 <- J1$coef[1] + J1$coef[2] * chemical[k]
#   e1[k] <- magnetic[k] - yhat1
#   J2 <- lm(y ~ x + I(x^2))
#   yhat2 <- J2$coef[1] + J2$coef[2] * chemical[k] +
#   J2$coef[3] * chemical[k]^2
#   e2[k] <- magnetic[k] - yhat2
#   J3 <- lm(log(y) ~ x)
#   logyhat3 <- J3$coef[1] + J3$coef[2] * chemical[k]
#   yhat3 <- exp(logyhat3)
#   e3[k] <- magnetic[k] - yhat3
#   J4 <- lm(y ~ x + I(x^2) + I(x^3))
#   yhat4 <- J4$coef[1] + J4$coef[2] * chemical[k] + J4$coef[3] * chemical[k]^2 + J4$coef[4] * chemical[k]^3
#   e4[k] <- magnetic[k] - yhat4
# }
# c(mean(e1^2), mean(e2^2), mean(e3^2), mean(e4^2))

## ----eval=FALSE---------------------------------------------------------------
# L2

## ----eval=FALSE---------------------------------------------------------------
# p1 <- 1
# R2_1 <- summary(J1)$r.squared
# adj_R2_1 <- 1 - (1 - R2_1) * (n - 1) / (n - p1 - 1)
# 
# p2 <- 2
# R2_2 <- summary(J2)$r.squared
# adj_R2_2 <- 1 - (1 - R2_2) * (n - 1) / (n - p2 - 1)
# 
# p3 <- 1
# R2_3 <- summary(J3)$r.squared
# adj_R2_3 <- 1 - (1 - R2_3) * (n - 1) / (n - p3 - 1)
# 
# p4 <- 3
# R2_4 <- summary(J4)$r.squared
# adj_R2_4 <- 1 - (1 - R2_4) * (n - 1) / (n - p4 - 1)
# 
# c(adj_R2_1,adj_R2_2,adj_R2_3,adj_R2_4)

## ----eval=FALSE---------------------------------------------------------------
# L4

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())
# attach(chickwts)
# x <- sort(as.vector(weight[feed == "soybean"]))
# y <- sort(as.vector(weight[feed == "linseed"]))
# detach(chickwts)
# cvm_statistic <- function(x, y) {
#   n <- length(x)
#   m <- length(y)
#   combined <- c(sort(x), sort(y))
#   ecdf_x <- cumsum(combined %in% x) / n
#   ecdf_y <- cumsum(combined %in% y) / m
#   D <- sum((ecdf_x - ecdf_y)^2) * (n * m) / (n + m)^2  # 归一化
#   return(D)
# }
# D0 <- cvm_statistic(x,y)
# R <- 999
# z <- c(x,y)
# n1 <- length(x)
# n2 <- length(y)
# D <- numeric(R)
# for (i in 1:R) {
#   set.seed(i)
#   permuted_indices <- sample(length(z))
#   x1 <- z[permuted_indices[1:n1]]
#   x2 <- z[permuted_indices[(n1 + 1):(n1 + n2)]]
#   D[i] <- cvm_statistic(x1, x2)
# }
# 
# p_value <- mean(c(D0, D) >= D0)
# p_value
# 

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())
# set.seed(1234)
# n <- 100
# X_h0 <- rnorm(n)
# Y_h0 <- rnorm(n)
# X_h1 <- rnorm(n)
# Y_h1 <- X_h1 + rnorm(n, mean = 0, sd = 0.5)
# spearman_permutation_test <- function(x, y, R = 999) {
#   original_correlation <- cor(x, y, method = "spearman")
#   perm_correlations <- numeric(R)
#   for (i in 1:R) {
#     set.seed(i)
#     perm_y <- sample(y)
#     perm_correlations[i] <- cor(x, perm_y, method = "spearman")
#   }
#   p_value <- mean(abs(perm_correlations) >= abs(original_correlation))
#   return(list(correlation = original_correlation, p_value = p_value))
# }
# 
# result_h0 <- spearman_permutation_test(X_h0, Y_h0)
# result_h1 <- spearman_permutation_test(X_h1, Y_h1)
# cor_test_h0 <- cor.test(X_h0, Y_h0, method = "spearman")
# cor_test_h1 <- cor.test(X_h1, Y_h1, method = "spearman")
# 
# cat("Under H0:\n")
# cat("Spearman correlation:", result_h0$correlation, "\n")
# cat("permutation p-value:", result_h0$p_value, "\n")
# cat("cor.test p-value:", cor_test_h0$p.value, "\n\n")
# 
# cat("Under H1:\n")
# cat("Spearman correlation:", result_h1$correlation, "\n")
# cat("permutation p-value:", result_h1$p_value, "\n")
# cat("cor.test p-value:", cor_test_h1$p.value, "\n")
# 

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())
# metropolis_hastings <- function(num_samples) {
#   samples <- numeric(num_samples)
#   current <- rnorm(1)
#   for (i in 1:num_samples) {
#     proposed <- rnorm(1, mean = current)
#     acceptance_ratio <- dcauchy(proposed) / dcauchy(current)
#     if (runif(1) < acceptance_ratio) {
#       current <- proposed
#     }
#     samples[i] <- current
#   }
#   return(samples)
# }
# num_samples <- 10000
# set.seed(1234)
# samples <- metropolis_hastings(num_samples)
# samples <- samples[-(1:1000)]
# deciles_generated <- quantile(samples, probs = seq(0.1, 1, by = 0.1))
# deciles_cauchy <- qcauchy(seq(0.1, 1, by = 0.1))
# cat("Deciles of generated samples:\n", deciles_generated, "\n")
# cat("Deciles of standard Cauchy distribution:\n", deciles_cauchy, "\n")
# samples_df <- data.frame(Value = samples)
# ggplot(samples_df, aes(x = Value)) +
#   geom_histogram(aes(y = after_stat(density)), bins = 50, fill = "lightblue", alpha = 0.7) +
#   stat_function(fun = dcauchy, color = "red", size = 1) +
#   labs(title = "Metropolis-Hastings Sampling from Standard Cauchy",
#        x = "Value",
#        y = "Density") +
#   theme_minimal() +
#   theme(legend.position = "topright") +
#   annotate("text", x = 3, y = 0.1, label = "Standard Cauchy PDF", color = "red")

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())
# gibbs_sampler <- function(n, a, b, num_samples) {
#   samples <- data.frame(iteration = 1:num_samples, x = numeric(num_samples), y = numeric(num_samples))
#   x <- 0
#   for (i in 1:num_samples) {
#     y <- rbeta(1, x + a, n - x + b)
#     x <- rbinom(1, n, y)
#     samples[i, "x"] <- x
#     samples[i, "y"] <- y
#   }
#   return(samples)
# }
# n <- 10
# a <- 1
# b <- 1
# num_samples <- 1000
# samples <- gibbs_sampler(n, a, b, num_samples)
# samples_long <- reshape2::melt(samples, id.vars = "iteration")
# ggplot(samples_long, aes(x = iteration, y = value, color = variable, group = variable)) +
#   geom_line(size = 1) +
#   labs(title = "Gibbs Sampler Chains for x and y",
#        x = "Iteration",
#        y = "Value") +
#   scale_color_manual(values = c("x" = "lightblue", "y" = "lightgreen"), labels = c("x Chain", "y Chain")) +
#   theme_minimal() +
#   theme(legend.title = element_blank())

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())

## ----eval=FALSE---------------------------------------------------------------
# metropolis_hastings <- function(num_samples) {
#   samples <- numeric(num_samples)
#   current <- rnorm(1)
#   for (i in 1:num_samples) {
#     proposed <- rnorm(1, mean = current)
#     acceptance_ratio <- dcauchy(proposed) / dcauchy(current)
#     if (runif(1) < acceptance_ratio) {
#       current <- proposed
#     }
#     samples[i] <- current
#   }
#   return(samples)
# }
# num_samples <- 10000
# set.seed(1234)
# chains <- lapply(1:3, function(i) metropolis_hastings(num_samples)[-c(1:1000)])
# chains_mcmc <- mcmc.list(lapply(chains, mcmc))
# gelman_diag_mh <- gelman.diag(chains_mcmc, autoburnin = FALSE)
# print("Gelman-Rubin Diagnostic:")
# print(gelman_diag_mh)

## ----eval=FALSE---------------------------------------------------------------
# gibbs_sampler <- function(n, a, b, num_samples) {
#   samples <- data.frame(x = numeric(num_samples), y = numeric(num_samples))
#   x <- 0
#   for (i in 1:num_samples) {
#     y <- rbeta(1, x + a, n - x + b)
#     x <- rbinom(1, n, y)
#     samples[i, "x"] <- x
#     samples[i, "y"] <- y
#   }
#   return(samples)
# }
# n <- 10
# a <- 1
# b <- 1
# num_samples <- 1000
# set.seed(1234)
# chains_gibbs <- lapply(1:3, function(i) gibbs_sampler(n, a, b, num_samples))
# chains_x <- lapply(chains_gibbs, function(chain) mcmc(chain$x))
# chains_y <- lapply(chains_gibbs, function(chain) mcmc(chain$y))
# chains_x_mcmc <- mcmc.list(chains_x)
# chains_y_mcmc <- mcmc.list(chains_y)
# gelman_diag_gibbs_x <- gelman.diag(chains_x_mcmc, autoburnin = FALSE)
# gelman_diag_gibbs_y <- gelman.diag(chains_y_mcmc, autoburnin = FALSE)
# print("Gelman-Rubin Diagnostic for X:")
# print(gelman_diag_gibbs_x)
# print("Gelman-Rubin Diagnostic for Y:")
# print(gelman_diag_gibbs_y)

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())
# compute_k <- function(k, a, d) {
#   norm_a <- sqrt(sum(a^2))
#   part1 <- ((-1)^k / (factorial(k) * 2^k)) * (norm_a^(2 * k + 2)) / ((2 * k + 1) * (2 * k + 2))
#   part2 <- gamma((d + 1) / 2) * gamma(k + 3/2) / gamma(k + d / 2 + 1)
#   result <- part1 * part2
#   return(result)
# }

## ----eval=FALSE---------------------------------------------------------------
# compute_sum <- function(a, d, max = 150) {
#   s <- 0
#   for (k in 0:max) {
#     s <- s + compute_k(k, a, d)
#   }
#   return(s)
# }

## ----eval=FALSE---------------------------------------------------------------
# a <- c(1, 2)
# d <- 2
# result <- compute_sum(a, d)
# print(result)

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())
# intersection <- function(k) {
#   curve_1 <- function(a) 1 - pt( sqrt((a^2 * (k - 1)) / (k - a^2)) , df = k - 1 )
#   curve_2 <- function(a) 1 - pt( sqrt((a^2 * k) / (k + 1 - a^2)) , df = k )
#   intersection <- uniroot(function(x) curve_1(x) - curve_2(x), lower = 1, upper = 2)
#   return(intersection$root)
# }
# k_values <- c(4:25, 100, 500, 1000)
# points <- numeric(length(k_values))
# for(i in 1:length(k_values)){
#   points[i] <- intersection(k_values[i])
# }
# solve_equation <- function(k) {
#   result <- tryCatch({
#     uniroot(equation_diff, lower = 1, upper = sqrt(k) - 1e-6, k = k)
#   }, error = function(e) {
#     return(NULL)
#   })
# 
#   if (is.null(result)) {
#     return(NA)
#   }
# 
#   return(result$root)
# }
# names(points) <- paste0("A(", k_values, ")")
# print(points)

## ----eval=FALSE---------------------------------------------------------------
# integrate_part <- function(c, k) {
#   integral <- integral(function(u) (1 + (u^2) / k)^(-(k+1)/2), 0, c)
#   return(integral)
# }
# c_k <- function(k,a){
#   result <- sqrt( a^2 * k /( k + 1 - a^2 ) )
#   return(result)
# }
# eq <- function(k,a){
#   c <- c_k(k,a)
#   result <- 2 * gamma((k+1)/2) / ( sqrt( pi * k ) * gamma( k / 2 ) ) * integrate_part(c,k)
#   return(result)
# }
# intersection_1 <- function(k){
#   lhs <- function(a) eq(k-1,a)
#   rhs <- function(a) eq(k,a)
#   intersection <- uniroot(function(x) lhs(x) - rhs(x), lower = 1+1e-1, upper = 1.8)
#   return(intersection$root)
# }
# k_values_1 <- c(4:25,100)
# points_1 <- numeric(length(k_values_1))
# for(i in 1:length(k_values_1)){
#   points_1[i] <- intersection_1(k_values_1[i])
# }
# names(points_1) <- paste0("A(", k_values_1, ")")
# print(points_1)

## ----eval=FALSE---------------------------------------------------------------
# error <- points_1 - points[1:length(points_1)]
# print(error)

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())
# T <- c(0.54, 0.48, 0.33, 0.43, 1.00, 1.00, 0.91, 1.00, 0.21, 0.85)
# tau <- 1
# n <- length(T)
# obs_indices <- which(T < tau)
# missing_indices <- which(T == tau)
# lambda <- 1
# error <- 1e-6
# max_iter <- 1000
# diff <- 1  # Set an initial difference
# iter <- 0  # Set iteration counter
# while (diff > error && iter < max_iter) {
#   iter <- iter + 1
#   lambda_old <- lambda
#   ET_missing <- tau + 1 / lambda
#   lambda <- n / (sum(T[obs_indices]) + length(missing_indices) * ET_missing)
#   diff <- abs(lambda - lambda_old)
# }
# cat("EM algorithm estimation:", lambda, "\n")
# lambda_mle <- 1 / mean(T[obs_indices])
# cat("MLE estimation:", lambda_mle, "\n")
# 

## ----eval=FALSE---------------------------------------------------------------
# formulas <- list(
#   mpg ~ disp,
#   mpg ~ I(1 / disp),
#   mpg ~ disp + wt,
#   mpg ~ I(1 / disp) + wt
# )

## ----eval=FALSE---------------------------------------------------------------
# bootstraps <- lapply(1:10, function(i) {
#   rows <- sample(1:nrow(mtcars), rep = TRUE)
#   mtcars[rows, ]
# })

## ----eval=FALSE---------------------------------------------------------------
# rsq <- function(mod) summary(mod)$r.squared

## ----eval=FALSE---------------------------------------------------------------
# trials <- replicate(
# 100,
# t.test(rpois(10, 10), rpois(7, 10)),
# simplify = FALSE
# )

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())
# objective <- c(4, 2, 9)
# constraints <- matrix(c(2, 1, 1, 1, -1, 3), nrow = 2, byrow = TRUE)
# direction <- c("<=", "<=")
# rhs <- c(2, 3)
# result <- lp("min", objective, constraints, direction, rhs)
# result$solution
# result$objval
# 

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())
# formulas <- list(
#   mpg ~ disp,
#   mpg ~ I(1 / disp),
#   mpg ~ disp + wt,
#   mpg ~ I(1 / disp) + wt
# )
# for (f in formulas) {
#   print(summary(lm(f, data = mtcars)))
# }
# lapply(formulas, function(f) summary(lm(f, data = mtcars)))

## ----eval=FALSE---------------------------------------------------------------
# bootstraps <- lapply(1:10, function(i) {
#   rows <- sample(1:nrow(mtcars), rep = TRUE)
#   mtcars[rows, ]
# })
# for (bootstrap in bootstraps) {
#   print(summary(lm(mpg ~ disp, data = bootstrap)))
# }
# lapply(bootstraps, function(bootstrap) summary(lm(mpg ~ disp, data = bootstrap)))
# 

## ----eval=FALSE---------------------------------------------------------------
# rsq <- function(mod) summary(mod)$r.squared
# for (f in formulas) {
#   mod <- lm(f, data = mtcars)
#   print(rsq(mod))
# }
# print(lapply(formulas, function(f) rsq(lm(f, data = mtcars))))
# 
# 
# for (bootstrap in bootstraps) {
#   mod <- lm(mpg ~ disp, data = bootstrap)
#   print(rsq(mod))
# }
# print(lapply(bootstraps, function(bootstrap) rsq(lm(mpg ~ disp, data = bootstrap))))

## ----eval=FALSE---------------------------------------------------------------
# rm(list = ls())
# trials <- replicate(
#   100,
#   t.test(rpois(10, 10), rpois(7, 10)),
#   simplify = FALSE
# )
# p_values <- sapply(trials, function(x) x$p.value)
# p_values
# 
# p_values <- sapply(trials,`[[`, "p.value")
# p_values

## ----eval=FALSE---------------------------------------------------------------
# rm(list = ls())
# parallel_lapply <- function(X, FUN, ...) {
#   result <- Map(FUN, X, ...)
#   vapply(result, FUN = identity, FUN.VALUE = numeric(1))
# }
# inputs <- list(1:5, 6:10)
# parallel_lapply(inputs, sum)
# 

## ----eval=FALSE---------------------------------------------------------------
# rm(list = ls())
# better_chisq <- function(x, y) {
#   observed <- table(x, y)
#   chisq_stat <- sum((observed - mean(observed))^2 / mean(observed))
#   return(chisq_stat)
# }
# x <- c(1, 2, 1, 2, 1)
# y <- c(1, 1, 2, 2, 2)
# better_chisq(x, y)

## ----eval=FALSE---------------------------------------------------------------
# rm(list = ls())
# better_table <- function(x, y) {
#   table_result <- table(x, y)
#   return(table_result)
# }
# x <- c(1, 2, 1, 2, 1)
# y <- c(1, 1, 2, 2, 2)
# better_table(x, y)
# observed <- better_table(x, y)
# chisq_stat <- sum((observed - mean(observed))^2 / mean(observed))
# chisq_stat
# 

## ----eval=FALSE---------------------------------------------------------------
# rm(list=ls())

## ----eval=FALSE---------------------------------------------------------------
# cppFunction('
#   DataFrame gibbs_sampler_C(int n, double a, double b, int num_samples) {
#   NumericVector x(num_samples);
#   NumericVector y(num_samples);
# 
#   int current_x = 0;
# 
#   for (int i = 0; i < num_samples; i++) {
#     y[i] = R::rbeta(current_x + a, n - current_x + b);
#     current_x = R::rbinom(n, y[i]);
#     x[i] = current_x;
#   }
#   return DataFrame::create(Named("iteration") = seq(1, num_samples),
#                            Named("x") = x,
#                            Named("y") = y);
# }
# ')

## ----eval=FALSE---------------------------------------------------------------
# 
# gibbs_sampler_R <- function(n, a, b, num_samples) {
#   samples <- data.frame(iteration = 1:num_samples, x = numeric(num_samples), y = numeric(num_samples))
#   x <- 0
#   for (i in 1:num_samples) {
#     y <- rbeta(1, x + a, n - x + b)
#     x <- rbinom(1, n, y)
#     samples[i, "x"] <- x
#     samples[i, "y"] <- y
#   }
#   return(samples)
# }
# 
# #library(Rcpp)
# #dir_cpp <- '../Rcpp/'
# #sourceCpp(paste0(dir_cpp,"Gibbs_sampler_C.cpp"))
# 
# n <- 10
# a <- 1
# b <- 1
# num_samples <- 1000
# 
# set.seed(1234)
# samples_R <- gibbs_sampler_R(n, a, b, num_samples)
# samples_C <- gibbs_sampler_C(n, a, b, num_samples)
# samples_long_R <- reshape2::melt(samples_R, id.vars = "iteration")
# samples_long_C <- reshape2::melt(samples_C, id.vars = "iteration")
# 
# par(mfrow = c(1, 2))
# qqplot(samples_R$x, samples_C$x, main = "QQ Plot of x", xlab = "R", ylab = "C", pch = 8, col = "red", cex = 0.8)
# abline(0, 1, col = "blue", lwd = 1.8)
# qqplot(samples_R$y, samples_C$y, main = "QQ Plot of y", xlab = "R", ylab = "C", pch = 16, col = "red", cex = 0.1)
# abline(0, 1, col = "blue", lwd = 1.8)
# 

## ----eval=FALSE---------------------------------------------------------------
# ts <- microbenchmark(
#   gibbsR = gibbs_sampler_R(n, a, b, num_samples),
#   gibbsC = gibbs_sampler_C(n, a, b, num_samples)
# )
# summary(ts)[, c(1, 3, 5, 6)]

