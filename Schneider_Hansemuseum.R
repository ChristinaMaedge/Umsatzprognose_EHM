### Include ANN-script file ####
source("c:/users/sschneid/documents/fh kiel/module/business analytics/r/knn/ANNUtils.r")

### Load data ####
library(readxl)
filename <- "c:/users/sschneid/documents/projekte/kultur/Beckerbilletdaten_2015_2020.xlsx"
rawdata <- read_excel(filename)
billets <- sapply(c(2:ncol(rawdata)-1), function(x) {sprintf("billet%d",x)})
colnames(rawdata) <- c("Day",billets)

### Process data ####
rawdata <- rawdata[-nrow(rawdata),]
rawdata <- rawdata[-nrow(rawdata),]

# Uniform date column
rawdata$Day <- sapply(1:nrow(rawdata), function(i) {ifelse(is.na(rawdata$Day[i]), rawdata$Day[i-1], substr(rawdata$Day[i],1,5))})
rawdata$Day <- substr(rawdata$Day,1,5)  

# Create complete date column
startyear <- 2015
rawdata$Day[1] <- paste(rawdata$Day[1],".",startyear, sep = "") 
rawdata$Day[2] <- paste(rawdata$Day[2],".",startyear, sep = "") 

seq1 <- seq(3, nrow(rawdata)-1, 2)
seq2 <- seq(4, nrow(rawdata), 2)

complete_date <- function(seq, column, startyear = 2015) {
  dates <- c()
  year <- startyear
  for (i in seq) {
    m_i <- as.integer(substr(column[i],4,5))
    m_i_prior <- as.integer(substr(column[i-2],4,5))
    if (m_i < m_i_prior) year <- year + 1
    column[i] <- paste(column[i],".",year, sep = "")
    dates <- c(dates, column[i])
  }
  # dates <- sapply(seq, function(i) {
  #   m_i <- as.integer(substr(column[i],4,5))
  #   m_i_prior <- as.integer(substr(column[i-2],4,5))
  #   if (m_i < m_i_prior) year <- year + 1
  #   column[i] <- paste(column[i],".",year, sep = "") 
  # })
  return(dates)
}

rawdata$Day[seq1] <- complete_date(seq1, rawdata$Day)
rawdata$Day[seq2] <- complete_date(seq2, rawdata$Day)
rawdata$Day <- as.Date(rawdata$Day, format = "%d.%m.%Y", tz = "GMT")

### Build sales amount times series ####
seq1 <- seq(1, nrow(rawdata)-1, 2)
ts_salesamount <- data.frame(Day   = rawdata$Day[seq1],
                             Value = sapply(seq1, function(i) {sum(rawdata[i,-1])}))

### Build sales revenue times series ####
seq2 <- seq(2, nrow(rawdata), 2)
ts_salesrevenue <- data.frame(Day   = rawdata$Day[seq2],
                              Value = sapply(seq2, function(i) {sum(rawdata[i,-1])}))

### Plot times series data ####
library(ggplot2)
plotdata <- function(data, title, x, y, xlab, ylab, color) {
  ggplot(data = data, aes(x = x, y = y)) +
    labs(title = title) +
    xlab(xlab) + ylab(ylab) +
    geom_line(color = color) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.border = element_blank(),
          axis.line = element_line(colour = "black"))
}

plotdata(ts_salesamount, "Hansemuseum Sales Amount", x = ts_salesamount$Day, y = ts_salesamount$Value, xlab = "Day", ylab = "Value", color = "darkblue")
plotdata(ts_salesrevenue, "Hansemuseum Sales Revenue", x = ts_salesrevenue$Day, y = ts_salesrevenue$Value, xlab = "Day", ylab = "Value", color = "darkgreen")

### Set time series for analysis ####
df <- ts_salesamount

### ARIMA(p,d,q) ####
# p: number of lags
# d: number of differentations
# q: number of moving white noises

# Use ARIMA for optimal number of lags for training data set
train.start <- as.Date("2015-01-01", tz = "GMT")
train.end   <- as.Date("2019-12-01", tz = "GMT")
df.ar <- subset(df, (Day >= train.start) & (Day <= train.end))

# Build ARIMA model
library(forecast)
ar.model <- auto.arima(df.ar$Value)
ar.model
non_seasonal_ar_order <- ar.model$arma[1] # p
non_seasonal_diff_order <- ar.model$arma[6] #d
non_seasonal_ma_order <- ar.model$arma[2] # q
# seasonal_ar_order <- ar.model$arma[3]
# seasonal_ma_order <- ar.model$arma[4]
# period_of_data <- ar.model$arma[5]
# seasonal_diff_order <- ar.model$arma[7]

# Plot forecast data
arima.fitted <- ar.model$fitted
arima.fitted <- cbind.data.frame(arima.fitted, df.ar)

graphdata <- data.frame(Day = arima.fitted$Day, Value = arima.fitted$Value, Predictions = arima.fitted$arima.fitted)
require(ggplot2)
ggplot() +
  labs(title = "ARIMA Forecasting for Interest Rate") +
  xlab("Day") + ylab("Interest Rate") +
  geom_line(data = graphdata, aes(y = Value, x = Day), color = "blue") +
  geom_line(data = graphdata, aes(y = Predictions, x = Day), color = "red") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(colour = "black"))

### LSTM ####

### Hyperparameter ####
ann.epochs    <- 50
ann.batchsize <- 1
ann.lags <- as.lags(non_seasonal_ar_order)
ann.timesteps <- as.timesteps(ann.lags)

### Transform data into stationary and create lagged dataset ####
# For a univariate time series the resulting data set is automatically in a resampled format for LSTM
tsdiff <- as.data.frame(build.stationary(df$Value, k = ann.lags))
tsdiff <- tsdiff[c(-1:-ann.lags),] # there's no x-value for the first y-observation
dateID <- df$Day[(ann.lags+2):nrow(df)]
tsdiff <- cbind.data.frame(dateID, tsdiff)
colnames(tsdiff) <- c("Day","diff", "diff_lag1","diff_lag2","diff_lag3")

### Split resampled dataset into training and test sets ####
df.train <- subset(tsdiff, (Day >= train.start) & (Day <= train.end))
plotdata(df.train, x = df.train$Day, y = df.train$diff, "Differenced Sales Amount for Training data set", "Day", "Amount", "black")

test.start  <- as.Date("2020-01-01", tz = "GMT")
test.end    <- as.Date("2020-12-01", tz = "GMT")
df.test     <- subset(tsdiff, (Day >= test.start) & (Day <= test.end))
plotdata(df.test, x = df.test$Day, y = df.test$diff, "Differenced Sales Amount for Test data set", "Day", "Amount", "black")

### Store origin y values as well as the prior y value of the first test y value ####
y_test <- df$Value[(df$Day >= test.start) & (df$Day <= test.end)] # origin interest rates
y_prior_first_test <- df$Value[which(df$Day == test.start)-1] # for invert differencing

### Normalize training and test data sets ####
nd <- normalize_data(df.train[,-1], df.test[,-1])

### Features (X) and Y ####
X.train <- nd$train[,-1]
Y.train <- nd$train[,1]

X.test <- nd$test[,-1]
Y.test <- nd$test[,1]

### Build and fit LSTM model ####
lstm <- fit.LSTM(X = X.train, Y = Y.train,
                 timesteps = ann.timesteps,
                 epochs = ann.epochs,
                 batch.size = c(ann.batchsize,ann.batchsize),
                 validation_split = 0.2,
                 k.fold = NULL, k.optimizer = NULL,
                 hidden = data.frame(u=c(3), a=c("sigmoid")),
                 dropout = NULL,
                 output.activation = "linear",
                 stateful = T,
                 return_sequences = T,
                 loss = "mean_squared_error",
                 optimizer = optimizer_adam(lr = 0.02, decay = 1e-6),
                 metrics = c('accuracy'))

### Forecast & quality ####
# Predict on test data
lstm.X.test <- as.LSTM.X(X.test, ann.timesteps)
predictiondiff <- lstm$model %>% predict(lstm.X.test, batch_size = ann.batchsize)
predictiondiff <- denormalize(predictiondiff[,1], nd$min[1], nd$max[1])
# invert differencing
testseries <- c(y_prior_first_test, y_test)
testseries <- testseries[-NROW(testseries)]
predictions <- invert_differencing(predictiondiff, testseries)
# RMSE
rmse(testseries, predictions)

### Visualize test and forecast values ####
graphdata <- data.frame(Day = df.test$Day, Value = testseries, Predictions = predictions)
require(ggplot2)
ggplot() +
  labs(title = "LSTM Forecasting for Sales Amount (Hansemuseum)") +
  xlab("Day") + ylab("Sales Amount") +
  geom_line(data = graphdata, aes(y = Value, x = Day), color = "darkblue") +
  geom_line(data = graphdata, aes(y = Predictions, x = Day), color = "red") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(colour = "black"))
