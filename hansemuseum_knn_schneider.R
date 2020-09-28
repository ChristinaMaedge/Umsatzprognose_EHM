### Clear environment ####
rm(list=ls())

### Load libraries ####
library(readxl)
library(ggplot2)
library(keras)
library(deepANN)
install_keras()
library(tensorflow)

### Load data ####
filename <- "Daten/Datensatz_KNN.xlsx"
rawdata <- read_excel(filename)
rawdata$Datum <- as.Date(rawdata$Datum, format = "%Y-%m-%d", tz = "UTC")
rawdata$Umsatz <- as.numeric(rawdata$Umsatz)
rawdata$Tickets <- as.integer(rawdata$Tickets)
rawdata$ChristiHimmelfahrt <- as.integer(rawdata$ChristiHimmelfahrt)
rawdata$ChristiHimmelfahrt_ext <- as.integer(rawdata$ChristiHimmelfahrt_ext)
rawdata$Feiertag <- as.integer(rawdata$Feiertag)
rawdata$Jahreszeit <- factor(rawdata$Jahreszeit, levels = c("Winter", "Fruehling", "Sommer", "Herbst"))
rawdata$KielerWoche <- as.integer(rawdata$KielerWoche)
rawdata$Ostern <- as.integer(rawdata$Ostern)
rawdata$Pfingsten <- as.integer(rawdata$Pfingsten)
rawdata$Silvester <- as.integer(rawdata$Silvester)
rawdata$Silvester_ext <- as.integer(rawdata$Silvester_ext)
rawdata$SommerferienBaW <- as.integer(rawdata$SommerferienBaW)
rawdata$SommerferienBY <- as.integer(rawdata$SommerferienBY)
rawdata$SommerferienHE <- as.integer(rawdata$SommerferienHE)
rawdata$SommerferienNDS <- as.integer(rawdata$SommerferienNDS)
rawdata$SommerferienNRW <- as.integer(rawdata$SommerferienNRW)
rawdata$SommerferienSH <- as.integer(rawdata$SommerferienSH)
rawdata$TDE <- as.integer(rawdata$TDE)
rawdata$Weihnachten <- as.integer(rawdata$Weihnachten)
rawdata$Tmax <- as.numeric(rawdata$Tmax)
rawdata$Tmittel <- as.numeric(rawdata$Tmittel)
rawdata$Tmin <- as.numeric(rawdata$Tmin)
rawdata$Tminboden <- as.numeric(rawdata$Tminboden)
rawdata$Windmax_BF <- as.numeric(rawdata$Windmax_BF)
rawdata$Niederschlag <- as.numeric(rawdata$Niederschlag)
rawdata$Sonnenstunden <- as.numeric(rawdata$Sonnenstunden)
rawdata$Wochenende <- as.integer(rawdata$Wochenende)
names(rawdata)[which(names(rawdata) == "Wochentag_c")] <- "Wochentag"
rawdata$Wochentag <- factor(rawdata$Wochentag, levels = c("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"))
names(rawdata)[which(names(rawdata) == "Monat_c")] <- "Monat"
rawdata$Monat <- factor(rawdata$Monat, levels = c("Januar", "Februar", "Maerz", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"))

### Check missing values ####
library(imputeTS)
sapply(rawdata, function(x) sum(is.na(x)))
rawdata$Windmax_BF <- na_interpolation(rawdata$Windmax_BF, option = "spline")
rawdata$Niederschlag <- na_interpolation(rawdata$Niederschlag, option = "spline")
sum(is.na(rawdata))

### Dummyfication ####
rawdata <- dummify(rawdata, columns = c("Jahreszeit", "Wochentag", "Monat"), remove_columns = T)

### Visualizations ####
# Umsatz
ggplot(rawdata, aes(x = Datum)) +
  labs(title = "Hansemuseum Lübeck") +
  xlab("Datum") + ylab("Umsatz") +
  geom_line(aes(y = Umsatz), color = "darkblue") +
  scale_x_date(date_labels = "%Y %b %d")

# Tickets
ggplot(rawdata, aes(x = Datum)) +
  labs(title = "Hansemuseum Lübeck") +
  xlab("Datum") + ylab("Tickets") +
  geom_line(aes(y = Tickets), color = "darkgreen") +
  scale_x_date(date_labels = "%Y %b %d")

### Build working data - either "Umsatz" or "Tickets ####
workdata <- rawdata
workdata$Tickets <- NULL; caption <- "Tickets" # use Tickets
#workdata$Umsatz <- NULL; caption <- "Umsatz" # use Umsatz

### Stationary time series ####
differences <- 1
df <- build.stationary(workdata, y = 2, differences = differences)
df <- df[, c(1:2, 47, 3:46)]

ggplot(df, aes(x = Datum)) +
  labs(title = "Hansemuseum Lübeck") +
  xlab("Datum") + ylab(paste0(caption, "-Differenzen")) +
  geom_line(aes(y = differences), color = "darkred") +
  scale_x_date(date_labels = "%Y %b %d")

### Define train and test time slot & scale data ####
train.start <- as.Date("2016-01-02", tz = "UTC")
train.end   <- as.Date("2019-06-30", tz = "UTC")
test.start  <- as.Date("2019-07-01", tz = "UTC")
test.end    <- as.Date("2019-12-31", tz = "UTC")

# For invert differencing it's better to recur on rawdata or workdata
train.rows <- which(workdata$Datum == train.start):which(workdata$Datum == train.end)
test.rows <- which(workdata$Datum == test.start):which(workdata$Datum == test.end)

df.train <- df[(train.rows - differences), ]
df.test  <- df[(test.rows - differences), ]

scale_type <- "minmax"
scaled <- scale.datasets(df.train, df.test, columns = 3L, scale_type)

### LSTM ####

### Hyperparameters ####
ann.epochs     <- 100L
ann.batch_size <- 1L
ann.timesteps  <- 1L
selected_features <- 4:47
selected_outcomes <- 3L
ts_type <- ifelse(is.null(selected_features), "univariate", "multivariate")

### Get train data in LSTM compatible preformat ####
lstmdata <- get.LSTM.XY(scaled$train, x = selected_features, y = selected_outcomes, other_columns = 1:2, timesteps = ann.timesteps)

### Fit LSTM model ####
lstm <- fit.LSTM(X = lstmdata$X, Y = lstmdata$Y,
                 timesteps = ann.timesteps,
                 epochs = ann.epochs,
                 batch_size = c(ann.batch_size, F),
                 validation_split = 0.2,
                 k.fold = NULL, k.optimizer = NULL,
                 hidden = data.frame(c(128L, 64L, 32L, 16L), c("relu")),
                 dropout = NULL,
                 output.activation = "linear",
                 stateful = F,
                 return_sequences = F,
                 loss = "mean_squared_error",
                 optimizer = optimizer_adam(lr = 0.02, decay = 1e-6),
                 metrics = list('mae'),
                 verbose = 1L)

### In-sample predictions with train data ####
X.tensor <- as.LSTM.X(lstmdata$X, ann.timesteps)
predictions <- lstm$model %>% 
  predict.ANN(X.tensor, batch_size = ann.batch_size,
              scale_type = scale_type, scaler = list(scaled$min[1L], scaled$max[1L]),
              diff_type = "simple", timesteps = ann.timesteps, lag = 0L, differences = differences,
              invert_first_row = train.rows[1L], Y.actual = workdata$Tickets, type = ts_type)
# RMSE
actual <- as.LSTM.period_outcome(df.train, p = 1L, y = 2L, timesteps = ann.timesteps, type = ts_type)
rmse(actual$outcome, predictions)

# Visualize results
graphdata <- cbind.data.frame(actual, predictions)
ggplot(data = graphdata, aes(x = period)) +
  labs(title = paste("LSTM In-sample Forecasting for", caption)) +
  xlab("Datum") + ylab(caption) +
  geom_line(aes(y = outcome, colour = "actual")) +
  geom_line(aes(y = predictions, colour = "predicted")) +
  scale_colour_manual(caption, 
                      breaks = c("actual", "predicted"),
                      values = c("darkblue", "red"),
                      labels = c("actual", "predicted")) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position = c(0.85, 0.85),
        #legend.background = element_rect(fill = c('#FFFFCC')),
        legend.title = element_text(color = "black", size = 8),
        legend.text = element_text(color = "black", size = 8))

### Out-of-sample predictions with test data ####
lstmdata <- get.LSTM.XY(scaled$test, x = selected_features, y = selected_outcomes, other_columns = 1:2, timesteps = ann.timesteps)
X.tensor <- as.LSTM.X(lstmdata$X, ann.timesteps)
predictions <- lstm$model %>% 
  predict.ANN(X.tensor, batch_size = ann.batch_size,
              scale_type = scale_type, scaler = list(scaled$min[1L], scaled$max[1L]),
              diff_type = "simple", timesteps = ann.timesteps, lag = 0L, differences = differences,
              invert_first_row = test.rows[1L], Y.actual = workdata$Tickets, type = ts_type)

# RMSE
actual <- as.LSTM.period_outcome(df.test, p = 1L, y = 2L, timesteps = ann.timesteps, type = ts_type)
rmse(actual$outcome, predictions)

# Visualize results
graphdata <- cbind.data.frame(actual, predictions)
ggplot(data = graphdata, aes(x = period)) +
  labs(title = paste("LSTM Out-of-sample Forecasting for", caption)) +
  xlab("Datum") + ylab(caption) +
  geom_line(aes(y = outcome, colour = "actual")) +
  geom_line(aes(y = predictions, colour = "predicted")) +
  scale_colour_manual(caption, 
                      breaks = c("actual", "predicted"),
                      values = c("darkblue", "red"),
                      labels = c("actual", "predicted")) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position = c(0.85, 0.85),
        #legend.background = element_rect(fill = c('#FFFFCC')),
        legend.title = element_text(color = "black", size = 8),
        legend.text = element_text(color = "black", size = 8))
