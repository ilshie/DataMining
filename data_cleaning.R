# load in useful packages
library(dplyr)
library(tidyr)

# read in data
indicators <- read.csv("world-development-indicators/Indicators.csv")

# look at counts for each indicator
counts <- indicators %>% 
  group_by(IndicatorName) %>%
  summarise(num_countries = n_distinct(CountryName),
            num_years = n_distinct(Year),
            first_year = min(Year),
            last_year = max(Year)) %>%
  arrange(desc(num_countries, last_year))

# look at the most common year in which data on indicators ended
counts %>% count(last_year) %>% arrange(desc(n))

# subset only 2014 data and cast data frame into wide format
indicators_2014 <- indicators %>% 
  filter(Year == 2014) %>%
  select(CountryName, IndicatorName, Value) %>%
  spread(key = "IndicatorName", value = "Value")

# for each indicator, find the number of countries that have data for that indicator
data_avail <- sort(apply(indicators_2014[,-1], MARGIN=2, 
                         FUN=function(col) sum(!is.na(col))), decreasing=TRUE)

# visualize data availability for 2014 indicators
plot(data_avail, xlab="2014 Indicator", ylab="Number of countries with data", cex.lab=1.5)

# create data frame of top 100 most available indicators
top100_indicators <- data.frame(indicator = names(data_avail[1:100]),
                                num_countries = as.vector(data_avail[1:100]))

# subset top 100 indicators from wide data frame
top100_cols <- match(top100_indicators$indicator[1:100], names(indicators_2014))
indicators_2014_subset <- indicators_2014 %>% select(CountryName, sort(top100_cols))

# visualize data availability for top 100 indicators
data_avail_top100 <- sort(apply(indicators_2014_subset[,-1], MARGIN=2, 
                                FUN=function(col) sum(!is.na(col))), decreasing=TRUE)
plot(data_avail_top100, xlab="2014 Indicator", 
     ylab="Number of countries with data", cex.lab=1.5, ylim=c(0,250))

# write data for top 100 2014 indicators to csv file
write.csv(indicators_2014_subset, "indicators_cleaned.csv", row.names=FALSE)