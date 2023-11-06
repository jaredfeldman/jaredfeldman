# Many of the wind values in the given dataset are not reliable or are NA.
# In order to attempt to address this, the following code
# will use regular expressions to extract the wind from the provided
# weather column, and then replace NAs with this value

# extract wind from weather column
all_fg$wind_extracted <- str_extract(all_fg$weather,
                                     "\\d{1,2}(?=\\s*(mph|MPH mph|mph mph))")

# convert to integer
all_fg$wind_extracted <- as.integer(all_fg$wind_extracted)

# fill temp NAs with extracted wind
all_fg <- all_fg %>%
  mutate(wind = ifelse(is.na(wind) & !is.na(wind_extracted),
                       wind_extracted, wind))

# Wind column to integer
all_fg$wind <- as.integer(all_fg$wind)

# create binary variable when wind above 10 mph
all_fg <- all_fg %>%
  mutate(high_wind = ifelse(wind > 10, 1, 0))

# Input values in wind column with NAs in domes using the average
avg_dome_wind = mean(all_fg[all_fg$roof == 'dome', ]$wind, na.rm = TRUE)

all_fg$wind <- ifelse(all_fg$roof == "dome" & is.na(all_fg$wind), avg_dome_wind, all_fg$wind)

# Input values in wind column with NAs in closed stadiums using the average
avg_closed_wind = mean(all_fg[all_fg$roof == 'closed', ]$wind, na.rm = TRUE)

all_fg$wind <- ifelse(all_fg$roof == "closed" & is.na(all_fg$wind), avg_closed_wind, all_fg$wind)

# Cleaning 0 values in wind column in outdoor stadiums when weather col is empty
all_fg$wind <- as.integer(all_fg$wind)
all_fg$wind[all_fg$roof == "outdoors" & all_fg$wind == 0 & is.na(all_fg$weather)]  <- NA

# Cleaning wind outliers
all_fg$wind <- as.integer(all_fg$wind)
all_fg$wind <- ifelse(all_fg$wind >= 35, NA_character_, all_fg$wind)
all_fg$wind <- as.integer(all_fg$wind)
