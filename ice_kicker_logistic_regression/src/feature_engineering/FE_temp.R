# Many of the temp values in the given dataset are not reliable or are NA.
# In order to attempt to address this, the following code
# will use regular expressions to extract the temperature from the provided
# weather column, and then replace NAs with this value

# extract temperature from weather column
all_fg$temp_extracted <- gsub("^.*Temp: *(-?\\d{1,3}).*$", "\\1", all_fg$weather)
all_fg$temp_extracted <- as.integer(all_fg$temp_extracted)

# fill temp NAs with extracted temp
all_fg <- all_fg %>%
  mutate(temp = ifelse(is.na(temp) & !is.na(temp_extracted), temp_extracted, temp))

# Lastly, create binary variable when temperature is below 32 F
# in case needed for analysis
all_fg <- all_fg %>%
  mutate(cold_temp = ifelse(temp < 32, 1, 0))

# Input temp of dome/closed stadiums to average
avg_dome_temp = mean(all_fg[all_fg$roof == 'dome', ]$temp, na.rm = TRUE)
all_fg$temp <- ifelse(all_fg$roof == "dome" & is.na(all_fg$temp), avg_dome_temp, all_fg$temp)

avg_closed_temp = mean(all_fg[all_fg$roof == 'closed', ]$temp, na.rm = TRUE)
all_fg$temp <- ifelse(all_fg$roof == "closed" & is.na(all_fg$temp), avg_closed_temp, all_fg$temp)

# Convert temp column to integer
all_fg$temp <- as.integer(all_fg$temp)
