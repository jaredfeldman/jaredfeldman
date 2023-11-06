# In order to include an additional factor of a field goal that may be
# higher pressure than others, we create a variable called high_pressure
# that is defined as a field goal that is kicked with less than or equal to
# 2 minutes remaining in the half OR is kicked at any time during overtime

# create binary variable to identify high_pressure field goals
all_fg <- all_fg %>%
  mutate(high_pressure = ifelse((half_seconds_remaining < 120
                                 | game_half == "Overtime"), 1, 0))
