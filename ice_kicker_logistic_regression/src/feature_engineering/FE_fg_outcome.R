# The nflverse dataset has a field_goal_result variable that is either
# 'made', 'missed', or 'blocked'. This new variable, fg_outcome, will
# be binary to indicate whether a field goal was made or not, regardless
# of whether it was not made due to a miss or a block

# Create binary variable from field_goal_result
all_fg <- all_fg %>%
  mutate(fg_outcome = ifelse(field_goal_result == 'made',1, 0))
