# Identify iced kicks and update the raw dataset

raw <- raw %>%
  mutate(iced = case_when(

    # timeout called by defensive team prior to a field goal
    field_goal_attempt == 1 & lag(timeout) == 1 & lag(play_type) == "no_play"
    & posteam != lag(timeout_team) ~ 1,

    # else zero
    TRUE ~ 0
  ))
