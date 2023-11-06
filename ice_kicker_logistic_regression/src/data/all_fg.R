# select only field goal observations and create corresponding `all_fg`
# containing variables of interest from raw

all_fg <- raw %>%
  select(game_id, desc, play_type, field_goal_attempt, iced, field_goal_result,
         half_seconds_remaining, game_half, defteam_timeouts_remaining,
         score_differential, kick_distance, kicker_player_name,
         weather, wind, temp, stadium, surface, roof, season, ID_raw) %>%
  filter(field_goal_attempt == 1)

write.csv(all_fg, '../data/interim/all_fg.csv')
