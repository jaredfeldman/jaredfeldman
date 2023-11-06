# To take into account whether a kicker is on a "hot streak" or not, we create
# the variables accuracy_5 to indicate the success rate
# for a kicker's last 5 attempts.

# create zero-value cols
all_fg$accuracy_5 <- 0

# clean kicker's name
all_fg$kicker_player_name <- gsub(" ", "", all_fg$kicker_player_name)

# sort dataframe by kicker_player_name and ID_raw
all_fg <- all_fg[with(all_fg, order(kicker_player_name, ID_raw)), ]

# Find unique kicker names in the dataframe
unique_kickers <- unique(all_fg$kicker_player_name)

# Loop over each unique kicker
for (kicker_name in unique_kickers) {

  # Find rows where the kicker name matches
  kicker_rows_mask <- all_fg$kicker_player_name == kicker_name

  # Extract the fg_outcome history per kicker
  kicker_fg_history <- all_fg$fg_outcome[kicker_rows_mask]

  # Calculate rolling mean
  temp_accuracy_5 <- rollapply(kicker_fg_history, width = 5, FUN = mean, fill = NA,
                               align = "right")

  # Assign rolling mean to df column
  all_fg$accuracy_5[kicker_rows_mask] <- move_n_point(temp_accuracy_5, n = 1, na_value = NA)
}

# Sort dataframe to original order
all_fg <- all_fg[with(all_fg, order(ID_raw)), ]
