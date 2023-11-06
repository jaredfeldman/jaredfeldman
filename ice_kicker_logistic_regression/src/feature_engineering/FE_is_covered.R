# To take into account whether a stadium is indoor or outdoor,
# we created the variable is_covered, that will equal "Yes" if the
# roof variable is "closed" or "dome", and will equal "No" if the roof
# variable is "open" or "outdoors"

all_fg <- all_fg %>%
  mutate(is_covered = case_when(
    roof %in% c("closed", "dome") ~ "Yes",
    roof %in% c("open", "outdoors") ~ "No",
    TRUE ~ NA_character_
  ))
