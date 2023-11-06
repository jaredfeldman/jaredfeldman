# There are 8 different values for the given surface variable,
# some of which seem equivalent.
# For our analysis, we only care about whether the surface is
# natural or synthetic
# To do this, we created surface_2 variable.

all_fg <- all_fg %>%
  mutate(surface_2 = case_when(
    surface %in% c("a_turf", "astroturf", "matrixturf",
                   "sportturf", "fieldturf", "astroplay") ~ "synthetic",
    surface %in% c("dessograss", "grass") ~ "natural",
    TRUE ~ NA_character_
  ))
