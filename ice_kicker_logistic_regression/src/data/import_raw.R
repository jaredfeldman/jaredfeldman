# load data from nflverse
raw <- nflreadr::load_pbp(2001:2022)

# create unique identifier column
raw$ID_raw <- 1:nrow(raw)

# write .csv - this is a large file, keep local for reference, if needed
write.csv(raw, '../data/raw/raw_pbp.csv')
