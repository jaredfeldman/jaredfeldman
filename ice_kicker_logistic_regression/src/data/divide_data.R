# Divide dataset into exploratory and confirmatory sets

set.seed(100)

# Extract 30% of rows where ice equals 1
ice_1_df <- all_fg %>% filter(iced == 1) %>% sample_frac(0.3)

# Extract 30% of rows where ice equals 0
ice_0_df <- all_fg %>% filter(iced == 0) %>% sample_frac(0.3)

# Combine both dataframes
exploratory_df <- rbind(ice_1_df, ice_0_df)

# Combine the remaining rows into a confirmation set
confirmation_df <- all_fg %>% anti_join(ice_1_df) %>% anti_join(ice_0_df)

write.csv(confirmation_df, '../data/processed/confirmation.csv')
write.csv(exploratory_df, '../data/interim/exploratory.csv')
