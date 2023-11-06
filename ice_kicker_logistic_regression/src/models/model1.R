#create model 1 based on confirmatory dataset

model1_confirmatory <- lm(data = confirmation_df, fg_outcome ~ iced +
                            kick_distance + I(kick_distance^2) +
                            wind + temp + is_covered + wind*is_covered + temp*is_covered)
