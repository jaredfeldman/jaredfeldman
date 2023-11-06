##

model2_confirmatory <- lm(data = confirmation_df, fg_outcome ~ iced +
                            kick_distance + I(kick_distance^2) +
                            wind + temp + is_covered +
                            wind*is_covered + temp*is_covered +
                            score_differential + high_pressure +
                            surface_2 +
                            accuracy_5 + pro_kicker)
