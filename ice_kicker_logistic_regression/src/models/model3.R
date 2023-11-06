##
library(MatchIt)

confirmation_clean <- confirmation_df[!is.na(confirmation_df$temp) ,]
confirmation_clean <- confirmation_clean[!is.na(confirmation_clean$wind) ,]
confirmation_clean <- confirmation_clean[!is.na(confirmation_clean$surface_2) ,]
confirmation_clean <- confirmation_clean[!is.na(confirmation_clean$accuracy_5) ,]

matched_fg_confirmatory <- matchit(data = confirmation_clean, iced ~
                                     kick_distance + I(kick_distance^2) +
                                     score_differential + high_pressure + defteam_timeouts_remaining +
                                     temp + wind + surface_2 +
                                     accuracy_5 + pro_kicker,
                                   method = "nearest", distance = 'glm', ratio = 2, replace = FALSE)


# Run model
matched_data_confirmatory <- match.data(matched_fg_confirmatory)

model3_confirmatory <- lm(data = matched_data_confirmatory , fg_outcome ~ iced +
                            kick_distance + I(kick_distance^2) +
                            wind + temp + is_covered +
                            wind*is_covered + temp*is_covered +
                            score_differential + high_pressure +
                            surface_2 +
                            accuracy_5 + pro_kicker)


