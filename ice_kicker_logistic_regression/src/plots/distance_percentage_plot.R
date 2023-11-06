# create a chart that plots the kick distance vs. FG% for iced kicks
# vs. non-iced kicks across the entire dataset, 2001 - 2022
distance_percentage_plot <- all_fg %>%
  group_by(kick_distance, iced) %>%
  summarize(mean_fg=mean(fg_outcome)) %>%
  ggplot() +
  aes(y=mean_fg, x=kick_distance, color=factor(iced)) +
  ylim(c(0,1)) +
  xlim(c(15,75)) +
  xlab("Kick Distance (Yards)") +
  ylab("Field Goal Percentage (%)") +
  ggtitle('Figure 2: Kick Distance vs. FG % (2001 - 2022)') +
  geom_jitter() +
  stat_smooth() +
  scale_color_manual(values = c("navy blue", "maroon"), labels = c("Not Iced", "Iced")) +
  labs(color = "Iced/Not Iced") +
  theme(text = element_text(size = 7), legend.position = "bottom")
