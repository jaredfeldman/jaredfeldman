# create a chart that plots the FG% for iced kicks vs. non-iced
# kicks across the entire dataset, 2001 - 2022
iced_vs_not_iced_make_plot <- ggplot(all_fg) +
  aes(x = factor(iced, labels = c("Not Iced", "Iced")), fill = factor(fg_outcome)) +
  geom_bar(position = 'fill') +
  xlab('Iced') +
  ylab('Proportion of Field Goals') +
  ggtitle('Figure 1: Result of FGs (2001-2022)') +
  scale_fill_discrete(name="Field Goal Result", labels=c("Miss", "Make")) +
  theme(text = element_text(size = 7), legend.position = "bottom")
