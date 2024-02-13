# Chess Banter: Does engaging with an opponent in online chess impact their performance? - Technology Used
R (tidyverse, caret, MatchIt, zoo, plotfunctions, writexl, ggthemes, stargazer, lmtest), Python, Selenium, OpenAI API, lichess.org API, RStudio, .Rmd, Git, GitHub

# Chess Banter: Does engaging with an opponent in online chess impact their performance? - Overview

## Abstract
Bullying in online games is a serious problem and online chess is no exception. The purpose of trashtalking is to try to decrease opponent performance by distracting opponents into heightened emotions andless logical decision making. However, these tactics may actually backfire. In our study, opponents wererandomly selected by the lichess.org matching algorithm and we randomly assigned these opponents to acontrol, placebo, or treatment group at time of matching. For those in the treatment group, they receivednon-harassing standardized comments about the inaccuracies in their play. In our follow-up experiment, weimplemented the OpenAI API so that we could use ChatGPT 4.0 to engage with players more dynamicallyto more directly measure any effects of a conversation. Similar to our first experiment, we randomly assignedopponents into control and ChatGPT. Our analysis for both experiments did not produce any significanteffects across multiple outcomes. Since chatting with an opponent in online gaming, particularly in chess,has mixed reviews from players, this lack of effect may be useful information to players who are opposed tochatting, knowing that it was not found to have any adverse effects on opponents.

# Report Reproduction Steps
1. Clone the repository
2. Open the `analysis/chess_report_master.RMD` file
3. Knit the report
  - Further detail around the bot used can be found in `/chess_bot`
  - The code used to run the bot and play chess can be found in `/chess_bot/maia-playerv3.ipynb`
