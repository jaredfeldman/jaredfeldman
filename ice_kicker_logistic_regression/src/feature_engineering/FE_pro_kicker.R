# In order to account for quality of a kicker, we use the following to
# identify field goals done by pro bowl/all pro kickers
# (in any season prior to the kick takeing place)

# create a dataframe with pro bowl data
pro <- data.frame(
  # All seasons dating back to 1998, 3 years before the first year in our
  # data
  season = c(2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012,
             2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001,
             2000, 1999, 1998),
  # Pro Bowl/All Pro players grouped by year
  players = c("J.Tucker M.Gay D.Carlson", "J.Tucker M.Gay J.Elliott",
              "J.Sanders J.Tucker", "J.Tucker W.Lutz J.Lambo",
              "A.Rosas J.Myers W.Lutz",
              "G.Zuerlein G.Gano C.Boswell M.Prater R.Gould",
              "J.Tucker M.Bryant M.Prater", "S.Gostkowski D.Bailey J.Brown",
              "A.Vinatieri S.Gostkowski C.Parkey M.Bryant",
              "J.Tucker M.Prater S.Gostkowski",
              "B.Walsh P.Dawson", "D.Akers S.Janikowski",
              "B.Cundiff D.Akers R.Bironas",
              "N.Kaeding D.Akers D.Carpenter S.Janikowski R.Bironas",
              "S.Gostkowski J.Carney", "R.Bironas N.Folk K.Brown",
              "R.Gould N.Kaeding M.Stover", "N.Rackers S.Graham",
              "A.Vinatieri D.Akers", "M.Vanderjagt J.Wilkins",
              "A.Vinatieri D.Akers", "D.Akers J.Elam",
              "M.Stover M.Gramatica", "O.Mare J.Hanson M.Vanderjagt",
              "G.Anderson J.Elam")
)

# Create new binary variable equals if kicker pro bowl/all pro
# in the years before kick
all_fg$pro_kicker <- 0

pro <- pro %>%
  separate_rows(players, sep = " ")

for (i in 2001:2023) {

  year_df <- all_fg[all_fg$season == i, ]

  unique_year_kickers <- unique(year_df$kicker_player_name)

  for (j in unique_year_kickers) {

    temp_pro <- pro[pro$season <= i-1, ]

    all_fg[all_fg$season == i & all_fg$kicker_player_name == j, ]$pro_kicker <- ifelse(
      j %in% temp_pro$players, 1, 0)

  }
}

# drop dataframes no longer needed after creation
rm(pro)
rm(temp_pro)
rm(year_df)
