### lab_2_team_aaj

# Evaluating the practice of “icing” kickers in the NFL using play-by-play data from 2001 to 2022

### Introduction
*Icing the kicker is a football strategy whereby the defense team calls a timeout prior to a field goal kick with the aim of forcing a mistake. Icing the kicker is an extended practice and is also a highly-debated topic among coaches, fans, and commentators. The strategy is often deployed in high-pressure situations at the end of games, where a field goal can lead to victory or defeat.*

*Given the increasingly popular usage of analytics in sports, the goal of the study is to use a data-driven approach to understand if icing the kicker has an impact on the make rate of field goals. To accomplish this, we leverage NFL data and apply a set of regression models to estimate the potential impact when controlling for confounders. Hence, this analysis aims to address the following research question:*

#### *Does icing affect the probability of a kicker making or missing a field goal?*


# Report Reproduction Steps
1. Clone the repository
2. Open the `lab-2-lab-2-aaj.Rproj` file
3. Open the `/reports/lab_2_report_aaj.Rmd` file
4. Knit the report (the initial `get data` chunk may take several minutes to run)
  - Further detail around models can be found in `/notebooks/aaj_project2_models`
  - Further detail around charts can be found in `/notebooks/aaj_project2_charts`
  - Further detail around EDA and feature engineering can be found in `notebooks/aaj_project2_EDA`

# Project Organization

We have used the following folder structure and can be used for reference. 
This is based on [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science).


    ├── LICENSE
    ├── README.md                 <- The top-level README describing the project aims
    ├── data
    │   ├── raw                   <- This will contain the raw dataset from nflverse
    │   ├── interim               <- This contains all FG plays and associated variables, as well as our exploratory dataset (30% of total)
    │   ├── processed             <- This contains our final dataset (70% of total) used for our 3 models
    │   └── homework              <- This contains our dataset for HW12
    ├── notebooks                 <- .Rmd notebooks
    ├── references                <- This contains a table describing our feature-engineered variables. All other information about nflverse can be found [here](https://www.nflfastr.com/)
    ├── reports                   <- This contains our final report in .Rmd and .pdf formats
    └── src                       <- Source code for use in this project.
       ├── data                   <- This contains the scripts used to generate initial dataframe and create FG dataframe
       ├── models                 <- This contains the scripts to run each of our 3 models and build the stargazer table
       ├── feature_engineering    <- This contains a script for each of the features we engineered
       └── plots                  <- This contains a script for each plot used in the report



