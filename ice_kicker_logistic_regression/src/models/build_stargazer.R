# create robust standard errors
cov1 <- vcovHC(model1_confirmatory, type = "HC1")
cov2 <- vcovHC(model2_confirmatory, type = "HC1")
cov3 <- vcovHC(model3_confirmatory, type = "HC1")

robust_se1 <- sqrt(diag(cov1))
robust_se2 <- sqrt(diag(cov2))
robust_se3 <- sqrt(diag(cov3))

# create stargazer table
stargazer(model1_confirmatory, model2_confirmatory, model3_confirmatory,
          type = "latex",
          header = FALSE,
          title="Iced Field Goals - Regression Results",
          dep.var.labels = c("Field Goal Outcome",
                             "Field Goal Outcome",
                             "Field Goal Outcome"),
          covariate.labels=c("Iced",
                             "Kick Distance",
                             "Kick Distance Squared",
                             "Wind",
                             "Temperature",
                             "Covered Stadium",
                             "Score Differential",
                             "High Pressure Situation",
                             "Synthetic Surface",
                             "Accuracy of Last 5 FGs",
                             "Pro Bowl/All Pro",
                             "Wind * Covered Stadium",
                             "Temperature * Covered Stadium",
                             "Constant"),
          column.labels = c("Model 1 (Baseline)",
                            "Model 2 (Additional Variables)",
                            "Model 2 with PSM"),
          se = list(robust_se1, robust_se2, robust_se3),
          no.space = TRUE,
          font.size = "footnotesize")
