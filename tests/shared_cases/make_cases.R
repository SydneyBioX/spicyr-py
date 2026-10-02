## Shared test cases for the twins: spicyR (R) computes the expected results; the Python tests
## (tests/test_shared_cases.py) must reproduce them. Rerun after any change to the core or to either front end:
##   Rscript tests/shared_cases/make_cases.R      (needs spicyR >= 1.99.0)
suppressPackageStartupMessages(library(spicyR))
out_dir <- file.path("tests", "shared_cases")
set.seed(20261002)
types <- c("tumour", "T", "B", "macro", "stroma", "rare"); prob <- c(.3, .15, .15, .15, .2, .05)
cells <- do.call(rbind, lapply(c("A", "B"), function(g) do.call(rbind, lapply(1:7, function(p) do.call(rbind, lapply(1:2, function(im) {
  n <- 300 + sample(-40:40, 1)
  z <- data.frame(x = round(runif(n, 0, 400), 2), y = round(runif(n, 0, 400), 2), cellType = sample(types, n, TRUE, prob))
  if (g == "B") { tum <- which(z$cellType == "tumour"); tc <- which(z$cellType == "T")
    mv <- tc[runif(length(tc)) < 0.35]; h <- sample(tum, length(mv), TRUE)
    z$x[mv] <- round(z$x[h] + runif(length(mv), -6, 6), 2); z$y[mv] <- round(z$y[h] + runif(length(mv), -6, 6), 2) }
  z$imageID <- sprintf("%s%02d_%d", g, p, im); z$patient <- sprintf("%s%02d", g, p); z$condition <- g
  z }))))))
pat <- unique(cells$patient)
cells$age <- round(50 + 10 * stats::rnorm(length(pat)), 1)[match(cells$patient, pat)]          # patient-level
cells$batch <- c("b1", "b2")[1 + (match(cells$imageID, unique(cells$imageID)) %% 2)]            # image-level
cells$stage <- c("I", "II", "III")[1 + (match(cells$patient, pat) %% 3)]                         # three levels
cells$time <- round(stats::rexp(length(pat), 0.1), 2)[match(cells$patient, pat)]
cells$event <- stats::rbinom(length(pat), 1, 0.7)[match(cells$patient, pat)]
cells$age_na <- ifelse(cells$patient %in% c("A02", "B05"), NA, cells$age)                 # missing for two patients
utils::write.csv(cells, file.path(out_dir, "cells.csv"), row.names = FALSE)

cases <- list(
  two_groups = list(condition = "condition", subject = "patient", r = 30),
  unadjusted = list(condition = "condition", subject = "patient", r = 30, adjustAbundance = FALSE),
  hk_no_clustering = list(condition = "condition", subject = "patient", r = 30, variance = "hartung_knapp", labelClustering = FALSE),
  images_as_units = list(condition = "condition", r = 30, from = "tumour", to = c("T", "B")),
  three_levels = list(condition = "stage", subject = "patient", r = 30, from = c("tumour", "T"), to = c("T", "B")),
  three_levels_covariates = list(condition = "stage", subject = "patient", r = 30, from = "tumour", to = c("T", "B"), covariates = c("age", "batch")),
  covariates = list(condition = "condition", subject = "patient", r = 30, covariates = c("age", "batch"), from = "tumour"),
  knn = list(condition = "condition", subject = "patient", k = 10, from = c("tumour", "macro"), to = c("T", "B")),
  radii_maxT = list(condition = "condition", subject = "patient", r = c(10, 20, 40), from = "tumour", to = c("T", "B", "macro")),
  radii_cauchy = list(condition = "condition", subject = "patient", r = c(10, 20, 40), from = "tumour", to = c("T", "B"), combine = "cauchy"),
  survival = list(condition = "os", subject = "patient", r = 30, from = c("tumour", "T"), to = c("T", "B", "macro"), covariates = "age"),
  covariates_missing = list(condition = "condition", subject = "patient", r = 30, covariates = c("age_na", "batch"), from = c("tumour", "T")),
  survival_missing = list(condition = "os", subject = "patient", r = 30, from = "T", to = c("tumour", "B"), covariates = "age_na")
)
manifest <- list()
for (nm in names(cases)) {
  a <- cases[[nm]]; d <- cells
  if (a$condition == "os") d$os <- survival::Surv(d$time, d$event)
  res <- suppressMessages(do.call(spicy, c(list(cells = d), a)))
  tab <- res$cellResults
  utils::write.csv(tab, file.path(out_dir, paste0(nm, ".csv")), row.names = TRUE)
  if (!is.null(res$radiusResults)) utils::write.csv(res$radiusResults, file.path(out_dir, paste0(nm, "_radii.csv")), row.names = FALSE)
  manifest[[nm]] <- a
}
writeLines(jsonlite::toJSON(manifest, auto_unbox = TRUE, digits = NA, pretty = TRUE), file.path(out_dir, "cases.json"))
cat("wrote", length(cases), "cases with spicyR", as.character(utils::packageVersion("spicyR")), "\n")
