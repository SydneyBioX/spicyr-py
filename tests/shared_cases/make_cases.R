## Shared test cases for the twins: spicyR (R) computes the expected results; the Python tests
## (tests/test_shared_cases.py) must reproduce them. Rerun after any change to the core or to either front end:
##   Rscript tests/shared_cases/make_cases.R      (needs spicyR >= 1.99.57)
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
# (no random draws below: the columns above are those of spicyR <= 1.99.56's cases)
cells$grp3 <- ifelse(cells$condition == "B", "B", ifelse(match(cells$patient, pat) %% 2 == 1, "A1", "A2"))  # A split in two
cells$time_na <- ifelse(cells$patient %in% c("A03", "B02", "B06"), NA, cells$time)         # no outcome for three patients
cells$img_cells <- as.vector(table(cells$imageID)[cells$imageID]) / 100                    # image-level, varies by patient
cells$time_few <- ifelse(cells$patient %in% c("A01", "A02", "A04", "B01", "B03"), cells$time, NA)   # five outcomes
utils::write.csv(cells, file.path(out_dir, "cells.csv"), row.names = FALSE)

cases <- list(
  two_groups = list(condition = "condition", subject = "patient", r = 30),
  adjusted = list(condition = "condition", subject = "patient", r = 30, adjustAbundance = TRUE),
  count = list(condition = "condition", subject = "patient", r = 30, effect = "count"),
  count_adjusted = list(condition = "condition", subject = "patient", r = 30, effect = "count", adjustAbundance = TRUE),
  knn_count = list(condition = "condition", subject = "patient", k = 10, from = c("tumour", "macro"), to = c("T", "B"), effect = "count"),
  survival_count = list(condition = "os", subject = "patient", r = 30, from = c("tumour", "T"), to = c("T", "B", "macro"), covariates = "age", effect = "count"),
  hartung_knapp = list(condition = "condition", subject = "patient", r = 30, variance = "hartung_knapp"),
  images_as_units = list(condition = "condition", r = 30, from = "tumour", to = c("T", "B")),
  three_levels = list(condition = "stage", subject = "patient", r = 30, from = c("tumour", "T"), to = c("T", "B")),
  three_levels_covariates = list(condition = "stage", subject = "patient", r = 30, from = "tumour", to = c("T", "B"), covariates = c("age", "batch")),
  covariates = list(condition = "condition", subject = "patient", r = 30, covariates = c("age", "batch"), from = "tumour"),
  knn = list(condition = "condition", subject = "patient", k = 10, from = c("tumour", "macro"), to = c("T", "B")),
  radii_maxT = list(condition = "condition", subject = "patient", r = c(10, 20, 40), from = "tumour", to = c("T", "B", "macro")),
  radii_cauchy = list(condition = "condition", subject = "patient", r = c(10, 20, 40), from = "tumour", to = c("T", "B"), combine = "cauchy"),
  radii_davies = list(condition = "condition", subject = "patient", r = seq(5, 40, by = 5), from = "tumour", to = c("T", "B"), combine = "davies"),
  survival = list(condition = "os", subject = "patient", r = 30, from = c("tumour", "T"), to = c("T", "B", "macro"), covariates = "age"),
  covariates_missing = list(condition = "condition", subject = "patient", r = 30, covariates = c("age_na", "batch"), from = c("tumour", "T")),
  survival_missing = list(condition = "os", subject = "patient", r = 30, from = "T", to = c("tumour", "B"), covariates = "age_na"),
  three_levels_hk = list(condition = "stage", subject = "patient", r = 30, from = c("tumour", "T"), to = c("T", "B"), variance = "hartung_knapp"),
  continuous = list(condition = "age", subject = "patient", r = 30),
  continuous_adjusted = list(condition = "age", subject = "patient", r = 30, from = c("tumour", "T"), adjustAbundance = TRUE, covariates = "batch"),
  continuous_hk = list(condition = "age", subject = "patient", r = 30, from = "tumour", variance = "hartung_knapp"),
  continuous_missing = list(condition = "age_na", subject = "patient", r = 30, from = c("tumour", "T"), to = c("T", "B")),
  # spicyR 1.99.57: follow-ups of more than two conditions, max-T for any number of radii, adjusted designs with an
  # image-level column (abundance share, batch), survival rules and covariates on both sides
  follow_ups = list(condition = "grp3", subject = "patient", r = 30, from = c("tumour", "T", "macro"), to = c("T", "B")),
  follow_ups_adjusted = list(condition = "grp3", subject = "patient", r = 30, from = c("tumour", "T"), to = c("T", "B"),
                             adjustAbundance = TRUE, covariates = "batch", alpha = 0.2),
  follow_ups_count = list(condition = "grp3", subject = "patient", r = 30, from = "tumour", to = c("T", "B", "macro"),
                          effect = "count", ref = "B"),
  radii_maxT_many = list(condition = "condition", subject = "patient", r = seq(5, 40, by = 5), from = "tumour", to = c("T", "B")),
  radii_maxT_adjusted = list(condition = "condition", subject = "patient", r = c(10, 20, 30, 40), from = c("tumour", "T"),
                             to = c("T", "B"), adjustAbundance = TRUE, covariates = "batch"),
  image_covariate = list(condition = "condition", subject = "patient", r = 30, covariates = "img_cells", from = c("tumour", "macro")),
  survival_adjusted = list(condition = "os", subject = "patient", r = 30, from = c("tumour", "T"), to = c("T", "B"),
                           covariates = c("age", "img_cells"), adjustAbundance = TRUE),
  survival_no_outcome = list(condition = "os_na", subject = "patient", r = 30, from = "tumour", to = c("T", "B", "macro"),
                             adjustAbundance = TRUE),
  survival_few = list(condition = "os_few", subject = "patient", r = 30, from = "tumour", to = c("T", "B"),
                      adjustAbundance = TRUE)
)
manifest <- list()
for (nm in names(cases)) {
  a <- cases[[nm]]; d <- cells
  if (a$condition == "os") d$os <- survival::Surv(d$time, d$event)
  if (a$condition == "os_na") d$os_na <- survival::Surv(d$time_na, d$event)
  if (a$condition == "os_few") d$os_few <- survival::Surv(d$time_few, d$event)
  res <- suppressMessages(do.call(spicy, c(list(cells = d), a)))
  tab <- res$cellResults
  utils::write.csv(tab, file.path(out_dir, paste0(nm, ".csv")), row.names = TRUE)
  if (!is.null(res$followUpResults)) utils::write.csv(res$followUpResults, file.path(out_dir, paste0(nm, "_followups.csv")), row.names = TRUE)
  if (!is.null(res$radiusResults)) utils::write.csv(res$radiusResults, file.path(out_dir, paste0(nm, "_radii.csv")), row.names = FALSE)
  manifest[[nm]] <- a
}
writeLines(jsonlite::toJSON(manifest, auto_unbox = TRUE, digits = NA, pretty = TRUE), file.path(out_dir, "cases.json"))
cat("wrote", length(cases), "cases with spicyR", as.character(utils::packageVersion("spicyR")), "\n")
