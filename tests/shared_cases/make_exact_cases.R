# The adjusted cases at full precision (17 significant digits), for the exact comparison in tests/test_pairs.py.
# Run from the repository root with spicyR installed: Rscript tests/shared_cases/make_exact_cases.R
suppressPackageStartupMessages(library(spicyR))
out_dir <- file.path("tests", "shared_cases")
cells <- utils::read.csv(file.path(out_dir, "cells.csv"))
base <- list(condition = "condition", subject = "patient")
cases <- list(
  adjusted = c(base, list(r = 30, adjustAbundance = TRUE)),
  count_adjusted = c(base, list(r = 30, effect = "count", adjustAbundance = TRUE)),
  covariates = c(base, list(r = 30, covariates = c("age", "batch"), from = "tumour")),
  covariates_missing = c(base, list(r = 30, covariates = c("age_na", "batch"), adjustAbundance = TRUE)),
  hartung_knapp_adjusted = c(base, list(r = 30, covariates = "age", adjustAbundance = TRUE, variance = "hartung_knapp")),
  knn_adjusted = c(base, list(k = 10, adjustAbundance = TRUE)),
  radii_adjusted = c(base, list(r = c(10, 20, 40), covariates = c("age_na", "batch"), adjustAbundance = TRUE)),
  radii_davies_adjusted = c(base, list(r = seq(5, 40, by = 5), combine = "davies", effect = "count",
                                       adjustAbundance = TRUE, from = c("tumour", "T"))),
  radii_cauchy_covariates = c(base, list(r = c(10, 20, 40), combine = "cauchy", covariates = "age"))
)
full <- function(d) {
  for (cn in names(d)) if (is.double(d[[cn]])) d[[cn]] <- ifelse(is.na(d[[cn]]), NA, sprintf("%.17g", d[[cn]]))
  d
}
for (nm in names(cases)) {
  res <- suppressMessages(do.call(spicy, c(list(cells = cells), cases[[nm]])))
  utils::write.csv(full(res$cellResults), file.path(out_dir, paste0("exact_", nm, ".csv")), row.names = TRUE)
  if (!is.null(res$radiusResults))
    utils::write.csv(full(res$radiusResults), file.path(out_dir, paste0("exact_", nm, "_radii.csv")), row.names = FALSE)
}
writeLines(jsonlite::toJSON(cases, auto_unbox = TRUE, digits = NA, pretty = TRUE), file.path(out_dir, "exact_cases.json"))
cat("wrote", length(cases), "exact cases with spicyR", as.character(utils::packageVersion("spicyR")), "\n")
