## Shared test cases for the accessors of the results: spicyR (R) writes what topPairs() and bind() return on the
## cells of make_cases.R; tests/test_accessors.py must give the same tables with top_pairs() and bind(). Rerun after
## any change to either front end's accessors:
##   Rscript tests/shared_cases/make_accessor_cases.R      (needs spicyR >= 1.99.42)
suppressPackageStartupMessages(library(spicyR))
out_dir <- file.path("tests", "shared_cases")
cells <- utils::read.csv(file.path(out_dir, "cells.csv"))
cells$os <- survival::Surv(cells$time, cells$event)

fits <- list(
  two_groups = list(condition = "condition", subject = "patient", r = 30),
  covariates = list(condition = "condition", subject = "patient", r = 30, covariates = c("age", "batch"),
                    adjustAbundance = TRUE, from = c("tumour", "T"), to = c("T", "B", "macro")),
  three_levels = list(condition = "stage", r = 30, from = "tumour"),
  survival = list(condition = "os", subject = "patient", r = 30, covariates = "age", from = c("tumour", "T")),
  radii = list(condition = "condition", subject = "patient", r = c(10, 20, 40), from = c("tumour", "T"),
               to = c("T", "B"))
)
# each case: a fit, and the topPairs() or bind() arguments
cases <- list(
  top_default = list(fit = "two_groups", call = "topPairs", args = list()),
  top_from_to = list(fit = "two_groups", call = "topPairs", args = list(from = c("tumour", "T"), to = c("T", "B"))),
  top_cutoff = list(fit = "two_groups", call = "topPairs", args = list(cutoff = 0.9)),
  top_cutoff_n = list(fit = "two_groups", call = "topPairs", args = list(cutoff = 0.9, n = 3)),
  top_figures = list(fit = "two_groups", call = "topPairs", args = list(n = 6, figures = 3)),
  top_full = list(fit = "two_groups", call = "topPairs", args = list(n = 5, full = TRUE)),
  top_covariates = list(fit = "covariates", call = "topPairs", args = list(n = NULL)),
  top_three_levels = list(fit = "three_levels", call = "topPairs", args = list(n = NULL)),
  top_three_levels_coef = list(fit = "three_levels", call = "topPairs", args = list(coef = "III", n = NULL)),
  top_survival = list(fit = "survival", call = "topPairs", args = list(n = NULL)),
  top_radii = list(fit = "radii", call = "topPairs", args = list(to = "T")),
  bind_pair = list(fit = "two_groups", call = "bind", args = list(from = "tumour", to = "T")),
  bind_pair_no_subject = list(fit = "three_levels", call = "bind", args = list(from = "tumour", to = "macro"))
)
res <- lapply(fits, function(a) suppressMessages(do.call(spicy, c(list(cells = cells), a))))
for (nm in names(cases)) {
  cs <- cases[[nm]]
  out <- do.call(cs$call, c(list(res[[cs$fit]]), cs$args))
  utils::write.csv(out, file.path(out_dir, paste0("accessor_", nm, ".csv")), row.names = TRUE)
}
writeLines(jsonlite::toJSON(list(fits = fits, cases = cases), auto_unbox = TRUE, digits = NA, pretty = TRUE,
                            null = "null"),
           file.path(out_dir, "accessor_cases.json"))
cat("wrote", length(cases), "cases with spicyR", as.character(utils::packageVersion("spicyR")), "\n")
