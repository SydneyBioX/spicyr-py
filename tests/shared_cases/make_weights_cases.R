## Shared test cases for soft cell type membership (spicy(cellWeights =)): spicyR (R) computes the expected results
## on the cells of make_cases.R with the weights in weights.csv; tests/test_weights.py must reproduce them. Rerun after
## any change to the core or to either front end:
##   Rscript tests/shared_cases/make_weights_cases.R      (needs spicyR >= 1.99.60)
## Cases with `weights = "binary"` use weights_binary.csv: 0/1 weights of the cell types and a "lymphocyte" group (T or
## B), so a label with itself and a group with its subtypes give the cell types' self-pairs and nested labels.
suppressPackageStartupMessages(library(spicyR))
out_dir <- file.path("tests", "shared_cases")
cells <- utils::read.csv(file.path(out_dir, "cells.csv"))
# each cell's own type with weight 0.5 to 1, another type now and then with 0.1 to 0.3, and an overlapping
# "lymphocyte" label (T or B, at most 1)
set.seed(20261020)
types <- c("tumour", "T", "B", "macro", "stroma", "rare")
W <- matrix(0, nrow(cells), length(types), dimnames = list(NULL, types))
W[cbind(seq_len(nrow(cells)), match(cells$cellType, types))] <- round(stats::runif(nrow(cells), 0.5, 1), 3)
other <- which(stats::runif(nrow(cells)) < 0.3)
W[cbind(other, sample(length(types), length(other), TRUE))] <- round(stats::runif(length(other), 0.1, 0.3), 3)
W <- cbind(W, lymphocyte = pmin(1, W[, "T"] + W[, "B"]))
utils::write.csv(W, file.path(out_dir, "weights.csv"), row.names = FALSE)
W <- as.matrix(utils::read.csv(file.path(out_dir, "weights.csv")))
B <- outer(cells$cellType, types, "==") * 1
colnames(B) <- types
B <- cbind(B, lymphocyte = B[, "T"] + B[, "B"])
utils::write.csv(B, file.path(out_dir, "weights_binary.csv"), row.names = FALSE)
B <- as.matrix(utils::read.csv(file.path(out_dir, "weights_binary.csv")))

cases <- list(
  soft_allocation = list(condition = "condition", subject = "patient", r = 30),
  soft_count_adjusted = list(condition = "condition", subject = "patient", r = 30, effect = "count",
                             adjustAbundance = TRUE, from = c("tumour", "macro"), to = c("T", "lymphocyte")),
  soft_covariates = list(condition = "condition", subject = "patient", r = 30, covariates = c("age", "batch"),
                         adjustAbundance = TRUE, from = "tumour", to = c("T", "B", "lymphocyte")),
  soft_three_levels = list(condition = "stage", subject = "patient", r = 30, from = c("tumour", "lymphocyte"),
                           to = c("T", "macro")),
  soft_survival = list(condition = "os", subject = "patient", r = 30, from = "tumour", to = c("T", "lymphocyte"),
                       covariates = "age", adjustAbundance = TRUE),
  # a label with itself and labels that share cells, under random labelling of all the cells
  soft_count_self = list(condition = "condition", subject = "patient", r = 30, effect = "count",
                         from = c("T", "lymphocyte", "tumour"), to = c("T", "lymphocyte", "B")),
  soft_continuous_self = list(condition = "age", subject = "patient", r = 30, from = c("T", "lymphocyte"),
                              to = c("T", "lymphocyte")),
  soft_survival_self = list(condition = "os", subject = "patient", r = 30, from = c("B", "lymphocyte"),
                            to = c("B", "lymphocyte"), effect = "count"),
  binary_allocation = list(condition = "condition", subject = "patient", r = 30, weights = "binary",
                           from = c("lymphocyte", "T", "tumour"), to = c("lymphocyte", "T", "B", "tumour")),
  binary_count = list(condition = "condition", subject = "patient", r = 30, weights = "binary", effect = "count",
                      from = c("lymphocyte", "T", "tumour"), to = c("lymphocyte", "T", "B", "tumour")),
  binary_three_levels = list(condition = "stage", subject = "patient", r = 30, weights = "binary",
                             from = c("lymphocyte", "B"), to = c("lymphocyte", "B"), selfVariance = "exact"),
  binary_survival = list(condition = "os", subject = "patient", r = 30, weights = "binary",
                         from = c("lymphocyte", "T"), to = c("lymphocyte", "T"), adjustAbundance = TRUE)
)
## Two conditions: each condition's information per pair (spicyR >= 1.99.59) and, where testCorrelation() applies,
## the correlation between the tests by CR2 and by 200 permutations
.shared_correlation <- function(res, nm, out_dir) {
  if (is.null(res$conditionInformation)) return(invisible())
  utils::write.csv(res$conditionInformation, file.path(out_dir, paste0(nm, "_information.csv")), row.names = TRUE)
  R <- tryCatch(testCorrelation(res), error = function(e) NULL)
  if (is.null(R)) return(invisible())
  utils::write.csv(R, file.path(out_dir, paste0(nm, "_correlation_cr2.csv")), row.names = TRUE)
  utils::write.csv(testCorrelation(res, "permutation", nPermutations = 200),
                   file.path(out_dir, paste0(nm, "_correlation_permutation.csv")), row.names = TRUE)
}
manifest <- list()
for (nm in names(cases)) {
  a <- cases[[nm]]; d <- cells
  if (a$condition == "os") d$os <- survival::Surv(d$time, d$event)
  w <- if (identical(a$weights, "binary")) B else W
  res <- suppressMessages(do.call(spicy, c(list(cells = d, cellWeights = w), a[names(a) != "weights"])))
  utils::write.csv(res$cellResults, file.path(out_dir, paste0(nm, ".csv")), row.names = TRUE)
  .shared_correlation(res, nm, out_dir)
  manifest[[nm]] <- a
}
writeLines(jsonlite::toJSON(manifest, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           file.path(out_dir, "weights_cases.json"))
cat("wrote", length(cases), "cases with spicyR", as.character(utils::packageVersion("spicyR")), "\n")
