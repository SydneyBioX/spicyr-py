## Shared cases for col_test() and get_prop(): spicyR's colTest() and getProp() on the cells of make_cases.R.
## Rerun after a change to either:  Rscript tests/shared_cases/make_col_test_cases.R
suppressPackageStartupMessages(library(spicyR))
out_dir <- file.path("tests", "shared_cases")
cells <- utils::read.csv(file.path(out_dir, "cells.csv"))
sce <- SingleCellExperiment::SingleCellExperiment(colData = S4Vectors::DataFrame(cells))
write <- function(x, name) utils::write.csv(x, file.path(out_dir, paste0("col_test_", name, ".csv")))

write(getProp(cells, feature = "cellType", imageID = "patient"), "prop_patient")
# exact Wilcoxon tests: patients (7 against 7) and images (14 against 14), no ties
write(colTest(sce, condition = "condition", feature = "cellType", imageID = "patient", type = "wilcox"), "wilcox_patient")
write(colTest(sce, condition = "condition", feature = "cellType", imageID = "imageID", type = "wilcox"), "wilcox_image")
write(colTest(sce, condition = "condition", feature = "cellType", imageID = "imageID"), "ttest_image")
# normal approximation: proportions rounded to two decimals have ties
props <- getProp(cells, feature = "cellType", imageID = "imageID")
cond <- unique(cells[, c("imageID", "condition")])
write(colTest(round(props, 2), cond$condition[match(rownames(props), cond$imageID)], type = "wilcox"), "wilcox_ties")
# survival: one row per patient
props <- getProp(cells, feature = "cellType", imageID = "patient")
surv <- unique(cells[, c("patient", "time", "event")])
write(colTest(props, surv[match(rownames(props), surv$patient), c("time", "event")], type = "survival"), "survival")
# normal approximation (60 against 70 values), without ties (Edgeworth-corrected) and with ties
set.seed(20261010)
big <- data.frame(a = stats::rnorm(130), b = round(stats::rnorm(130), 1), c = stats::rexp(130) + rep(c(0, 0.4), c(60, 70)))
big$group <- rep(c("lo", "hi"), c(60, 70))
utils::write.csv(big, file.path(out_dir, "col_test_big_input.csv"), row.names = FALSE)
write(colTest(big[, c("a", "b", "c")], big$group, type = "wilcox"), "wilcox_big")
