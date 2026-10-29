## The few-effective-patients case of spicyR's tests (ne_cells, test-cell.R) and spicyR's results on it, for spicyr's
## tests/test_effective_units.py: Rscript tests/shared_cases/make_neff_cases.R <spicyr tests/shared_cases dir>
suppressPackageStartupMessages(library(spicyR)); cat(as.character(packageVersion("spicyR")), "\n")
od <- commandArgs(TRUE)[1]
ne_cells <- function(seed = 4, n_img = 16, big = 30, n0 = 300, levels = c("A", "B")) {
  set.seed(seed)
  do.call(rbind, lapply(seq_len(n_img), function(i) {
    m <- if (i == 1) big * n0 else n0; side <- 200 * sqrt(m / n0)
    z <- data.frame(x = runif(m, 0, side), y = runif(m, 0, side),
                    cellType = sample(c("tumour", "T", "other"), m, TRUE, c(.3, .2, .5)))
    tum <- which(z$cellType == "tumour"); tc <- which(z$cellType == "T"); f <- runif(1, 0, 0.6)
    mv <- tc[runif(length(tc)) < f]; h <- sample(tum, length(mv), TRUE)
    a <- runif(length(mv), 0, 2 * pi); dd <- runif(length(mv), 8, 10)
    z$x[mv] <- z$x[h] + dd * cos(a); z$y[mv] <- z$y[h] + dd * sin(a)
    z$imageID <- sprintf("img%02d", i); z$condition <- levels[(i - 1) %% length(levels) + 1]; z$age <- 40 + (i * 7) %% 13
    z }))
}
wr <- function(d, f) { d$x <- sprintf("%.17g", d$x); d$y <- sprintf("%.17g", d$y)
  utils::write.csv(d, file.path(od, f), row.names = FALSE, quote = FALSE) }
d <- ne_cells(); d3 <- ne_cells(seed = 5, n_img = 24, levels = c("A", "B", "C"))
wr(d, "neff_cells.csv"); wr(d3, "neff_cells3.csv")
d2 <- utils::read.csv(file.path(od, "neff_cells.csv"))   # as Python reads it
stopifnot(identical(d2$x, d$x), identical(d2$y, d$y))
run <- function(r, ...) suppressMessages(spicy(d, "condition", r = r, from = "tumour", to = "T", ...))
cases <- list(single = list(r = 5), single_ok = list(r = 12), single_cov = list(r = 5, covariates = "age"),
              single_ok_cov = list(r = 12, covariates = "age"), single_ok_ab = list(r = 12, adjustAbundance = TRUE),
              single_ok_hk = list(r = 12, variance = "hartung_knapp"), count_single = list(r = 5, effect = "count"),
              count_davies = list(r = c(5, 12, 15), combine = "davies", effect = "count"))
for (cmb in c("davies", "maxT", "cauchy")) {
  cases[[cmb]] <- list(r = c(5, 12, 15), combine = cmb)
  cases[[paste0(cmb, "_adj")]] <- list(r = c(5, 12, 15), combine = cmb, adjustAbundance = TRUE)
}
num <- function(z) if (is.null(z)) NULL else unname(as.list(ifelse(is.na(z), NA, sprintf("%.17g", z))))
out <- lapply(cases, function(a) {
  x <- do.call(run, a); cr <- x$cellResults; rr <- x$radiusResults
  list(args = a, p_value = num(cr$p_value), unadjusted_p_value = num(cr$unadjusted_p_value), r = num(cr$r),
       excess_difference = num(cr$excess_difference), se = num(cr$se), df = num(cr$df),
       adjusted_for = cr$adjusted_for, radius_p = num(rr$p_value), radius_r = num(rr$r), radius_adjusted_for = rr$adjusted_for)
})
writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, null = "null", na = "null", pretty = TRUE),
           file.path(od, "neff_expected.json"))
cat("wrote", length(out), "cases\n")
