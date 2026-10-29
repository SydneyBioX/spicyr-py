## The no-information case of spicyR's tests (ni_cells, test-cell.R) and spicyR's results on it, for spicyr's
## tests/test_no_information.py: Rscript tests/shared_cases/make_noinfo_cases.R <spicyr tests/shared_cases dir>
suppressPackageStartupMessages(library(spicyR)); cat(as.character(packageVersion("spicyR")), "\n")
od <- commandArgs(TRUE)[1]
ni_cells <- function(n_img = 16, seed = 5, levels = c("A", "B")) {
  set.seed(seed)
  do.call(rbind, lapply(seq_len(n_img), function(i) {
    g <- expand.grid(x = seq(10, 190, 20), y = seq(10, 190, 20)); nt <- sample(15:25, 1)
    g <- g[sample(nrow(g), nt), ]; tx <- g$x; ty <- g$y
    near <- function(n, lo, hi) { k <- sample(nt, n, TRUE); a <- runif(n, 0, 2 * pi); d <- runif(n, lo, hi)
      cbind(tx[k] + d * cos(a), ty[k] + d * sin(a)) }
    grp <- levels[(i - 1) %% length(levels) + 1]
    nT <- sample(30:50, 1); nc <- if (grp == "A") 0 else round(nT * runif(1, 0.05, 0.3))
    nB <- sample(40:60, 1); nF <- sample(30:50, 1)
    xy <- rbind(cbind(tx, ty), near(nc, 2.5, 3.5), near(nT - nc, 4.5, 5.5), near(nB, 0.2, 1),
                cbind(runif(nF, 300, 400), runif(nF, 300, 400)))
    data.frame(imageID = sprintf("img%02d", i), condition = grp, x = xy[, 1], y = xy[, 2],
               cellType = rep(c("tumour", "T", "B", "far"), c(nt, nT, nB, nF)), age = 40 + (i * 7) %% 13)
  }))
}
wr <- function(d, f) { d$x <- sprintf("%.17g", d$x); d$y <- sprintf("%.17g", d$y)
  utils::write.csv(d, file.path(od, f), row.names = FALSE, quote = FALSE) }
d <- ni_cells(); d3 <- ni_cells(24, levels = c("A", "B", "C"))
wr(d, "noinfo_cells.csv"); wr(d3, "noinfo_cells3.csv")
d2 <- utils::read.csv(file.path(od, "noinfo_cells.csv"))   # as Python reads it
stopifnot(identical(d2$x, d$x), identical(d2$y, d$y))
run <- function(r, ...) suppressMessages(spicy(d, "condition", r = r, from = "tumour", to = "T", ...))
cases <- list(single = list(r = 4), single_adj = list(r = 4, covariates = "age", adjustAbundance = TRUE),
              single_ok = list(r = 4.8), count_single = list(r = 4, effect = "count"),
              count_davies = list(r = c(4, 4.8, 5.2), combine = "davies", effect = "count"))
for (cmb in c("davies", "maxT", "cauchy")) {
  cases[[cmb]] <- list(r = c(4, 4.8, 5.2), combine = cmb)
  cases[[paste0(cmb, "_adj")]] <- list(r = c(4, 4.8, 5.2), combine = cmb, adjustAbundance = TRUE)
}
num <- function(z) if (is.null(z)) NULL else unname(as.list(ifelse(is.na(z), NA, sprintf("%.17g", z))))
out <- lapply(cases, function(a) {
  x <- do.call(run, a); cr <- x$cellResults; rr <- x$radiusResults
  list(args = a, p_value = num(cr$p_value), unadjusted_p_value = num(cr$unadjusted_p_value), r = num(cr$r),
       excess_difference = num(cr$excess_difference), se = num(cr$se), df = num(cr$df),
       adjusted_for = cr$adjusted_for, radius_p = num(rr$p_value), radius_r = num(rr$r), radius_adjusted_for = rr$adjusted_for)
})
writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, null = "null", na = "null", pretty = TRUE),
           file.path(od, "noinfo_expected.json"))
cat("wrote", length(out), "cases\n")
