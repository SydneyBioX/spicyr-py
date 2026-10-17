# spicyr news

## 1.99.25

- A radius is now not tested when a condition's effective patients fall below min(3, m_g / 2), where m_g is the number
  of that condition's patients that contribute at that radius; before, the bound was 3. For conditions of 6 or more
  patients nothing changes. The rule applies wherever the old one did: `spicy()` (both effects, with or without
  adjustment, Hartung-Knapp, and each level with more than two conditions). The label stays
  `"few_effective_units"`.
- Small studies get most of their tests back. Over the small real-data analyses, the radii of the grid that are tested
  rise from 78% to 95%, and a condition of 3 patients can now be tested (none could before).
- Core synced from spicyR 1.99.30.

## 1.99.24

- `spicy()` (both effects) no longer tests a radius where a condition has fewer than 3 effective patients: Kish's
  (sum w)^2 / sum w^2 of the patients' weights in that condition. A patient's weight grows with the information of its
  images, and when the patients do not differ (frailty variance 0) nothing caps it, so one patient with many cells can
  carry nearly all of a condition's weight. Such a radius has p-value NaN (and the unadjusted one; with adjustment,
  `adjusted_for` is `"none (few_effective_units)"`) and is left out of the combination over radii, as a radius with no
  informative patient is. With more than two conditions, a pair where any condition has fewer than 3 is not tested. A
  condition with fewer than 3 patients is never tested, and one with 3 or 4 rarely is.
- The test now shares one dispersion between the conditions. Each condition's CR2 variance is divided by its
  model-based variance, the ratios are averaged with the conditions' numbers of patients as weights, and the
  model-based variance of the difference is multiplied by this average; the Satterthwaite df are unchanged. Before,
  each condition's CR2 variance entered on its own, so with conditions of unequal size and skewed patient effects a
  small condition's low spread set the standard error and the test gave too many small p-values. With adjustment, the
  ratios are those of each condition's coefficient in the adjusted design, and every coefficient's variance (also the
  reported effect of each adjustment) is scaled by their average. `variance="hartung_knapp"` is unchanged.
- Images with a missing condition are now also left out of the choice of an allocation pair's side and of the
  between-image variance (`tau2_image`), so the result equals the one with those images removed, also with adjustment
  and several images per patient.
- Survival is unchanged.
- Calibration, on 11 real-data analyses with 30 relabellings of the conditions between patients each (spicyR 1.99.29,
  which gives the same results), as the share of relabellings with any BH-significant pair over 25 radii (davies): for
  the allocation effect, from 37% to 3.3% in the worst analysis (404 patients, 340 against 64), from 23% to 3.3% in
  the next (one patient carried 99% of a condition's weight), and from 5.2% to 3.0% on average over the other 9; for
  the count effect, from 33% to 3.3%, from 27% to 0% and from 4.1% to 3.0%. With the true labels, BH-significant pairs
  go from 535 to 510 (allocation) and from 641 to 546 (count).
- Of the 62 reference configurations, the 4 of survival and the 2 with Hartung-Knapp and covariates are identical to
  1.99.23's; the 28 of each effect change in their standard errors and p-values.
- The shared C++ core is synced from spicyR 1.99.29.

## 1.99.23

- `spicy()` leaves images whose condition is missing out of the unadjusted test. Before, the test of two conditions
  counted them in the second condition: its excess, the difference, the standard error, the df, the p-value and the
  frailty variance all included them. This affected the test without adjustment and the `unadjusted_*` columns of an
  adjusted test. The adjusted design already left these images out, so its results are unchanged, except where its
  frailty variance is held at the unadjusted value (an added column that varies between the images of a patient, such
  as `adjust_abundance=True` with several images per patient). There the held variance is now that of the images with
  a condition, and the adjusted test changes with it (spicyR#83).
- With more than two conditions, images with a missing condition entered the test with no condition and changed its
  results. They are now left out, as with two conditions.
- The test now equals the one with these images removed, except for two steps that use every image: the side of an
  allocation pair (attraction or avoidance), and the between-image variance (`tau2_image`) when patients have several
  images.
- Results with no missing condition are unchanged: every reference configuration is identical to 1.99.22's.
- The shared C++ core is synced from spicyR 1.99.28, which gives the same results.
