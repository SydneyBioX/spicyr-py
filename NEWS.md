# spicyr news

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
