# spicyr

**Test whether cell types co-localise differently between groups of patients.**

Do T cells gather around tumour cells more in one group of patients than in another? spicyr tests this for every
pair of cell types in imaging and spatial transcriptomics data, comparing groups of patients or relating
co-localisation to survival. For each image it counts the cells of one type within a radius of each cell of another,
and compares that count with what you would expect if the cells had been labelled at random, using the cells
actually present. Holes, air spaces and uneven cell density therefore do not by themselves create a signal.
Patients, not images or cells, are the units of the test.

spicyr is the Python version of the Bioconductor package [spicyR](https://bioconductor.org/packages/spicyR) and
works with SpatialData, AnnData and pandas objects.

```{image} _static/spicyR_overview.png
:width: 100%
:alt: Left, a tumour cell with a 25 µm circle and the T cells inside it. Right, box plots of the extra T cells per tumour cell, one point per patient, higher in ER-positive than ER-negative tumours.
```

```python
import spicyr

cells = spicyr.datasets.metabric_ali2020()                      # breast cancer imaging mass cytometry
cells = cells[cells["ER.Status"].isin(["neg", "pos"])]
res = spicyr.spicy(cells, condition="ER.Status", subject="metabricId", r=25,
                   image_id="file_id", cell_type="description")
res.top_pairs()
res.signif_plot()
res.box_plot("HR- Ki67+", "T cells")
```

```{toctree}
:maxdepth: 2

tutorial
api
```
