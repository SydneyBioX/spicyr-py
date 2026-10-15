// Python bindings for the spicyR C++ core (src/core, synced from spicyR; never edited here).
// They mirror spicyR's R bindings (src/glm_bindings.cpp, src/stats_bindings.cpp) one for one: convert,
// call the core, return. No statistics live in this file.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "spicyglm/core.hpp"
#include "spicyglm/stats.hpp"

namespace py = pybind11;
using namespace spicyglm;

namespace {

using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

template <typename T>
std::vector<T> vec(const py::array_t<T, py::array::c_style | py::array::forcecast>& a) {
  return std::vector<T>(a.data(), a.data() + a.size());
}

template <typename T>
py::array_t<T> arr(const std::vector<T>& v) {
  py::array_t<T> a(v.size());
  std::copy(v.begin(), v.end(), a.mutable_data());
  return a;
}

Window parse_window(const std::string& w) {
  if (w == "convex") return Window::Convex;
  if (w == "rectangle") return Window::Rectangle;
  throw std::invalid_argument("window must be 'convex' or 'rectangle'");
}

ImageRows rows_from(const py::dict& d) {
  ImageRows r;
  r.image = vec<int>(d["img"].cast<IntArray>());
  r.O = vec<double>(d["O"].cast<DoubleArray>());
  r.E = vec<double>(d["E"].cast<DoubleArray>());
  r.n = vec<double>(d["n"].cast<DoubleArray>());
  r.v = vec<double>(d["v"].cast<DoubleArray>());
  return r;
}

py::dict rows_to(const ImageRows& r) {
  py::dict d;
  d["img"] = arr(r.image); d["O"] = arr(r.O); d["E"] = arr(r.E); d["n"] = arr(r.n); d["v"] = arr(r.v);
  return d;
}

py::dict design_dict(const DesignResult& d) {
  py::dict o;
  o["ok"] = d.ok; o["reason"] = d.reason; o["theta"] = arr(d.theta); o["estimate"] = d.estimate; o["se"] = d.se;
  o["df"] = d.df; o["p"] = d.p; o["tau2"] = d.tau2; o["influence"] = arr(d.influence);
  return o;
}

// image-major (img * T + t) layout from a 2-D images x types array
std::vector<double> image_major(const DoubleArray& counts) {
  return std::vector<double>(counts.data(), counts.data() + counts.size());
}


// ---- the cell method image by image -------------------------------------------------------------------------------
// spicy()'s cell method needs only per-image summaries of the neighbour graph, so the cells never sit in the core as a
// whole: they stay in the caller's columns (in input order, in memory or on disk) and a batch of whole images at a
// time is gathered into a Dataset of its own. Every Dataset function used here works image by image on the image's
// cells in row order, so a batch's Dataset gives each of its images exactly the numbers the whole-data Dataset gives
// it (cells sorted stably by image). Mirrors cell_layout / cell_stream_totals of spicyR's src/glm_bindings.cpp.

// Integer codes of any width (pandas categorical codes are int8/int16/int32; -1 is missing).
struct Codes {
  const void* p = nullptr;
  int size = 0;  // bytes per code
  std::size_t n = 0;
  explicit Codes(const py::array& a, const char* what) {
    if (a.ndim() != 1) throw std::invalid_argument(std::string(what) + ": expected a 1-d array of codes");
    if (!(a.flags() & py::array::c_style)) throw std::invalid_argument(std::string(what) + ": codes must be contiguous");
    const char kind = a.dtype().kind();  // native byte order (the caller converts other arrays)
    if (kind != 'i') throw std::invalid_argument(std::string(what) + ": codes must be signed integers");
    p = a.data(); size = static_cast<int>(a.itemsize()); n = static_cast<std::size_t>(a.size());
  }
  long long operator[](std::size_t r) const {
    switch (size) {
      case 1: return static_cast<const std::int8_t*>(p)[r];
      case 2: return static_cast<const std::int16_t*>(p)[r];
      case 4: return static_cast<const std::int32_t*>(p)[r];
      default: return static_cast<const std::int64_t*>(p)[r];
    }
  }
};

// Coordinates as float32 or float64 (a float32 value converts to double exactly).
struct Coords {
  const float* f = nullptr;
  const double* d = nullptr;
  std::size_t n = 0;
  Coords(const py::array& a, const char* what) {
    if (a.ndim() != 1 || !(a.flags() & py::array::c_style))
      throw std::invalid_argument(std::string(what) + ": expected a contiguous 1-d array");
    // native byte order (the caller converts other arrays)
    if (a.dtype().kind() == 'f' && a.itemsize() == 8) d = static_cast<const double*>(a.data());
    else if (a.dtype().kind() == 'f' && a.itemsize() == 4) f = static_cast<const float*>(a.data());
    else throw std::invalid_argument(std::string(what) + ": coordinates must be float32 or float64");
    n = static_cast<std::size_t>(a.size());
  }
  double operator[](std::size_t r) const { return d ? d[r] : static_cast<double>(f[r]); }
};

// One pass over the cells: for every (image code, type code) the number of cells and the first row (-1 when there is
// none), and for every image code its last row. A missing type (code -1) is the extra last column; a missing image is
// an error.
py::dict cell_layout(const py::array& image_codes, int n_image_codes, const py::array& type_codes, int n_type_codes) {
  const Codes im(image_codes, "image"), ty(type_codes, "cell type");
  if (im.n != ty.n) throw std::invalid_argument("image and cell type codes differ in length");
  const std::size_t T = static_cast<std::size_t>(n_type_codes) + 1, I = static_cast<std::size_t>(n_image_codes);
  py::array_t<long long> counts({I, T}), first({I, T}), last(I);
  long long* c = counts.mutable_data();
  long long* f = first.mutable_data();
  long long* l = last.mutable_data();
  std::fill(c, c + I * T, 0LL); std::fill(f, f + I * T, -1LL); std::fill(l, l + I, -1LL);
  long long bad = -1;
  {
    py::gil_scoped_release release;
    for (std::size_t r = 0; r < im.n; ++r) {
      const long long i = im[r], t = ty[r] < 0 ? n_type_codes : ty[r];
      if (i < 0 || i >= n_image_codes || t > n_type_codes) { bad = static_cast<long long>(r); break; }
      const std::size_t at = static_cast<std::size_t>(i) * T + static_cast<std::size_t>(t);
      if (c[at]++ == 0) f[at] = static_cast<long long>(r);
      l[i] = static_cast<long long>(r);
    }
  }
  if (bad >= 0) throw std::invalid_argument("missing or invalid image or cell type code in row " + std::to_string(bad));
  py::dict out;
  out["counts"] = counts; out["first"] = first; out["last"] = last;
  return out;
}

// The rows (ascending) of the images selected by `keep` (indexed by image code), in one pass.
py::array_t<long long> rows_of_images(const py::array& image_codes, const py::array_t<bool, py::array::c_style | py::array::forcecast>& keep) {
  const Codes im(image_codes, "image");
  const bool* k = keep.data();
  const long long nk = keep.size();
  std::vector<long long> rows;
  {
    py::gil_scoped_release release;
    for (std::size_t r = 0; r < im.n; ++r) {
      const long long i = im[r];
      if (i >= 0 && i < nk && k[i]) rows.push_back(static_cast<long long>(r));
    }
  }
  return arr(rows);
}

// Whether the values (numbers, compared by their bits) are the same in every row of each image as in the image's first
// row (first_row: per image code).
bool constant_within(const py::array& values, const py::array& image_codes, const py::array_t<long long, py::array::c_style | py::array::forcecast>& first_row) {
  const Codes im(image_codes, "image");
  if (values.ndim() != 1 || static_cast<std::size_t>(values.size()) != im.n || !(values.flags() & py::array::c_style))
    throw std::invalid_argument("values: expected a contiguous 1-d array as long as the codes");
  const std::size_t w = static_cast<std::size_t>(values.itemsize());
  const char* v = static_cast<const char*>(values.data());
  const long long* f = first_row.data();
  py::gil_scoped_release release;
  for (std::size_t r = 0; r < im.n; ++r) {
    const long long fr = f[im[r]];
    if (std::memcmp(v + r * w, v + static_cast<std::size_t>(fr) * w, w) != 0) return false;
  }
  return true;
}

// The neighbour totals of one batch of whole images, added into the full per-image arrays. The batch's cells are in
// input order: x, y, image and type codes; image_map / type_map (by code; type_map's last entry is the missing type)
// give the image (in the sorted order of the analysis) and cell type. k > 0: the k nearest neighbours; else the radius
// graph at `radii` (one radius, or several from one neighbour search). out_a and out_b hold one array per radius (one
// for kNN): any and self_expected (allocation), or totals and sq (count), laid out as the whole-data Dataset gives them
// (image-major). Every image present must be in the batch whole.
void stream_batch(const py::array& x, const py::array& y, const py::array& image_codes, const py::array& type_codes,
                  const IntArray& image_map, const IntArray& type_map, int n_types, const DoubleArray& radii, int k,
                  bool allocation, const py::list& out_a, const py::list& out_b, int n_threads) {
  const Coords X(x, "x"), Y(y, "y");
  const Codes im(image_codes, "image"), ty(type_codes, "cell type");
  const std::size_t n = im.n;
  if (X.n != n || Y.n != n || ty.n != n) throw std::invalid_argument("x, y and the codes differ in length");
  const bool knn = k > 0;
  const std::size_t K = knn ? 1 : static_cast<std::size_t>(radii.size());
  if (K < 1) throw std::invalid_argument("give a radius or k");
  if (out_a.size() != K || out_b.size() != K) throw std::invalid_argument("one output array per radius");
  const std::size_t T = static_cast<std::size_t>(n_types);
  const int n_images = static_cast<int>(image_map.size()) == 0 ? 0 : *std::max_element(image_map.data(), image_map.data() + image_map.size()) + 1;
  std::vector<double*> a(K), b(K);
  for (std::size_t j = 0; j < K; ++j) {
    auto pa = out_a[j].cast<py::array_t<double, py::array::c_style>>();
    auto pb = out_b[j].cast<py::array_t<double, py::array::c_style>>();
    if (static_cast<std::size_t>(pa.size()) != static_cast<std::size_t>(n_images) * T * T ||
        static_cast<std::size_t>(pb.size()) != static_cast<std::size_t>(n_images) * T * (allocation ? 1 : T))
      throw std::invalid_argument("output arrays of the wrong size");
    a[j] = pa.mutable_data(); b[j] = pb.mutable_data();
  }
  const int* imap = image_map.data();
  const int* tmap = type_map.data();
  const long long n_icodes = image_map.size(), n_tcodes = type_map.size() - 1;
  const std::vector<double> rad(radii.data(), radii.data() + radii.size());
  std::string error;
  {
    py::gil_scoped_release release;
    // images of the batch in order of first appearance, with their cells in row order
    std::vector<int> local(static_cast<std::size_t>(n_images), -1), images, size;
    std::vector<int> row_local(n);
    for (std::size_t r = 0; r < n; ++r) {
      const long long c = im[r];
      if (c < 0 || c >= n_icodes || imap[c] < 0) { error = "missing or invalid image code"; break; }
      int& l = local[static_cast<std::size_t>(imap[c])];
      if (l < 0) { l = static_cast<int>(images.size()); images.push_back(imap[c]); size.push_back(0); }
      ++size[static_cast<std::size_t>(l)];
      row_local[r] = l;
    }
    if (error.empty()) {
      std::vector<int> offsets(images.size() + 1, 0);
      for (std::size_t i = 0; i < images.size(); ++i) offsets[i + 1] = offsets[i] + size[i];
      std::vector<int> fill(offsets.begin(), offsets.end() - 1), bt(n);
      std::vector<double> bx(n), by(n);
      for (std::size_t r = 0; r < n && error.empty(); ++r) {
        const long long tc = ty[r] < 0 ? n_tcodes : ty[r];
        if (tc > n_tcodes || tmap[tc] < 0 || tmap[tc] >= n_types) { error = "invalid cell type code"; break; }
        const int p = fill[static_cast<std::size_t>(row_local[r])]++;
        bx[p] = X[r]; by[p] = Y[r]; bt[p] = tmap[tc];
      }
      std::vector<int>().swap(row_local);
      if (error.empty()) {
        Dataset d(std::move(bx), std::move(by), std::move(bt), std::move(offsets), n_types);
        // copy the batch's per-image blocks (TT entries each) to the images' places
        auto put = [&](const std::vector<double>& from, double* to, std::size_t TT) {
          for (std::size_t i = 0; i < images.size(); ++i)
            std::copy(from.begin() + i * TT, from.begin() + (i + 1) * TT, to + static_cast<std::size_t>(images[i]) * TT);
        };
        if (K > 1) {
          Dataset::MultiRadiusTotals m = d.multi_radius_totals(rad, allocation);
          for (std::size_t j = 0; j < K; ++j) {
            if (allocation) { put(m.any[j], a[j], T * T); put(m.self_expected[j], b[j], T); }
            else { put(m.totals[j], a[j], T * T); put(m.sq[j], b[j], T * T); }
          }
        } else {
          if (knn) d.build_knn(k, n_threads); else d.build_radius_index(rad[0]);
          if (allocation) { put(d.pair_neighbour_any_totals(knn), a[0], T * T); put(d.self_any_expected(knn), b[0], T); }
          else { put(d.pair_neighbour_totals(knn), a[0], T * T); put(d.pair_neighbour_out_sq_totals(knn), b[0], T * T); }
        }
      }
    }
  }
  if (!error.empty()) throw std::invalid_argument(error);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "spicyR C++ core (shared with the R package spicyR)";

  py::class_<Dataset>(m, "Dataset")
      .def(py::init([](const DoubleArray& x, const DoubleArray& y, const IntArray& cell_type, const IntArray& image_offsets,
                       int n_types) { return Dataset(vec<double>(x), vec<double>(y), vec<int>(cell_type), vec<int>(image_offsets), n_types); }),
           py::arg("x"), py::arg("y"), py::arg("cell_type"), py::arg("image_offsets"), py::arg("n_types"))
      .def("image_areas", [](const Dataset& d, const std::string& w) { return arr(d.image_areas(parse_window(w))); })
      .def("build_radius_index", &Dataset::build_radius_index, py::arg("r"))
      .def("build_knn", &Dataset::build_knn, py::arg("k"), py::arg("n_threads") = 1)
      .def("pair_neighbour_totals", [](const Dataset& d, bool knn) { return arr(d.pair_neighbour_totals(knn)); })
      .def("pair_neighbour_out_sq_totals", [](const Dataset& d, bool knn) { return arr(d.pair_neighbour_out_sq_totals(knn)); })
      .def("pair_neighbour_any_totals", [](const Dataset& d, bool knn) { return arr(d.pair_neighbour_any_totals(knn)); })
      .def("self_any_expected", [](const Dataset& d, bool knn) { return arr(d.self_any_expected(knn)); })
      // several radii from one neighbour search: per radius (any, self_expected) or (totals, sq), as the single-radius calls
      .def("multi_radius_totals", [](Dataset& d, const DoubleArray& radii, bool allocation) {
        Dataset::MultiRadiusTotals m = d.multi_radius_totals(vec<double>(radii), allocation);
        py::list out;
        for (std::size_t k = 0; k < radii.size(); ++k)
          out.append(allocation ? py::make_tuple(arr(m.any[k]), arr(m.self_expected[k])) : py::make_tuple(arr(m.totals[k]), arr(m.sq[k])));
        return out;
      }, py::arg("radii"), py::arg("allocation"));

  m.def("cell_layout", &cell_layout, py::arg("image_codes"), py::arg("n_image_codes"), py::arg("type_codes"),
        py::arg("n_type_codes"));
  m.def("rows_of_images", &rows_of_images, py::arg("image_codes"), py::arg("keep"));
  m.def("constant_within", &constant_within, py::arg("values"), py::arg("image_codes"), py::arg("first_row"));
  m.def("stream_batch", &stream_batch, py::arg("x"), py::arg("y"), py::arg("image_codes"), py::arg("type_codes"),
        py::arg("image_map"), py::arg("type_map"), py::arg("n_types"), py::arg("radii"), py::arg("k"),
        py::arg("allocation"), py::arg("out_a"), py::arg("out_b"), py::arg("n_threads") = 1);

  m.def("pt_two_sided", [](double t, double df) { return pt_two_sided(t, df); });
  m.def("norm_quantile", [](double p) { return norm_quantile(p); });

  m.def("excess_image_rows", [](const DoubleArray& totals, const DoubleArray& sq, const DoubleArray& counts, int n_types,
                                int from, int to, bool knn, const DoubleArray& psi) {
    int n_images = static_cast<int>(counts.size() / n_types);
    return rows_to(excess_image_rows(vec<double>(totals), vec<double>(sq), image_major(counts), n_types, n_images, from, to,
                                     knn, vec<double>(psi)));
  });

  m.def("allocation_image_rows", [](const DoubleArray& any_totals, const DoubleArray& self_expected, const DoubleArray& counts,
                                    int n_types, int from, int to, const DoubleArray& psi) {
    int n_images = static_cast<int>(counts.size() / n_types);
    return rows_to(allocation_image_rows(vec<double>(any_totals), vec<double>(self_expected), image_major(counts), n_types,
                                         n_images, from, to, vec<double>(psi)));
  });

  m.def("image_heterogeneity_rows", [](const py::dict& rows, const IntArray& unit) {
    double tau2 = 0;
    py::dict d = rows_to(image_heterogeneity_rows(rows_from(rows), vec<int>(unit), &tau2));
    d["tau2_image"] = tau2;
    return d;
  });
  m.def("label_clustering_factor", [](const Dataset& d, const IntArray& from, const IntArray& to, const DoubleArray& counts,
                                      int n_types, bool knn, double h, bool allocation) {
    int n_images = static_cast<int>(counts.size() / n_types);
    return arr(label_clustering_factor(d, vec<int>(from), vec<int>(to), image_major(counts), n_types, n_images, knn, h,
                                       allocation));
  });

  m.def("excess_test", [](const py::dict& rows, const IntArray& unit, const IntArray& group, int n_units, bool frailty,
                          const std::string& variance) {
    ExcessResult r = excess_test(rows_from(rows), vec<int>(unit), vec<int>(group), n_units, frailty,
                                 variance == "hartung_knapp" ? Variance::HartungKnapp : Variance::CR2);
    py::dict o;
    o["ok"] = r.ok; o["reason"] = r.reason; o["coef_ref"] = r.coef_ref; o["coef_comp"] = r.coef_comp;
    o["difference"] = r.difference; o["se"] = r.se; o["df"] = r.df; o["p"] = r.p; o["tau2"] = r.tau2;
    o["influence"] = arr(r.influence); o["unit_summary"] = arr(r.unit_summary); o["unit_info"] = arr(r.unit_info);
    o["image_weight"] = arr(r.image_weight);
    return o;
  });

  m.def("design_test", [](const py::dict& rows, const IntArray& unit, int n_units, const DoubleArray& Z,
                          const DoubleArray& contrast, double tau2) {
    int q = static_cast<int>(contrast.size());
    return design_dict(design_test(rows_from(rows), vec<int>(unit), n_units, vec<double>(Z), q, vec<double>(contrast), tau2));
  });

  m.def("design_tests", [](const py::dict& rows, const IntArray& unit, int n_units, const DoubleArray& Z,
                           const DoubleArray& contrasts, int k, double tau2, bool hartung_knapp) {
    // Z: row-major images x q; contrasts: row-major k x q
    int q = static_cast<int>(contrasts.size() / k);
    std::vector<DesignResult> r = design_tests(rows_from(rows), vec<int>(unit), n_units, vec<double>(Z), q,
                                               vec<double>(contrasts), k, tau2, hartung_knapp);
    py::list out;
    for (const auto& d : r) out.append(design_dict(d));
    return out;
  });

  m.def("availability_test", [](const py::dict& rows, const IntArray& unit, const IntArray& group, int n_units,
                                const DoubleArray& x, double tau2) {
    return design_dict(availability_test(rows_from(rows), vec<int>(unit), vec<int>(group), n_units, vec<double>(x), tau2));
  });

  m.def("cox_fit", [](const DoubleArray& time, const IntArray& event, const DoubleArray& X, int p) {
    CoxResult r = cox_fit(vec<double>(time), vec<int>(event), vec<double>(X), p);
    py::dict o;
    o["ok"] = r.ok; o["beta"] = arr(r.beta); o["se"] = arr(r.se); o["p"] = arr(r.p);
    o["martingale"] = arr(r.martingale); o["loglik"] = r.loglik;
    return o;
  });

  m.def("survival_test", [](const py::dict& rows, const IntArray& unit, int n_units, const DoubleArray& M,
                            const DoubleArray& time, const IntArray& event, const DoubleArray& x) {
    SurvivalResult r = survival_test(rows_from(rows), vec<int>(unit), n_units, vec<double>(M), vec<double>(time), vec<int>(event),
                                     vec<double>(x));
    py::dict o;
    o["ok"] = r.ok; o["reason"] = r.reason; o["score_coef"] = r.score_coef; o["score_se"] = r.score_se;
    o["score_df"] = r.score_df; o["score_p"] = r.score_p; o["log_hr_sd"] = r.log_hr_sd; o["hr_sd"] = r.hr_sd;
    o["hr_se"] = r.hr_se; o["hr_p"] = r.hr_p; o["log_hr_unit"] = r.log_hr_unit; o["tau2"] = r.tau2;
    return o;
  });

  m.def("cauchy_combine", [](const DoubleArray& p) { return cauchy_combine(vec<double>(p)); });

  m.def("max_t", [](const std::vector<std::vector<double>>& influence, const DoubleArray& t, const DoubleArray& df) {
    MaxTResult r = max_t(influence, vec<double>(t), vec<double>(df));
    return py::make_tuple(r.p, r.best);
  });

  m.def("davies_max", [](const std::vector<std::vector<double>>& influence, const DoubleArray& t, const DoubleArray& df) {
    MaxTResult r = davies_max(influence, vec<double>(t), vec<double>(df));
    return py::make_tuple(r.p, r.best);
  });

  m.def("mvn_outside", [](const DoubleArray& lower, const DoubleArray& upper, const DoubleArray& R) {
    return mvn_outside(vec<double>(lower), vec<double>(upper), vec<double>(R), static_cast<int>(lower.size()));
  });
}
