// Python bindings for the spicyR C++ core (src/core, synced from spicyR; never edited here).
// They mirror spicyR's R bindings (src/glm_bindings.cpp, src/stats_bindings.cpp) one for one: convert,
// call the core, return. No statistics live in this file.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "spicyglm/core.hpp"
#include "spicyglm/stats.hpp"
#include "spicyglm/stream.hpp"

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
// The loop over images and the passes over the codes live in the core (spicyglm/stream.hpp), shared with spicyR;
// these wrappers only describe NumPy arrays to it and convert the results.

// A 1-d NumPy array (native byte order; the caller converts others) as a core column, without a copy.
spicyglm::Column np_column(const py::array& a, const char* what) {
  if (a.ndim() != 1) throw std::invalid_argument(std::string(what) + ": expected a 1-d array");
  const char kind = a.dtype().kind();
  const py::ssize_t w = a.itemsize();
  spicyglm::Column c;
  using T = spicyglm::Column::Type;
  if (kind == 'f' && w == 8) c.type = T::F64;
  else if (kind == 'f' && w == 4) c.type = T::F32;
  else if (kind == 'i' && w == 8) c.type = T::I64;
  else if (kind == 'i' && w == 4) c.type = T::I32;
  else if (kind == 'i' && w == 2) c.type = T::I16;
  else if (kind == 'i' && w == 1) c.type = T::I8;
  else throw std::invalid_argument(std::string(what) + ": expected float32/float64 or signed integers");
  if (a.size() > 1 && a.strides(0) % w != 0) throw std::invalid_argument(std::string(what) + ": misaligned strides");
  c.data = a.data();
  c.size = static_cast<std::size_t>(a.size());
  c.stride = a.size() > 1 ? a.strides(0) / w : 1;
  return c;
}

// Codes (pandas / Zarr categorical codes, -1 missing) as core codes.
spicyglm::Codes np_codes(const py::array& a, const char* what) {
  if (a.dtype().kind() != 'i') throw std::invalid_argument(std::string(what) + ": codes must be signed integers");
  spicyglm::Codes c;
  c.column = np_column(a, what);
  return c;
}

py::array_t<long long> np_matrix(const std::vector<long long>& v, std::size_t rows, std::size_t cols) {
  py::array_t<long long> a({rows, cols});
  std::copy(v.begin(), v.end(), a.mutable_data());
  return a;
}

// One pass over the cells: for every (image code, type code) the number of cells and the first row (-1 when there is
// none), and for every image code its last row. A missing type (code -1) is the extra last column.
py::dict cell_layout(const py::array& image_codes, int n_image_codes, const py::array& type_codes, int n_type_codes) {
  const spicyglm::Codes im = np_codes(image_codes, "image"), ty = np_codes(type_codes, "cell type");
  spicyglm::CodeScan s;
  {
    py::gil_scoped_release release;
    s = spicyglm::scan_codes(im, n_image_codes, ty, n_type_codes);
  }
  py::dict out;
  out["counts"] = np_matrix(s.counts, n_image_codes, n_type_codes + 1);
  out["first"] = np_matrix(s.first, n_image_codes, n_type_codes + 1);
  out["last"] = arr(s.last);
  return out;
}

// The rows (ascending) of the images selected by `keep` (indexed by image code).
py::array_t<long long> np_rows_of_images(const py::array& image_codes, const py::array_t<bool, py::array::c_style | py::array::forcecast>& keep) {
  const spicyglm::Codes im = np_codes(image_codes, "image");
  std::vector<long long> rows;
  {
    py::gil_scoped_release release;
    rows = spicyglm::rows_of_images(im, keep.data(), static_cast<std::size_t>(keep.size()));
  }
  return arr(rows);
}

// Whether the values (numbers, compared by their bits) are the same in every row of each image as in the image's first
// row (first_row: per image code).
bool np_constant_within(const py::array& values, const py::array& image_codes, const py::array_t<long long, py::array::c_style | py::array::forcecast>& first_row) {
  const spicyglm::Codes im = np_codes(image_codes, "image");
  if (values.ndim() != 1 || static_cast<std::size_t>(values.size()) != im.column.size || !(values.flags() & py::array::c_style))
    throw std::invalid_argument("values: expected a contiguous 1-d array as long as the codes");
  py::gil_scoped_release release;
  return spicyglm::constant_within(values.data(), static_cast<std::size_t>(values.itemsize()), im, first_row.data());
}

// The neighbour totals of the images in a batch of rows (input order; every image present whole), written into the
// full per-image arrays out_a / out_b (one per radius, or one for kNN): any and self_expected (allocation), or totals
// and sq (count), image-major. image_map / type_map (by code; type_map's last entry is the missing type) give the
// image (in the analysis' order) and cell type. Each image runs on its own, on up to n_threads threads.
void stream_batch(const py::array& x, const py::array& y, const py::array& image_codes, const py::array& type_codes,
                  const IntArray& image_map, const IntArray& type_map, int n_types, const DoubleArray& radii, int k,
                  bool allocation, const py::list& out_a, const py::list& out_b, int n_threads) {
  const spicyglm::CellColumns rows{np_column(x, "x"), np_column(y, "y"), np_codes(image_codes, "image"),
                                   np_codes(type_codes, "cell type")};
  spicyglm::StreamOptions o;
  o.radii = vec<double>(radii); o.k = k; o.allocation = allocation; o.n_types = n_types;
  if (out_a.size() != o.n_results() || out_b.size() != o.n_results())
    throw std::invalid_argument("one output array per radius");
  spicyglm::StreamOutput out;
  std::size_t n_images = 0;
  for (std::size_t j = 0; j < o.n_results(); ++j) {
    auto pa = out_a[j].cast<py::array_t<double, py::array::c_style>>();
    auto pb = out_b[j].cast<py::array_t<double, py::array::c_style>>();
    n_images = static_cast<std::size_t>(pa.size()) / o.block_a();
    if (static_cast<std::size_t>(pa.size()) != n_images * o.block_a() || static_cast<std::size_t>(pb.size()) != n_images * o.block_b())
      throw std::invalid_argument("output arrays of the wrong size");
    out.a.push_back(pa.mutable_data()); out.b.push_back(pb.mutable_data());
  }
  const spicyglm::CodeMaps maps{image_map.data(), static_cast<std::size_t>(image_map.size()), type_map.data(),
                                static_cast<std::size_t>(type_map.size()) - 1, static_cast<int>(n_images)};
  py::gil_scoped_release release;
  spicyglm::rows_totals(rows, maps, o, out, n_threads);
}

// spicy()'s two-condition test of every pair at every radius (or at the one k) in one call (core: pair_tests), as
// spicyR's stats_pair_tests(). a, b: one array per radius (allocation: any totals and self_expected; count: totals
// and sq); counts: images x types; psi: types x images, or empty for psi = 1; from / to: the pairs (0-based types);
// image_group: 0 or 1, -1 when missing; z: images x n_z (centred, NaN where missing), or empty; keep_radius: 0-based.
// Per pair and radius the results are (P, K) arrays (effects: (P, K, 1 + n_z), the abundance then the covariates);
// best is 0-based, -1 for none.
py::dict pair_tests_np(const py::list& a, const py::list& b, bool allocation, bool knn, const DoubleArray& counts,
                       int n_types, const DoubleArray& psi, const IntArray& from, const IntArray& to,
                       const IntArray& image_unit, const IntArray& image_group, int n_units, bool image_heterogeneity,
                       bool frailty, const std::string& variance, bool adjust, const DoubleArray& z, int n_z,
                       int keep_radius, const std::string& combine, int n_threads) {
  const std::size_t K = a.size(), T = static_cast<std::size_t>(n_types);
  if (K < 1 || b.size() != K) throw std::invalid_argument("pair_tests: one a and one b per radius");
  if (n_types < 1 || counts.size() % n_types != 0) throw std::invalid_argument("pair_tests: counts must be images x types");
  const std::size_t n_images = static_cast<std::size_t>(counts.size()) / T;
  // the arrays (converted once, if they need it) live through the call
  std::vector<DoubleArray> av, bv;
  PairTestInput in;
  for (std::size_t k = 0; k < K; ++k) {
    av.push_back(a[k].cast<DoubleArray>()); bv.push_back(b[k].cast<DoubleArray>());
    if (static_cast<std::size_t>(av.back().size()) != n_images * T * T ||
        static_cast<std::size_t>(bv.back().size()) != n_images * T * (allocation ? 1 : T))
      throw std::invalid_argument("pair_tests: neighbour totals of the wrong size");
    in.a.push_back(av.back().data()); in.b.push_back(bv.back().data());
  }
  if (psi.size() > 0) {
    if (static_cast<std::size_t>(psi.size()) != T * n_images) throw std::invalid_argument("pair_tests: psi must be types x images");
    in.psi = psi.data();
  }
  if (n_z > 0) {
    if (static_cast<std::size_t>(z.size()) != n_images * n_z) throw std::invalid_argument("pair_tests: z must be images x n_z");
    in.z = z.data(); in.n_z = n_z;
  }
  if (from.size() != to.size()) throw std::invalid_argument("pair_tests: from and to differ in length");
  for (py::ssize_t i = 0; i < from.size(); ++i)
    if (from.data()[i] < 0 || from.data()[i] >= n_types || to.data()[i] < 0 || to.data()[i] >= n_types)
      throw std::invalid_argument("pair_tests: cell type out of range");
  if (static_cast<std::size_t>(image_unit.size()) != n_images || static_cast<std::size_t>(image_group.size()) != n_images)
    throw std::invalid_argument("pair_tests: one unit and one group per image");
  if (keep_radius < 0 || static_cast<std::size_t>(keep_radius) >= K) throw std::invalid_argument("pair_tests: keep_radius out of range");
  in.allocation = allocation; in.knn = knn; in.counts = counts.data(); in.n_types = n_types;
  in.n_images = static_cast<int>(n_images);
  in.from = vec<int>(from); in.to = vec<int>(to);
  in.image_unit = image_unit.data(); in.image_group = image_group.data(); in.n_units = n_units;
  in.image_heterogeneity = image_heterogeneity; in.frailty = frailty;
  in.variance = variance == "hartung_knapp" ? Variance::HartungKnapp : Variance::CR2;
  in.adjust = adjust; in.keep_radius = keep_radius; in.n_threads = n_threads;
  in.combine = combine == "maxT" ? Combine::MaxT : combine == "davies" ? Combine::Davies
             : combine == "cauchy" ? Combine::Cauchy : Combine::None;
  PairTests r;
  {
    py::gil_scoped_release release;
    r = pair_tests(in);
  }
  const std::size_t P = static_cast<std::size_t>(r.n_pairs), ne = 1 + static_cast<std::size_t>(r.n_z);
  auto pk = [&](const auto& v) {
    using V = typename std::decay_t<decltype(v)>::value_type;
    py::array_t<V> x({P, K});
    std::copy(v.begin(), v.end(), x.mutable_data());
    return x;
  };
  auto pke = [&](const std::vector<double>& v) {
    py::array_t<double> x({P, K, ne});
    std::copy(v.begin(), v.end(), x.mutable_data());
    return x;
  };
  py::dict out;
  out["ok"] = pk(r.ok); out["side"] = pk(r.side); out["n_rows"] = pk(r.n_rows); out["tau2_image"] = pk(r.tau2_image);
  out["coef_ref"] = pk(r.coef_ref); out["coef_comp"] = pk(r.coef_comp); out["difference"] = pk(r.difference);
  out["se"] = pk(r.se); out["df"] = pk(r.df); out["p"] = pk(r.p); out["tau2"] = pk(r.tau2);
  out["u_difference"] = pk(r.u_difference); out["u_se"] = pk(r.u_se); out["u_df"] = pk(r.u_df); out["u_p"] = pk(r.u_p);
  out["design"] = pk(r.design); out["abundance"] = pk(r.abundance);
  out["design_reason"] = py::cast(r.design_reason);   // a list, pair-major (p * K + k)
  out["effect"] = pke(r.effect); out["effect_p"] = pke(r.effect_p);
  py::list rows, weight;
  for (std::size_t p = 0; p < P; ++p) { rows.append(rows_to(r.rows[p])); weight.append(arr(r.image_weight[p])); }
  out["rows"] = rows; out["image_weight"] = weight;
  out["best"] = arr(r.best); out["comb_p"] = arr(r.comb_p); out["comb_p_best"] = arr(r.comb_p_best);
  out["comb_u_p"] = arr(r.comb_u_p);
  return out;
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
  m.def("rows_of_images", &np_rows_of_images, py::arg("image_codes"), py::arg("keep"));
  m.def("constant_within", &np_constant_within, py::arg("values"), py::arg("image_codes"), py::arg("first_row"));
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
  m.def("informative_units", [](const py::dict& rows, const IntArray& unit, const IntArray& group, int n_groups) {
    return arr(informative_units(rows_from(rows), vec<int>(unit), vec<int>(group), n_groups));
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

  m.def("pair_tests", &pair_tests_np, py::arg("a"), py::arg("b"), py::arg("allocation"), py::arg("knn"),
        py::arg("counts"), py::arg("n_types"), py::arg("psi"), py::arg("from_"), py::arg("to"), py::arg("image_unit"),
        py::arg("image_group"), py::arg("n_units"), py::arg("image_heterogeneity"), py::arg("frailty"),
        py::arg("variance"), py::arg("adjust"), py::arg("z"), py::arg("n_z"), py::arg("keep_radius"),
        py::arg("combine"), py::arg("n_threads") = 1);

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

  m.def("mvn_outside_qmc", [](const DoubleArray& lower, const DoubleArray& upper, const DoubleArray& R, double abseps,
                              long max_points) {
    MvnResult r = mvn_outside_qmc(vec<double>(lower), vec<double>(upper), vec<double>(R), static_cast<int>(lower.size()),
                                  abseps, max_points);
    return py::make_tuple(r.p, r.error, r.points);
  }, py::arg("lower"), py::arg("upper"), py::arg("R"), py::arg("abseps") = 1e-4, py::arg("max_points") = 100000);

  m.def("mvn_outside_quadrature", [](const DoubleArray& lower, const DoubleArray& upper, const DoubleArray& R) {
    return mvn_outside_quadrature(vec<double>(lower), vec<double>(upper), vec<double>(R), static_cast<int>(lower.size()));
  });
}
