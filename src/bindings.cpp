// Python bindings for the spicyR C++ core (src/core, synced from spicyR; never edited here).
// They mirror spicyR's R bindings (src/glm_bindings.cpp, src/stats_bindings.cpp) one for one: convert,
// call the core, return. No statistics live in this file.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
      .def("self_any_expected", [](const Dataset& d, bool knn) { return arr(d.self_any_expected(knn)); });

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

  m.def("mvn_outside", [](const DoubleArray& lower, const DoubleArray& upper, const DoubleArray& R) {
    return mvn_outside(vec<double>(lower), vec<double>(upper), vec<double>(R), static_cast<int>(lower.size()));
  });
}
