#include <string>

#include <torch/extension.h>

#include "lptorch.cpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("linear_quantize", &linear_quantize);
  m.def("linear_quantize_sr", &linear_quantize_sr);
  m.def("linear_hysteresis", &linear_hysteresis);

  m.def("custom_fp_quantize", &custom_fp_quantize);
  m.def("custom_fp_quantize_sr", &custom_fp_quantize_sr);
  m.def("custom_fp_hysteresis", &custom_fp_hysteresis);

  m.def("fp_quantize", &fp_quantize);
  m.def("fp_quantize_sr", &fp_quantize_sr);
  m.def("fp_hysteresis", &fp_hysteresis);

  m.def("log4_trim_mantissa", &log4_trim_mantissa);
}
