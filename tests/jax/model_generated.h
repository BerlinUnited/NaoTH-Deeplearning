#include "emitc/std.h"
#include "emitc/mhlo.h"
Tensor<float, 50> main(Tensor<float, 50> v1) {
  Tensor<bool> v2 = {false};
  Tensor<bool> v3 = {false};
  Tensor<float> v4 = {(float)0.0e+00};
  Tensor<float, 50> v5 = {(float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00, (float)0.0e+00};
  Tensor<float, 50> v6 = emitc::mhlo::max(v1, v5);
  return v6;
}