
#include "tensor.hpp"
#include "transformations.hpp"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("local_to_world_transform", &local_to_world_transform,
        "Create transformation matrix");
  m.def("local_to_local_transform", &local_to_local_transform,
        "Homogeneous Transformation");
  m.def("apply_transformation", &apply_transformation,
        "Applies a transformation matrix to a point cloud");
  m.def("change_sparse_representation", &change_sparse_representation,
        "Changes the representation of sparse tensors.");
}
