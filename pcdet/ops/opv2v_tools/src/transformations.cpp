#include "transformations.hpp"
#include "label.hpp"
#include "name.hpp"
#include "stats.hpp"
#include "utils.hpp"
#include <ATen/ops/index_select.h>
#include <algorithm>
#include <cmath>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/types.h>

using Slice = torch::indexing::Slice;
using namespace torch_utils;

[[nodiscard]] torch::Tensor
local_to_world_transform(const torch::Tensor &lidar_pose) {

  auto transformation = torch::eye(4);

  // translations
  transformation.index_put_({0, 3}, lidar_pose[0]);
  transformation.index_put_({1, 3}, lidar_pose[1]);
  transformation.index_put_({2, 3}, lidar_pose[2]);

  // rotations
  const auto cos_roll = lidar_pose[3].deg2rad().cos();
  const auto sin_roll = lidar_pose[3].deg2rad().sin();
  const auto cos_yaw = lidar_pose[4].deg2rad().cos();
  const auto sin_yaw = lidar_pose[4].deg2rad().sin();
  const auto cos_pitch = lidar_pose[5].deg2rad().cos();
  const auto sin_pitch = lidar_pose[5].deg2rad().sin();

  transformation.index_put_({2, 0}, sin_pitch);

  transformation.index_put_({0, 0}, cos_pitch * cos_yaw);
  transformation.index_put_({1, 0}, sin_yaw * cos_pitch);
  transformation.index_put_({2, 1}, -cos_pitch * sin_roll);
  transformation.index_put_({2, 2}, cos_pitch * cos_roll);

  transformation.index_put_({0, 1}, cos_yaw * sin_pitch * sin_roll -
                                        sin_yaw * cos_roll);
  transformation.index_put_({0, 2}, -cos_yaw * sin_pitch * cos_roll -
                                        sin_yaw * sin_roll);
  transformation.index_put_({1, 1}, sin_yaw * sin_pitch * sin_roll +
                                        cos_yaw * cos_roll);
  transformation.index_put_({1, 2}, -sin_yaw * sin_pitch * cos_roll +
                                        cos_yaw * sin_roll);

  return transformation;
}

[[nodiscard]] torch::Tensor
local_to_local_transform(const torch::Tensor &from_pose,
                         const torch::Tensor &to_pose) {

  auto local_to_world = local_to_world_transform(from_pose);
  auto world_to_local = torch::linalg_inv(local_to_world_transform(to_pose));

  return world_to_local.mm(local_to_world);
}

void apply_transformation(torch::Tensor points,
                          const torch::Tensor &transformation_matrix) {

  // save intensity values
  const auto intensity =
      torch::index_select(points, 2, torch::tensor({POINT_CLOUD_I_IDX}))
          .flatten(0);

  // apply transformation
  points.dot(transformation_matrix.permute({1, 0}));

  // write back intensity
  points.index({Slice(), Slice(), POINT_CLOUD_I_IDX}) = intensity;
}

#ifdef BUILD_MODULE
#undef TEST_RNG
#include "transformation_bindings.hpp"
#else
#include "gtest/gtest.h"

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
