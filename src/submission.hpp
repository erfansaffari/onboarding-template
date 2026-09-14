#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;

  // Keep a little extra space between rows.
  std::size_t stride_;

  std::vector<double> data_;

public:
  Grid(std::size_t rows, std::size_t cols)
    : rows_(rows),
      cols_(cols),
      stride_(cols == 0 ? 0 : cols + 8),
      data_(rows * stride_) {
  }

  double& operator()(std::size_t i, std::size_t j) {
    return data_[i * stride_ + j];
  }

  double operator()(std::size_t i, std::size_t j) const {
    return data_[i * stride_ + j];
  }

  std::size_t rows() const {
    return rows_;
  }

  std::size_t cols() const {
    return cols_;
  }

  std::size_t stride() const {
    return stride_;
  }

  double* data() {
    return data_.data();
  }

  const double* data() const {
    return data_.data();
  }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();
  const std::size_t stride = old_grid.stride();

  if (rows == 0 || cols == 0) {
    return;
  }

  // Input and output use different buffers in the harness.
  const double* __restrict old_data = old_grid.data();
  double* __restrict new_data = new_grid.data();

  // No interior cells in really small grids.
  if (rows < 3 || cols < 3) {
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        new_grid(i, j) = old_grid(i, j);
      }
    }
    return;
  }

  // Copy the top row.
  std::memcpy(
    new_data,
    old_data,
    cols * sizeof(double)
  );

  // Copy the bottom row.
  std::memcpy(
    new_data + (rows - 1) * stride,
    old_data + (rows - 1) * stride,
    cols * sizeof(double)
  );

  #pragma omp parallel for schedule(static)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    const double* top = old_data + (i - 1) * stride;
    const double* mid = old_data + i * stride;
    const double* bottom = old_data + (i + 1) * stride;
    double* out = new_data + i * stride;


    out[0] = mid[0];
    out[cols - 1] = mid[cols - 1];

    #pragma omp simd
    for (std::size_t j = 1; j < cols - 1; ++j) {
      out[j] =
          0.5 * mid[j] +
          0.125 * (
              top[j] +
              bottom[j] +
              mid[j - 1] +
              mid[j + 1]
          );
    }
  }
}