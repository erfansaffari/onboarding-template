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
  std::vector<double> data_;

public:
  Grid(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols) {
  }

  double& operator()(std::size_t i, std::size_t j) {
    return data_[i * cols_ + j];
  }

  double operator()(std::size_t i, std::size_t j) const {
    return data_[i * cols_ + j];
  }

  std::size_t rows() const {
    return rows_;
  }

  std::size_t cols() const {
    return cols_;
  }

  // Gives the stencil direct access to the contiguous storage.
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

  if (rows == 0 || cols == 0) {
    return;
  }

  const double* old_data = old_grid.data();
  double* new_data = new_grid.data();

  // Small grids have no interior points, so everything is boundary.
  if (rows < 3 || cols < 3) {
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        new_grid(i, j) = old_grid(i, j);
      }
    }
    return;
  }

  // Top and bottom rows are contiguous, so copy them as blocks.
  std::memcpy(
    new_data,
    old_data,
    cols * sizeof(double)
  );

  std::memcpy(
    new_data + (rows - 1) * cols,
    old_data + (rows - 1) * cols,
    cols * sizeof(double)
  );

  // Each output row is independent, so rows can be split across threads.
  #pragma omp parallel for schedule(static)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    const double* top = old_data + (i - 1) * cols;
    const double* mid = old_data + i * cols;
    const double* bottom = old_data + (i + 1) * cols;
    double* out = new_data + i * cols;

    // Left and right boundaries stay unchanged.
    out[0] = mid[0];
    out[cols - 1] = mid[cols - 1];

    // Consecutive columns are consecutive in memory, which is SIMD-friendly.
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