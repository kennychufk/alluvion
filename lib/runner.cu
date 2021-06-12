#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include "alluvion/runner.hpp"

namespace alluvion {
Runner::Runner() : launch_cursor_(0) {
  Allocator::abort_if_error(cudaEventCreate(&abs_start_));
  Allocator::abort_if_error(cudaEventRecord(abs_start_));
};
Runner::~Runner() { cudaEventDestroy(abs_start_); };
void Runner::export_stat() const {
  std::ofstream stream("kernel-stat.yaml", std::ios::trunc);
  for (auto const& item : launch_stat_dict_) {
    stream << item.first << ": ";
    stream << "[";
    for (auto const& mean : item.second) {
      stream << std::setprecision(std::numeric_limits<float>::max_digits10)
             << mean << ", ";
    }
    stream << "]" << std::endl;
  }
}
void Runner::load_stat() {
  std::ifstream stream("kernel-stat.yaml");
  std::string line;
  std::stringstream line_stream;
  std::string function_name;
  std::string elapsed_str;
  stream.exceptions(std::ios_base::badbit);
  while (std::getline(stream, line)) {
    line_stream.clear();
    line_stream.str(line);
    std::getline(line_stream, function_name, ':');
    std::getline(line_stream, elapsed_str, '[');
    U optimal_index;
    float min_elapsed = kFMax<float>;
    for (U i = 0; i < kNumBlockSizeCandidates &&
                  std::getline(line_stream, elapsed_str, ',');
         ++i) {
      float elapsed = from_string<float>(elapsed_str);
      if (elapsed < min_elapsed) {
        min_elapsed = elapsed;
        optimal_index = i;
      }
    }
    optimal_block_size_dict_[function_name] = (optimal_index + 1) * kWarpSize;
  }
}
}  // namespace alluvion
