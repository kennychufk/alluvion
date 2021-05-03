#ifndef ALLUVION_DG_SPINLOCK_HPP
#define ALLUVION_DG_SPINLOCK_HPP

#include <atomic>

namespace alluvion {
namespace dg {
class SpinLock {
 public:
  void lock() {
    while (m_flag.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() { m_flag.clear(std::memory_order_release); }

 private:
  std::atomic_flag m_flag = ATOMIC_FLAG_INIT;
};
}  // namespace dg
}  // namespace alluvion
#endif
