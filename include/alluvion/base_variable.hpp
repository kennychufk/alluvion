#ifndef ALLUVION_BASE_VARIABLE_HPP
#define ALLUVION_BASE_VARIABLE_HPP

namespace alluvion {
class BaseVariable {
 public:
  BaseVariable(){};
  virtual ~BaseVariable(){};
  virtual void set_pointer(void* ptr) = 0;
};
}  // namespace alluvion

#endif /* ALLUVION_BASE_VARIABLE_HPP */
