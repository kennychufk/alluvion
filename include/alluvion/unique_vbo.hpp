#ifndef ALLUVION_UNIQUE_VBO_HPP
#define ALLUVION_UNIQUE_VBO_HPP

namespace alluvion {
class UniqueVbo {
 public:
  UniqueVbo() = delete;
  UniqueVbo(GLuint vbo);
  UniqueVbo(const UniqueVbo &) = delete;
  virtual ~UniqueVbo();
  GLuint vbo_;
};
}  // namespace alluvion

#endif /* ALLUVION_UNIQUE_VBO_HPP */
