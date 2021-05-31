#ifndef ALLUVION_UNIQUE_VAO_HPP
#define ALLUVION_UNIQUE_VAO_HPP

namespace alluvion {
class UniqueVao {
 public:
  UniqueVao();
  UniqueVao(GLuint vao);
  UniqueVao(const UniqueVao&) = delete;
  virtual ~UniqueVao();
  void set(GLuint vao);
  GLuint vao_;
};
}  // namespace alluvion

#endif /* ALLUVION_UNIQUE_VAO_HPP */
