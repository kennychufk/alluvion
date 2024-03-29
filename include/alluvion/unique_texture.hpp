#ifndef ALLUVION_UNIQUE_TEXTURE_HPP
#define ALLUVION_UNIQUE_TEXTURE_HPP

namespace alluvion {
class UniqueTexture {
 public:
  UniqueTexture() = delete;
  UniqueTexture(GLuint tex);
  UniqueTexture(const UniqueTexture &) = delete;
  virtual ~UniqueTexture();
  GLuint tex_;
};
}  // namespace alluvion

#endif /* ALLUVION_UNIQUE_TEXTURE_HPP */
