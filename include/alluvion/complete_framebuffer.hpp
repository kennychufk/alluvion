#ifndef ALLUVION_COMPLETE_FRAMEBUFFER_HPP
#define ALLUVION_COMPLETE_FRAMEBUFFER_HPP
#include <glad/glad.h>
//
#include <vector>

namespace alluvion {
class CompleteFramebuffer {
 public:
  CompleteFramebuffer() = delete;
  CompleteFramebuffer(GLsizei width, GLsizei height);
  CompleteFramebuffer(const CompleteFramebuffer&) = delete;
  CompleteFramebuffer(CompleteFramebuffer&& other) noexcept;
  virtual ~CompleteFramebuffer();
  void resize(GLsizei width, GLsizei height);
  GLsizei get_read_stride() const;
  void write(const char*);

  static constexpr GLsizei kNumChannels = 3;
  GLuint fbo_;
  GLuint color_tex_;
  GLuint depth_stencil_rbo_;
  GLsizei width_;
  GLsizei height_;
  std::vector<unsigned char> tight_buffer_;
  std::vector<unsigned char> read_buffer_;

 private:
  void resize_host_buffers();
};
}  // namespace alluvion
#endif /* ALLUVION_COMPLETE_FRAMEBUFFER_HPP */
