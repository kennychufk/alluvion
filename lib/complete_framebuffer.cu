#include <cstring>
#include <fstream>

#include "alluvion/complete_framebuffer.hpp"
#include "alluvion/graphical_allocator.hpp"

namespace alluvion {
CompleteFramebuffer::CompleteFramebuffer(GLsizei width, GLsizei height)
    : fbo_(GraphicalAllocator::allocate_framebuffer()),
      color_tex_(
          GraphicalAllocator::allocate_texture2d(nullptr, width, height)),
      depth_stencil_rbo_(
          GraphicalAllocator::allocate_render_buffer(width, height)),
      width_(width),
      height_(height) {
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glBindTexture(GL_TEXTURE_2D, color_tex_);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         color_tex_, 0);

  glBindRenderbuffer(GL_RENDERBUFFER, depth_stencil_rbo_);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                            GL_RENDERBUFFER, depth_stencil_rbo_);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cerr << "ERROR::FRAMEBUFFER:: Framebuffer is not complete."
              << std::endl;
    abort();
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

CompleteFramebuffer::CompleteFramebuffer(CompleteFramebuffer&& other) noexcept
    : fbo_(std::move(other.fbo_)),
      color_tex_(std::move(other.color_tex_)),
      depth_stencil_rbo_(std::move(other.depth_stencil_rbo_)),
      width_(std::move(other.width_)),
      height_(std::move(other.height_)) {
  other.fbo_ = 0;
  other.color_tex_ = 0;
  other.depth_stencil_rbo_ = 0;
}

CompleteFramebuffer::~CompleteFramebuffer() {
  GraphicalAllocator::free_framebuffer(&fbo_);
  GraphicalAllocator::free_texture(&color_tex_);
  GraphicalAllocator::free_render_buffer(&depth_stencil_rbo_);
}

void CompleteFramebuffer::resize(GLsizei width, GLsizei height) {
  width_ = width;
  height_ = height;
  GraphicalAllocator::reallocate_render_buffer(depth_stencil_rbo_, width_,
                                               height_);
  GraphicalAllocator::reallocate_texture2d(color_tex_, nullptr, width_,
                                           height_);
}

GLsizei CompleteFramebuffer::get_read_stride() const {
  U tight_stride = kNumChannels * width_;
  return tight_stride + ((tight_stride % 4) ? (4 - tight_stride % 4) : 0);
}

void CompleteFramebuffer::write(const char* filename) {
  resize_host_buffers();
  const GLsizei tight_stride = kNumChannels * width_;
  const GLsizei read_stride = get_read_stride();
  glReadPixels(0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE,
               read_buffer_.data());
  for (int row = 0; row < height_; ++row) {
    std::memcpy(tight_buffer_.data() + (tight_stride * (height_ - row - 1)),
                read_buffer_.data() + (read_stride * row), tight_stride);
  }
  std::ofstream stream(filename, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(tight_buffer_.data()),
               tight_buffer_.size() * sizeof(unsigned char));
}

void CompleteFramebuffer::resize_host_buffers() {
  const GLsizei read_buffer_size = get_read_stride() * height_;
  const GLsizei tight_buffer_size = width_ * height_ * kNumChannels;
  if (read_buffer_.size() == read_buffer_size &&
      tight_buffer_.size() == tight_buffer_size)
    return;
  read_buffer_.resize(read_buffer_size);
  tight_buffer_.resize(tight_buffer_size);
}
}  // namespace alluvion
