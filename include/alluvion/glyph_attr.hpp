#ifndef ALLUVION_GLYPH_ATTR_HPP
#define ALLUVION_GLYPH_ATTR_HPP
#include <glad/glad.h>
//
#include <glm/gtc/type_ptr.hpp>

namespace alluvion {
struct GlyphAttr {
  GLuint tex;
  glm::ivec2 dimension;
  glm::ivec2 bearing;    // Offset from baseline to left/top of glyph
  unsigned int advance;  // Horizontal offset to advance to next glyph
};
}  // namespace alluvion
#endif
