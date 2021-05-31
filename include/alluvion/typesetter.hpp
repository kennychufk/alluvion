#ifndef ALLUVION_TYPESETTER_HPP
#define ALLUVION_TYPESETTER_HPP

#include <ft2build.h>
#include FT_FREETYPE_H

#include <unordered_map>

#include "alluvion/display.hpp"
#include "alluvion/glyph_attr.hpp"

namespace alluvion {
class Typesetter {
 private:
  FT_Library lib_;
  FT_Face face_;
  std::unordered_map<unsigned char, GlyphAttr> glyphs_;
  Display* display_;

  float text_x_;
  float text_y_;
  float scale_;

 public:
  Typesetter(Display* display, const char* font_filename, unsigned int width,
             unsigned int height);
  virtual ~Typesetter();
  void load_char(unsigned char c);
  void load_ascii();
  void start(float text_x, float text_y, float scale);
  GLuint place_glyph(unsigned char c);
  GlyphAttr const* get_glyph_attr(unsigned char c) const;

  float vertices_info_[6][4];
};
}  // namespace alluvion

#endif /* ALLUVION_TYPESETTER_HPP */
