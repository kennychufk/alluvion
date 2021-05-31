#include <iostream>

#include "alluvion/typesetter.hpp"

namespace alluvion {
Typesetter::Typesetter(Display* display, const char* font_filename,
                       unsigned int width, unsigned int height)
    : display_(display) {
  if (FT_Init_FreeType(&lib_)) {
    std::cerr << "ERROR::FREETYPE: Could not init FreeType Library"
              << std::endl;
    abort();
  }
  if (FT_New_Face(lib_, font_filename, 0, &face_)) {
    std::cerr << "ERROR::FREETYPE: Failed to load font" << std::endl;
    abort();
  }
  // set size to load glyphs as
  FT_Set_Pixel_Sizes(face_, width, height);
  vertices_info_[0][2] = 0.0f;
  vertices_info_[0][3] = 0.0f;
  vertices_info_[1][2] = 0.0f;
  vertices_info_[1][3] = 1.0f;
  vertices_info_[2][2] = 1.0f;
  vertices_info_[2][3] = 1.0f;
  vertices_info_[3][2] = 0.0f;
  vertices_info_[3][3] = 0.0f;
  vertices_info_[4][2] = 1.0f;
  vertices_info_[4][3] = 1.0f;
  vertices_info_[5][2] = 1.0f;
  vertices_info_[5][3] = 0.0f;
}

Typesetter::~Typesetter() {
  FT_Done_Face(face_);
  FT_Done_FreeType(lib_);
}

void Typesetter::load_char(unsigned char c) {
  if (FT_Load_Char(face_, c, FT_LOAD_RENDER)) {
    std::cout << "ERROR::FREETYTPE: Failed to load Glyph for " << c
              << std::endl;
    abort();
  }
  GLuint tex = display_->create_monochrome_texture(face_->glyph->bitmap.buffer,
                                                   face_->glyph->bitmap.width,
                                                   face_->glyph->bitmap.rows);
  GlyphAttr glyph_attr = {
      tex, glm::ivec2(face_->glyph->bitmap.width, face_->glyph->bitmap.rows),
      glm::ivec2(face_->glyph->bitmap_left, face_->glyph->bitmap_top),
      static_cast<unsigned int>(face_->glyph->advance.x)};
  glyphs_.insert(std::pair<unsigned char, GlyphAttr>(c, glyph_attr));
}

void Typesetter::load_ascii() {
  for (unsigned char c = 0; c < 128; c++) {
    load_char(c);
  }
}

GlyphAttr const* Typesetter::get_glyph_attr(unsigned char c) const {
  return &(glyphs_.at(c));
}

void Typesetter::start(float text_x, float text_y, float scale) {
  text_x_ = text_x;
  text_y_ = text_y;
  scale_ = scale;
}

GLuint Typesetter::place_glyph(unsigned char c) {
  GlyphAttr const* glyph_attr = get_glyph_attr(c);
  float xpos = text_x_ + glyph_attr->bearing.x * scale_;
  float ypos =
      text_y_ - (glyph_attr->dimension.y - glyph_attr->bearing.y) * scale_;

  float w = glyph_attr->dimension.x * scale_;
  float h = glyph_attr->dimension.y * scale_;
  vertices_info_[0][0] = xpos;
  vertices_info_[0][1] = ypos + h;
  vertices_info_[1][0] = xpos;
  vertices_info_[1][1] = ypos;
  vertices_info_[2][0] = xpos + w;
  vertices_info_[2][1] = ypos;
  vertices_info_[3][0] = xpos;
  vertices_info_[3][1] = ypos + h;
  vertices_info_[4][0] = xpos + w;
  vertices_info_[4][1] = ypos;
  vertices_info_[5][0] = xpos + w;
  vertices_info_[5][1] = ypos + h;
  // now advance cursors for next glyph (note that advance is number of
  // 1/64 pixels)
  text_x_ += (glyph_attr->advance >> 6) *
             scale_;  // bitshift by 6 to get value in pixels (2^6 = 64 (divide
                      // amount of 1/64th pixels by 64 to get amount of pixels))
  return glyph_attr->tex;
}
}  // namespace alluvion
