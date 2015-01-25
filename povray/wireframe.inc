#declare WireframeTexture = texture {pigment {color rgb <1,1,0>} finish{phong 1}}

#declare WireframeXTexture = texture {
  pigment {
    gradient x
    turbulence 0
    color_map {
      [0     rgbt <0,1,0,0>]
      [0.05  rgbt <0,1,0,0>]
      [0.05  rgbt <1,1,1,1>]
      [0.95  rgbt <1,1,1,1>]
      [0.95  rgbt <0,1,0,0>]
      [1     rgbt <0,1,0,0>]
    }
  scale 1
  }
}

#declare WireframeTexture =
  texture {WireframeXTexture}
  texture {WireframeXTexture rotate 90}
