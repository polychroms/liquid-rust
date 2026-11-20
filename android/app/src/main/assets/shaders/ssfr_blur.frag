#version 300 es
precision highp float;

in vec2 v_uv;

uniform sampler2D t_depth;
uniform vec2 u_texel_size;
uniform vec2 u_direction;
uniform float u_fusion;
uniform float u_tap_stride;
uniform float u_depth_threshold;

out vec4 fragColor;

float unpackDepthRGB(vec3 rgb) {
    return rgb.r + rgb.g / 255.0 + rgb.b / 65025.0;
}

vec3 packDepthRGB(float v) {
    float c = clamp(v, 0.0, 0.999999);
    float r = floor(c * 255.0);
    float rem = c * 255.0 - r;
    float g = floor(rem * 255.0);
    float b = floor((rem * 255.0 - g) * 255.0);
    return vec3(r, g, b) / 255.0;
}

void main() {
    vec4 c = texture(t_depth, v_uv);
    if (c.a < 0.5) {
        fragColor = vec4(0.0);
        return;
    }

    float centerD = unpackDepthRGB(c.rgb);
    float sum = centerD;
    float wSum = 1.0;

    float blurStrength = smoothstep(0.02, 1.0, u_fusion);
    float thresh = max(u_depth_threshold, 0.0005);
    float inv2sd2 = 1.0 / (2.0 * thresh * thresh);
    float inv2ss2 = 1.0 / (2.0 * 2.6 * 2.6);
    vec2 stepUv = u_direction * u_texel_size * u_tap_stride;

    for (int i = -4; i <= 4; i++) {
        if (i == 0) continue;
        float fi = float(i) * blurStrength;
        vec2 suv = clamp(v_uv + stepUv * fi, vec2(0.0005), vec2(0.9995));
        vec4 s = texture(t_depth, suv);
        if (s.a > 0.5) {
            float sd = unpackDepthRGB(s.rgb);
            float dd = sd - centerD;
            float w = exp(-(fi * fi) * inv2ss2 - (dd * dd) * inv2sd2);
            sum += sd * w;
            wSum += w;
        }
    }

    fragColor = vec4(packDepthRGB(sum / wSum), 1.0);
}
