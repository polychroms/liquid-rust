#version 300 es
precision highp float;

in vec2 v_uv;
in vec3 v_pos_eye;
in float v_speed_t;
in float v_metric_t;

uniform mat4 u_proj;
uniform float u_radius;
uniform float u_fusion;
uniform float u_depth_scale;

out vec4 fragColor;

vec3 packDepthRGB(float v) {
    float c = clamp(v, 0.0, 0.999999);
    float r = floor(c * 255.0);
    float rem = c * 255.0 - r;
    float g = floor(rem * 255.0);
    float b = floor((rem * 255.0 - g) * 255.0);
    return vec3(r, g, b) / 255.0;
}

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) discard;

    float z = sqrt(max(1.0 - r2, 0.0));
    // Flatten forward-facing sphere domes smoothly as fusion rises to suppress pebble dimples
    float flatZ = mix(1.0, 0.36, smoothstep(0.15, 1.0, u_fusion));
    vec3 sphere_eye = v_pos_eye + vec3(v_uv.x * u_radius, v_uv.y * u_radius, z * u_radius * flatZ);

    vec4 clip = u_proj * vec4(sphere_eye, 1.0);
    float ndcZ = clip.z / clip.w;
    gl_FragDepth = clamp(ndcZ * 0.5 + 0.5, 0.0, 1.0);

    float normD = clamp(-sphere_eye.z / u_depth_scale, 0.0001, 0.999999);
    fragColor = vec4(packDepthRGB(normD), 1.0);
}
