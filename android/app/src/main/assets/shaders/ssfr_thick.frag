#version 300 es
precision highp float;

in vec2 v_uv;
in vec3 v_pos_eye;
in float v_speed_t;
in float v_metric_t;

uniform float u_thick_weight;

out vec4 fragColor;

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) discard;

    float z = sqrt(max(1.0 - r2, 0.0));
    float w = pow(z, 1.35) * u_thick_weight;
    fragColor = vec4(w, w * v_metric_t, w * v_speed_t, 0.0);
}
