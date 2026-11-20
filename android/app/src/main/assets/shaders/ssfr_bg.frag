#version 300 es
precision highp float;

in vec2 v_uv;

uniform float u_time;
uniform vec3 u_water_color;
uniform int u_is_2d;
uniform int u_light_mode;

out vec4 fragColor;

float causticWeb(vec2 uv, float t) {
    vec2 p = uv * 5.0;
    float a = sin(p.x * 1.4 + t * 1.15 + sin(p.y * 1.6 - t * 0.95));
    float b = sin(p.y * 1.5 - t * 1.05 + cos(p.x * 1.3 + t * 1.25));
    float c = sin((p.x + p.y) * 1.1 + t * 0.8);
    return pow(clamp(1.0 - abs(a + b + c * 0.65) * 0.65, 0.0, 1.0), 3.6);
}

void main() {
    // Material 3 Slate / Studio Daylight backdrop with subtle caustic floor refraction cues
    vec3 low  = (u_light_mode == 1) ? vec3(0.790, 0.845, 0.920) : vec3(0.036, 0.055, 0.084);
    vec3 high = (u_light_mode == 1) ? vec3(0.925, 0.955, 0.985) : vec3(0.072, 0.108, 0.162);
    vec3 bg = mix(low, high, smoothstep(0.0, 1.0, v_uv.y));

    // Subtle soft studio grid pattern so background refraction bending is visibly crisp
    vec2 g = abs(fract(v_uv * vec2(12.0, 20.0)) - 0.5);
    float gridLine = smoothstep(0.47, 0.50, max(g.x, g.y));
    vec3 gridTint = (u_light_mode == 1) ? vec3(-0.085, -0.068, -0.042) : vec3(0.025, 0.042, 0.068);
    bg += gridTint * gridLine * (u_is_2d == 1 ? 0.55 : 0.35);

    float floorMask = smoothstep(0.60, 0.04, v_uv.y);
    float c = causticWeb(v_uv * vec2(1.3, 1.8), u_time) * 0.60
            + causticWeb(v_uv * vec2(1.9, 2.4) + vec2(0.37, -0.21), -u_time * 0.8) * 0.40;
    vec3 cTint = (u_light_mode == 1) ? vec3(0.28, 0.64, 0.92) : mix(u_water_color, vec3(0.75, 0.94, 1.0), 0.35);
    bg += cTint * c * floorMask * ((u_light_mode == 1) ? 0.060 : 0.085);

    fragColor = vec4(bg, 1.0);
}
