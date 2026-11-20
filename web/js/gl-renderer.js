// WebGL 2.0 Hardware-Accelerated Studio Renderer for LiquidRust Web
// High-performance point sprite spheres with glossy Fresnel sky-rim, specular highlights,
// true 3D spherical depth, studio floor grid, and zero per-frame JS heap allocations.

import { ColorMap } from './color-map.js?v=20261113b';
import { Mat4 } from './math.js?v=20261113b';

export class GlRenderer {
    constructor(gl) {
        this.gl = gl;
        this.viewProjScratch = new Mat4();
        this.lineScratch = new Float32Array(4096);
        this.isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || window.innerWidth < 768;
        this.ssfrScale = this.isMobile ? 0.75 : 1.0;
        this.ssfrTargets = null;
        this.initShaders();
        this.initSsfrShaders();
        this.initBuffers();
    }

    initShaders() {
        const gl = this.gl;

        // 1. Glossy Liquid Particle Shader (WebGL 2.0 / GLSL 3.00 ES)
        const pVsSource = `#version 300 es
        precision highp float;

        layout(location = 0) in vec3 a_pos;
        layout(location = 1) in vec4 a_color;

        uniform mat4 u_view;
        uniform mat4 u_proj;
        uniform float u_point_radius;
        uniform float u_viewport_height;
        uniform float u_tan_half_fov;

        out vec4 v_color;
        out vec3 v_pos_eye;
        out vec3 v_world_pos;

        void main() {
            v_color = a_color;
            v_world_pos = a_pos;
            vec4 eye_pos = u_view * vec4(a_pos, 1.0);
            v_pos_eye = eye_pos.xyz;
            gl_Position = u_proj * eye_pos;

            float dist = max(-eye_pos.z, 0.001);
            float size_px = (u_point_radius / (dist * u_tan_half_fov)) * u_viewport_height;
            gl_PointSize = clamp(size_px, 2.0, 128.0);
        }
        `;

        const pFsSource = `#version 300 es
        precision highp float;

        in vec4 v_color;
        in vec3 v_pos_eye;
        in vec3 v_world_pos;

        uniform mat4 u_proj;
        uniform float u_point_radius;
        uniform vec3 u_light_dir;
        uniform int u_lit_spheres;
        uniform int u_is_2d;
        uniform int u_color_mode;
        uniform int u_tex_mode;
        uniform float u_tex_strength;
        uniform float u_anim_time;
        uniform vec3 u_shallow_color;
        uniform vec3 u_deep_color;
        uniform vec4 u_optics;
        uniform float u_refraction;
        uniform float u_foam_amount;
        uniform float u_light_mode;

        out vec4 fragColor;

        void main() {
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(coord, coord);
            if (r2 > 1.0) {
                discard;
            }

            float z = sqrt(max(1.0 - r2, 0.0));
            vec3 n = vec3(coord.x, -coord.y, z);
            vec3 surf_pos = v_world_pos + n * u_point_radius;

            if (u_is_2d == 0) {
                vec3 sphere_pos_eye = v_pos_eye + vec3(coord.x * u_point_radius, -coord.y * u_point_radius, z * u_point_radius);
                vec4 clip_pos = u_proj * vec4(sphere_pos_eye, 1.0);
                gl_FragDepth = clamp(((clip_pos.z / clip_pos.w) + 1.0) * 0.5, 0.0, 1.0);
            }

            float speed_t = v_color.a;
            float tex_str = clamp(u_tex_strength, 0.0, 1.0);
            float t_anim = u_anim_time;

            float caustic_boost = 0.0;
            float foam_tex = 0.0;
            float grid_mod = 1.0;

            if (u_tex_mode != 0 && tex_str > 0.005) {
                if (u_tex_mode == 1) {
                    float w1 = cos(surf_pos.x * 28.0 + surf_pos.y * 19.0 - t_anim * 3.8);
                    float w2 = sin(surf_pos.z * 26.0 - surf_pos.x * 16.0 + t_anim * 3.1);
                    n = normalize(n + vec3(w1, w2, 0.0) * (tex_str * 0.42));
                    caustic_boost = max(w1 * w2, 0.0) * tex_str * 0.20;
                } else if (u_tex_mode == 2) {
                    vec3 p = surf_pos * 18.0;
                    float c1 = abs(sin(p.x + sin(p.y * 1.3 + t_anim * 2.4)) * cos(p.y + cos(p.z * 1.4 - t_anim * 1.9)));
                    float web = pow(1.0 - clamp(c1, 0.0, 1.0), 3.0);
                    caustic_boost = web * tex_str * 0.85;
                    n = normalize(n + vec3(sin(p.x * 1.7), cos(p.y * 1.7), 0.0) * (tex_str * 0.25));
                } else if (u_tex_mode == 3) {
                    vec3 p = surf_pos * 24.0 + vec3(t_anim * 1.5, -t_anim * 0.8, t_anim * 1.2);
                    float f = (sin(p.x) * cos(p.y) + sin(p.y * 1.7 + p.z) + cos(p.z * 2.3 - p.x)) * 0.333 + 0.5;
                    foam_tex = clamp((f - 0.38) * 2.6, 0.0, 1.0) * tex_str * (0.28 + 0.72 * speed_t);
                    n = normalize(n + vec3(f - 0.5, (1.0 - f) - 0.5, 0.0) * (tex_str * 0.22));
                } else if (u_tex_mode == 4) {
                    float gx = abs(fract(surf_pos.x * 8.0) - 0.5);
                    float gy = abs(fract(surf_pos.y * 8.0) - 0.5);
                    float gz = abs(fract(surf_pos.z * 8.0) - 0.5);
                    float line = max(1.0 - min(gx, gy) * 14.0, 1.0 - gz * 14.0);
                    float line_c = clamp(line, 0.0, 1.0) * tex_str;
                    grid_mod = 1.0 - tex_str * 0.14;
                    caustic_boost = line_c * 0.45;
                }
            }

            float metallic = clamp(u_optics.w, 0.0, 1.0);
            vec3 base_color = v_color.rgb;
            if (u_color_mode == 3) {
                float rim = 1.0 - z;
                float shallow_mix = clamp(0.28 + 0.48 * speed_t + 0.42 * rim, 0.0, 1.0);
                base_color = mix(u_deep_color, u_shallow_color, shallow_mix);
                float crest_foam = smoothstep(0.62, 0.98, speed_t) * u_foam_amount * (1.0 - metallic);
                base_color = mix(base_color, vec3(0.94, 0.98, 1.0), crest_foam * 0.72);
            }

            base_color = base_color * grid_mod + mix(u_shallow_color, vec3(0.96, 0.99, 1.0), 0.35) * caustic_boost;
            if (foam_tex > 0.005) {
                base_color = mix(base_color, vec3(0.95, 0.98, 1.0), clamp(foam_tex, 0.0, 0.85));
            }

            vec3 rgb = base_color;
            if (u_lit_spheres != 0) {
                float ior = max(u_optics.x, 1.01);
                float rough = clamp(u_optics.y, 0.015, 0.95);
                float spec_mult = max(u_optics.z, 0.0);
                float refr_glow = clamp(u_refraction, 0.0, 1.5);

                float f0_diel = pow((ior - 1.0) / (ior + 1.0), 2.0);
                vec3 f0 = mix(vec3(f0_diel), base_color, metallic);
                float ndotv = max(n.z, 0.001);
                vec3 fresnel = f0 + (vec3(1.0) - f0) * pow(1.0 - ndotv, 4.2);

                float ndotl = max(dot(n, u_light_dir), 0.0);
                float hemi = 0.5 + 0.5 * n.y;
                vec3 half_dir = normalize(u_light_dir + vec3(0.0, 0.0, 1.0));
                float spec_pow = clamp(1.8 / (rough * rough + 0.003), 8.0, 420.0);
                float spec_energy = (0.28 + 0.72 * (1.0 - rough)) * spec_mult;
                vec3 spec = pow(max(dot(n, half_dir), 0.0), spec_pow) * spec_energy * fresnel;

                float refl_y = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
                vec3 sky_low = mix(vec3(0.08, 0.14, 0.24), vec3(0.46, 0.64, 0.86), u_light_mode);
                vec3 sky_hi  = mix(vec3(0.55, 0.78, 0.98), vec3(0.92, 0.96, 1.00), u_light_mode);
                vec3 sky_env = mix(sky_low, sky_hi, pow(refl_y, 1.5));
                vec3 body_lit = base_color * (0.28 + 0.18 * hemi + 0.62 * ndotl * (1.0 - 0.65 * metallic));

                float sss_rim = pow(1.0 - ndotv, 2.2) * refr_glow * (1.0 - metallic);
                vec3 sss_tint = u_shallow_color * (0.35 + 0.65 * (0.5 + 0.5 * dot(n, -u_light_dir))) * sss_rim;

                rgb = mix(body_lit, sky_env * mix(vec3(1.0), base_color, metallic), clamp(fresnel * (0.75 + 0.25 * metallic), 0.0, 0.96))
                    + spec
                    + sss_tint;
            }
            fragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
        }
        `;

        this.particleProgram = this.createProgram(pVsSource, pFsSource);

        this.uView = gl.getUniformLocation(this.particleProgram, 'u_view');
        this.uProj = gl.getUniformLocation(this.particleProgram, 'u_proj');
        this.uPointRadius = gl.getUniformLocation(this.particleProgram, 'u_point_radius');
        this.uViewportHeight = gl.getUniformLocation(this.particleProgram, 'u_viewport_height');
        this.uTanHalfFov = gl.getUniformLocation(this.particleProgram, 'u_tan_half_fov');
        this.uLightDir = gl.getUniformLocation(this.particleProgram, 'u_light_dir');
        this.uLitSpheres = gl.getUniformLocation(this.particleProgram, 'u_lit_spheres');
        this.uIs2D = gl.getUniformLocation(this.particleProgram, 'u_is_2d');
        this.uColorMode = gl.getUniformLocation(this.particleProgram, 'u_color_mode');
        this.uTexMode = gl.getUniformLocation(this.particleProgram, 'u_tex_mode');
        this.uTexStrength = gl.getUniformLocation(this.particleProgram, 'u_tex_strength');
        this.uAnimTime = gl.getUniformLocation(this.particleProgram, 'u_anim_time');
        this.uShallowColor = gl.getUniformLocation(this.particleProgram, 'u_shallow_color');
        this.uDeepColor = gl.getUniformLocation(this.particleProgram, 'u_deep_color');
        this.uOptics = gl.getUniformLocation(this.particleProgram, 'u_optics');
        this.uRefraction = gl.getUniformLocation(this.particleProgram, 'u_refraction');
        this.uFoamAmount = gl.getUniformLocation(this.particleProgram, 'u_foam_amount');
        this.uLightMode = gl.getUniformLocation(this.particleProgram, 'u_light_mode');

        // 2. Line Shader (Studio floor grid, chamber box, and colliders)
        const lVsSource = `#version 300 es
        precision highp float;

        layout(location = 0) in vec3 a_pos;
        uniform mat4 u_view_proj;

        void main() {
            gl_Position = u_view_proj * vec4(a_pos, 1.0);
        }
        `;

        const lFsSource = `#version 300 es
        precision highp float;

        uniform vec4 u_line_color;
        out vec4 fragColor;

        void main() {
            fragColor = u_line_color;
        }
        `;

        this.lineProgram = this.createProgram(lVsSource, lFsSource);
        this.uViewProj = gl.getUniformLocation(this.lineProgram, 'u_view_proj');
        this.uLineColor = gl.getUniformLocation(this.lineProgram, 'u_line_color');
    }

    initSsfrShaders() {
        const gl = this.gl;

        const quadVs = `#version 300 es
        precision highp float;
        out vec2 v_uv;
        void main() {
            vec2 pos = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
            v_uv = pos * 0.5 + 0.5;
            gl_Position = vec4(pos, 0.0, 1.0);
        }`;

        // Pass A: Studio Sky + Subtle Underwater Caustics Backdrop
        const bgFs = `#version 300 es
        precision highp float;
        in vec2 v_uv;
        uniform float u_time;
        uniform vec3 u_water_color;
        uniform float u_light_mode;
        out vec4 fragColor;
        float caustic(vec2 uv, float t) {
            vec2 p = uv * 5.2;
            float a = sin(p.x * 1.4 + t * 1.15 + sin(p.y * 1.6 - t * 0.95));
            float b = sin(p.y * 1.5 - t * 1.05 + cos(p.x * 1.3 + t * 1.25));
            float c = sin((p.x + p.y) * 1.1 + t * 0.8);
            return pow(clamp(1.0 - abs(a + b + c * 0.65) * 0.65, 0.0, 1.0), 3.6);
        }
        void main() {
            vec3 darkLow = vec3(0.028, 0.043, 0.072);
            vec3 darkHigh = vec3(0.062, 0.095, 0.155);
            vec3 lightLow = vec3(0.790, 0.845, 0.920);
            vec3 lightHigh = vec3(0.925, 0.955, 0.985);
            vec3 low = mix(darkLow, lightLow, u_light_mode);
            vec3 high = mix(darkHigh, lightHigh, u_light_mode);
            vec3 bg = mix(low, high, smoothstep(0.0, 1.0, v_uv.y));
            float floor_mask = smoothstep(0.55, 0.05, v_uv.y);
            float c = caustic(v_uv * vec2(1.4, 1.1), u_time) * 0.6 + caustic(v_uv * 1.9 + vec2(0.37, -0.21), -u_time * 0.8) * 0.4;
            vec3 causticTint = mix(mix(u_water_color, vec3(0.75, 0.94, 1.0), 0.35), vec3(0.28, 0.64, 0.92), u_light_mode);
            float causticAmp = mix(0.09, 0.06, u_light_mode);
            bg += causticTint * c * floor_mask * causticAmp;
            fragColor = vec4(bg, 1.0);
        }`;

        // Pass B: Eye-Space Sphere Depth (RGB24 packed in RGBA8 for universal WebGL2 mobile support)
        const depthVs = `#version 300 es
        precision highp float;
        layout(location = 0) in vec3 a_pos;
        layout(location = 1) in vec4 a_color;
        uniform mat4 u_view;
        uniform mat4 u_proj;
        uniform float u_point_radius;
        uniform float u_viewport_height;
        uniform float u_tan_half_fov;
        out vec3 v_pos_eye;
        out vec3 v_color_rgb;
        out float v_speed_t;
        void main() {
            vec4 eye_pos = u_view * vec4(a_pos, 1.0);
            v_pos_eye = eye_pos.xyz;
            v_color_rgb = a_color.rgb;
            v_speed_t = a_color.a;
            gl_Position = u_proj * eye_pos;
            float dist = max(-eye_pos.z, 0.001);
            float size_px = (u_point_radius / (dist * u_tan_half_fov)) * u_viewport_height;
            gl_PointSize = clamp(size_px, 2.0, 192.0);
        }`;

        const depthFs = `#version 300 es
        precision highp float;
        in vec3 v_pos_eye;
        in vec3 v_color_rgb;
        in float v_speed_t;
        uniform mat4 u_proj;
        uniform float u_point_radius;
        uniform float u_fusion;
        uniform float u_depth_scale;
        out vec4 fragColor;
        vec3 packDepthRGB(float d) {
            vec3 enc = vec3(1.0, 255.0, 65025.0) * clamp(d, 0.0, 0.999999);
            enc = fract(enc);
            enc -= enc.yzx * vec3(1.0 / 255.0, 1.0 / 255.0, 0.0);
            return enc;
        }
        void main() {
            vec2 coord = vec2(gl_PointCoord.x * 2.0 - 1.0, 1.0 - gl_PointCoord.y * 2.0);
            float r2 = dot(coord, coord);
            if (r2 > 1.0) discard;
            float z = sqrt(max(1.0 - r2, 0.0));
            float flat_z = mix(1.0, 0.36, smoothstep(0.15, 1.0, u_fusion));
            vec3 sphere_eye = v_pos_eye + vec3(coord.x * u_point_radius, coord.y * u_point_radius, z * u_point_radius * flat_z);
            vec4 clip_pos = u_proj * vec4(sphere_eye, 1.0);
            gl_FragDepth = clamp(((clip_pos.z / clip_pos.w) + 1.0) * 0.5, 0.0, 1.0);
            float norm_depth = clamp((-sphere_eye.z) / u_depth_scale, 0.0, 0.999999);
            fragColor = vec4(packDepthRGB(norm_depth), 1.0);
        }`;

        // Pass C: Half-Res Additive Thickness Splatting
        const thickFs = `#version 300 es
        precision highp float;
        in vec3 v_pos_eye;
        in vec3 v_color_rgb;
        in float v_speed_t;
        uniform float u_thick_weight;
        out vec4 fragColor;
        void main() {
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(coord, coord);
            if (r2 > 1.0) discard;
            float z = sqrt(max(1.0 - r2, 0.0));
            float w = pow(z, 1.35) * u_thick_weight;
            fragColor = vec4(w, w * v_speed_t, 0.0, 1.0);
        }`;

        // Pass D: 9-Tap Separable Edge-Preserving Bilateral Depth Blur
        const blurFs = `#version 300 es
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
            return dot(rgb, vec3(1.0, 1.0 / 255.0, 1.0 / 65025.0));
        }
        vec3 packDepthRGB(float d) {
            vec3 enc = vec3(1.0, 255.0, 65025.0) * clamp(d, 0.0, 0.999999);
            enc = fract(enc);
            enc -= enc.yzx * vec3(1.0 / 255.0, 1.0 / 255.0, 0.0);
            return enc;
        }
        void main() {
            vec4 centerSample = texture(t_depth, v_uv);
            if (centerSample.a < 0.5) {
                fragColor = vec4(0.0);
                return;
            }
            float centerDepth = unpackDepthRGB(centerSample.rgb);
            float sum = centerDepth;
            float wSum = 1.0;
            float blurStrength = smoothstep(0.02, 1.0, u_fusion);
            float thresh = max(u_depth_threshold, 0.002);
            float invTwoSigmaD2 = 1.0 / (2.0 * thresh * thresh);
            float invTwoSigmaS2 = 1.0 / (2.0 * 2.6 * 2.6);
            for (int i = -4; i <= 4; i++) {
                if (i == 0) continue;
                float fi = float(i) * blurStrength;
                vec2 suv = clamp(v_uv + u_direction * u_texel_size * fi * u_tap_stride, vec2(0.0005), vec2(0.9995));
                vec4 s = texture(t_depth, suv);
                if (s.a > 0.5) {
                    float d = unpackDepthRGB(s.rgb);
                    float dd = d - centerDepth;
                    float w = exp(-(fi * fi) * invTwoSigmaS2 - (dd * dd) * invTwoSigmaD2);
                    sum += d * w;
                    wSum += w;
                }
            }
            fragColor = vec4(packDepthRGB(sum / wSum), 1.0);
        }`;

        // Pass E: Refractive Optical Water Composite
        const compFs = `#version 300 es
        precision highp float;
        in vec2 v_uv;
        uniform sampler2D t_bg;
        uniform sampler2D t_smooth_depth;
        uniform sampler2D t_raw_depth;
        uniform sampler2D t_thickness;
        uniform vec2 u_texel_size;
        uniform float u_tan_half_fov;
        uniform float u_aspect;
        uniform float u_depth_scale;
        uniform float u_fusion;
        uniform vec3 u_shallow_color;
        uniform vec3 u_deep_color;
        uniform vec4 u_optics; // x: ior, y: roughness, z: specular, w: metallic
        uniform float u_refraction;
        uniform float u_foam_amount;
        uniform vec3 u_light_dir;
        uniform float u_light_mode;
        out vec4 fragColor;

        float unpackDepthRGB(vec3 rgb) {
            return dot(rgb, vec3(1.0, 1.0 / 255.0, 1.0 / 65025.0));
        }
        vec3 uvToEyePos(vec2 uv, float normDepth) {
            float z = -normDepth * u_depth_scale;
            vec2 ndc = uv * 2.0 - 1.0;
            return vec3(ndc.x * (-z) * u_tan_half_fov * u_aspect, ndc.y * (-z) * u_tan_half_fov, z);
        }
        void main() {
            vec3 bgCol = texture(t_bg, v_uv).rgb;
            vec4 depthSample = texture(t_smooth_depth, v_uv);
            vec2 thickSample = texture(t_thickness, v_uv).rg;
            float thickness = thickSample.r * 3.0;
            float speedEnergy = (thickSample.r > 0.004) ? clamp(thickSample.g / thickSample.r, 0.0, 1.0) : 0.0;

            float sheathStrength = smoothstep(0.05, 0.50, u_fusion) * (1.0 - smoothstep(0.88, 1.0, u_fusion));
            if (depthSample.a < 0.5) {
                if (sheathStrength > 0.01 && thickness > 0.08) {
                    float halo = smoothstep(0.08, 0.34, thickness) * sheathStrength;
                    vec3 rimTint = mix(u_shallow_color, vec3(0.88, 0.98, 1.0), 0.45);
                    fragColor = vec4(mix(bgCol, rimTint, halo * 0.42), 1.0);
                } else {
                    fragColor = vec4(bgCol, 1.0);
                }
                return;
            }

            float centerD = unpackDepthRGB(depthSample.rgb);
            float rawD = unpackDepthRGB(texture(t_raw_depth, v_uv).rgb);
            vec3 eyePos = uvToEyePos(v_uv, centerD);

            float stepPx = (u_texel_size.y < 0.00075) ? 2.0 : 1.0;
            vec2 stepX = vec2(u_texel_size.x * stepPx, 0.0);
            vec2 stepY = vec2(0.0, u_texel_size.y * stepPx);

            vec4 sR = texture(t_smooth_depth, clamp(v_uv + stepX, vec2(0.0005), vec2(0.9995)));
            vec4 sL = texture(t_smooth_depth, clamp(v_uv - stepX, vec2(0.0005), vec2(0.9995)));
            vec4 sU = texture(t_smooth_depth, clamp(v_uv + stepY, vec2(0.0005), vec2(0.9995)));
            vec4 sD = texture(t_smooth_depth, clamp(v_uv - stepY, vec2(0.0005), vec2(0.9995)));

            bool vR = sR.a > 0.5;
            bool vL = sL.a > 0.5;
            bool vU = sU.a > 0.5;
            bool vD = sD.a > 0.5;

            float dR = vR ? unpackDepthRGB(sR.rgb) : centerD;
            float dL = vL ? unpackDepthRGB(sL.rgb) : centerD;
            float dU = vU ? unpackDepthRGB(sU.rgb) : centerD;
            float dD = vD ? unpackDepthRGB(sD.rgb) : centerD;

            vec3 posR = uvToEyePos(v_uv + stepX, dR);
            vec3 posL = uvToEyePos(v_uv - stepX, dL);
            vec3 posU = uvToEyePos(v_uv + stepY, dU);
            vec3 posD = uvToEyePos(v_uv - stepY, dD);

            vec3 ddx = (vR && vL)
                ? ((abs(dR - centerD) < abs(centerD - dL)) ? (posR - eyePos) : (eyePos - posL))
                : (vR ? (posR - eyePos) : (vL ? (eyePos - posL) : vec3(posR.x - eyePos.x, 0.0, 0.0)));
            vec3 ddy = (vU && vD)
                ? ((abs(dU - centerD) < abs(centerD - dD)) ? (posU - eyePos) : (eyePos - posD))
                : (vU ? (posU - eyePos) : (vD ? (eyePos - posD) : vec3(0.0, posU.y - eyePos.y, 0.0)));
            vec3 normal = normalize(cross(ddx, ddy));
            if (normal.z < 0.0) normal = -normal;

            vec3 viewDir = normalize(-eyePos);
            vec3 lightDir = normalize(u_light_dir);
            vec3 halfVec = normalize(lightDir + viewDir);

            float ndotl = max(0.0, dot(normal, lightDir));
            float ndotv = max(0.001, dot(normal, viewDir));
            float ndoth = max(0.0, dot(normal, halfVec));

            float ior = max(u_optics.x, 1.01);
            float rough = clamp(max(u_optics.y, 0.055), 0.055, 0.95);
            float specMult = max(u_optics.z, 0.0);
            float metallic = clamp(u_optics.w, 0.0, 1.0);
            float f0Diel = pow((ior - 1.0) / (ior + 1.0), 2.0);
            vec3 f0 = mix(vec3(f0Diel), u_shallow_color, metallic);
            vec3 fresnel = f0 + (vec3(1.0) - f0) * pow(1.0 - ndotv, 4.4);

            float specPow = clamp(1.8 / (rough * rough + 0.003), 18.0, 180.0);
            vec3 spec = pow(ndoth, specPow) * mix(0.85, 1.35, u_fusion) * specMult * mix(vec3(1.0, 0.99, 0.97), u_shallow_color, metallic * 0.5);

            float ballVis = 1.0 - smoothstep(0.15, 0.92, u_fusion);
            float creviceAO = clamp(1.0 - max(0.0, rawD - centerD) * 180.0 * ballVis, 0.55, 1.0);

            // Chromatic screen-space background refraction + Beer-Lambert absorption
            float refrAmt = mix(0.014, 0.054, u_fusion) * clamp(u_refraction * 1.8, 0.0, 1.8) * (0.35 + min(thickness, 2.0));
            vec2 uvR = clamp(v_uv + normal.xy * (refrAmt * 0.94), vec2(0.001), vec2(0.999));
            vec2 uvG = clamp(v_uv + normal.xy * refrAmt, vec2(0.001), vec2(0.999));
            vec2 uvB = clamp(v_uv + normal.xy * (refrAmt * 1.06), vec2(0.001), vec2(0.999));
            vec3 refrBg = vec3(texture(t_bg, uvR).r, texture(t_bg, uvG).g, texture(t_bg, uvB).b);

            vec3 absorbCoeff = max(vec3(1.0) - u_shallow_color * 0.90, vec3(0.05)) * 1.35 + vec3(0.05, 0.025, 0.012);
            vec3 transmittance = exp(-absorbCoeff * thickness * 1.45);

            vec3 tintCol = mix(u_shallow_color, u_deep_color, clamp(1.0 - exp(-thickness * 1.05), 0.0, 1.0));
            vec3 deepTint = tintCol * (0.36 + 0.64 * ndotl) * creviceAO;
            vec3 shallowWater = mix(refrBg * transmittance + deepTint * (vec3(1.0) - transmittance) * 0.88, deepTint, metallic);

            vec3 reflDir = reflect(-viewDir, normal);
            float skyT = pow(clamp(reflDir.y * 0.5 + 0.5, 0.0, 1.0), 1.45);
            vec3 skyLow = mix(vec3(0.09, 0.17, 0.30), vec3(0.50, 0.68, 0.88), u_light_mode);
            vec3 skyHi = mix(vec3(0.76, 0.90, 1.0), vec3(0.95, 0.98, 1.00), u_light_mode);
            vec3 skyReflect = mix(skyLow, skyHi, skyT);
            float softbox = pow(clamp(dot(reflDir, normalize(vec3(-0.20, 0.82, 0.38))), 0.0, 1.0), 14.0);
            skyReflect += vec3(0.72, 0.84, 0.96) * softbox;
            skyReflect *= mix(vec3(1.0), u_shallow_color, metallic);

            float crestFoam = smoothstep(0.45, 0.95, speedEnergy) * smoothstep(0.04, 0.24, thickness)
                            * 0.55 * u_fusion * u_foam_amount * (1.0 - metallic);

            vec3 liquidCol = mix(shallowWater, skyReflect, clamp(fresnel * mix(0.45, 0.92, u_fusion), vec3(0.0), vec3(0.94)))
                           + spec * fresnel
                           + vec3(0.90, 0.97, 1.0) * crestFoam;

            vec3 solidBallCol = (deepTint + skyReflect * fresnel * 0.35) * creviceAO + spec * 0.65;
            fragColor = vec4(clamp(mix(solidBallCol, liquidCol, smoothstep(0.0, 0.65, u_fusion)), 0.0, 1.0), 1.0);
        }`;

        this.ssfrBgProgram = this.createProgram(quadVs, bgFs);
        this.ssfrDepthProgram = this.createProgram(depthVs, depthFs);
        this.ssfrThickProgram = this.createProgram(depthVs, thickFs);
        this.ssfrBlurProgram = this.createProgram(quadVs, blurFs);
        this.ssfrCompositeProgram = this.createProgram(quadVs, compFs);
        this.quadVao = gl.createVertexArray();
    }

    ensureSsfrTargets(fullW, fullH) {
        const gl = this.gl;
        const w = Math.max(2, Math.floor(fullW * this.ssfrScale));
        const h = Math.max(2, Math.floor(fullH * this.ssfrScale));
        if (this.ssfrTargets && this.ssfrTargets.w === w && this.ssfrTargets.h === h) {
            return this.ssfrTargets;
        }
        if (this.ssfrTargets) {
            const old = this.ssfrTargets;
            for (const fbo of [old.bgFbo, old.depthFbo, old.blurTempFbo, old.blurFinalFbo, old.thickFbo]) gl.deleteFramebuffer(fbo);
            for (const tex of [old.bgTex, old.depthTex, old.blurTempTex, old.blurFinalTex, old.thickTex]) gl.deleteTexture(tex);
            for (const rb of [old.bgDepthRb, old.depthRb]) gl.deleteRenderbuffer(rb);
        }

        const makeRT = (tw, th, withDepth, nearest = false) => {
            const tex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, tw, th, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            const filt = nearest ? gl.NEAREST : gl.LINEAR;
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filt);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filt);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            const fbo = gl.createFramebuffer();
            gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
            let rb = null;
            if (withDepth) {
                rb = gl.createRenderbuffer();
                gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
                gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, tw, th);
                gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, rb);
            }
            return { fbo, tex, rb };
        };

        const thickW = Math.max(2, w >> 1);
        const thickH = Math.max(2, h >> 1);
        const bg = makeRT(w, h, true, false);
        const dep = makeRT(w, h, true, true);
        const b1 = makeRT(w, h, false, true);
        const b2 = makeRT(w, h, false, true);
        const th = makeRT(thickW, thickH, false, false);

        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.bindRenderbuffer(gl.RENDERBUFFER, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        this.ssfrTargets = {
            w, h, thickW, thickH,
            bgFbo: bg.fbo, bgTex: bg.tex, bgDepthRb: bg.rb,
            depthFbo: dep.fbo, depthTex: dep.tex, depthRb: dep.rb,
            blurTempFbo: b1.fbo, blurTempTex: b1.tex,
            blurFinalFbo: b2.fbo, blurFinalTex: b2.tex,
            thickFbo: th.fbo, thickTex: th.tex,
        };
        return this.ssfrTargets;
    }

    createProgram(vsSource, fsSource) {
        const gl = this.gl;
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, vsSource);
        gl.compileShader(vs);
        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
            const err = gl.getShaderInfoLog(vs);
            gl.deleteShader(vs);
            throw new Error('Vertex shader compile error: ' + err);
        }

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, fsSource);
        gl.compileShader(fs);
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
            const err = gl.getShaderInfoLog(fs);
            gl.deleteShader(vs);
            gl.deleteShader(fs);
            throw new Error('Fragment shader compile error: ' + err);
        }

        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);

        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            const err = gl.getProgramInfoLog(prog);
            gl.deleteProgram(prog);
            throw new Error('Shader program link error: ' + err);
        }

        gl.deleteShader(vs);
        gl.deleteShader(fs);
        return prog;
    }

    initBuffers() {
        const gl = this.gl;

        this.particleVao = gl.createVertexArray();
        this.particleVbo = gl.createBuffer();

        gl.bindVertexArray(this.particleVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.particleVbo);

        const stride = 28;
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);

        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 4, gl.FLOAT, false, stride, 12);

        this.lineVao = gl.createVertexArray();
        this.lineVbo = gl.createBuffer();

        gl.bindVertexArray(this.lineVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.lineVbo);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);

        gl.bindVertexArray(null);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        this.particleVertexCapacity = 8000;
        this.particleVertexData = new Float32Array(this.particleVertexCapacity * 7);
    }

    render(solver, camera, viewportWidth, viewportHeight, opts = {}) {
        const gl = this.gl;

        const aspect = Math.max(0.1, viewportWidth / Math.max(1.0, viewportHeight));
        const viewMat = camera.viewMatrix();
        const projMat = camera.projectionMatrix(aspect);

        this.viewProjScratch.elements.set(projMat.elements);
        const viewProjMat = this.viewProjScratch.multiply(viewMat);

        const fusion = opts.liquidFusion ?? 1.0;
        const useSsfr = (opts.liquidSurface ?? true) && fusion > 0.005 && solver.particleCount > 0;
        if (useSsfr) {
            this.renderSsfr(solver, camera, viewMat, projMat, viewProjMat, viewportWidth, viewportHeight, opts);
            return;
        }

        const lightMode = opts.lightMode ? 1.0 : 0.0;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, viewportWidth, viewportHeight);
        if (opts.lightMode) {
            gl.clearColor(230.0 / 255.0, 236.0 / 255.0, 245.0 / 255.0, 1.0);
        } else {
            gl.clearColor(9.0 / 255.0, 12.0 / 255.0, 18.0 / 255.0, 1.0);
        }
        gl.clearDepth(1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        this.renderWireframes(solver, viewProjMat, !!opts.lightMode);
        this.renderParticles(solver, camera, viewMat, projMat, viewportHeight, opts);
    }

    renderSsfr(solver, camera, viewMat, projMat, viewProjMat, fullW, fullH, opts) {
        const gl = this.gl;
        const n = solver.particleCount;
        const t = this.ensureSsfrTargets(fullW, fullH);
        const lightMode = opts.lightMode ? 1.0 : 0.0;

        const fusion = Math.max(0.0, Math.min(1.0, opts.liquidFusion ?? 1.0));
        const shallow = opts.shallowColor ?? [0.22, 0.74, 0.97];
        const deep = opts.deepColor ?? [0.05, 0.24, 0.54];
        const ior = opts.ior ?? 1.333;
        const roughness = opts.roughness ?? 0.045;
        const specular = opts.specular ?? 1.35;
        const refraction = opts.refraction ?? 0.42;
        const metallic = opts.metallic ?? 0.0;
        const foamAmount = opts.foamAmount ?? 0.35;
        const animTime = opts.animTime ?? 0.0;

        if (n > this.particleVertexCapacity) {
            this.particleVertexCapacity = n * 2;
            this.particleVertexData = new Float32Array(this.particleVertexCapacity * 7);
        }
        const data = this.particleVertexData;
        const pos = solver.positions;
        const vel = solver.velocities;
        for (let i = 0; i < n; i++) {
            const i3 = i * 3;
            const i7 = i * 7;
            data[i7] = pos[i3];
            data[i7 + 1] = pos[i3 + 1];
            data[i7 + 2] = pos[i3 + 2];
            data[i7 + 3] = shallow[0];
            data[i7 + 4] = shallow[1];
            data[i7 + 5] = shallow[2];
            const vx = vel[i3], vy = vel[i3 + 1], vz = vel[i3 + 2];
            data[i7 + 6] = Math.min(1.0, Math.hypot(vx, vy, vz) / 2.8);
        }

        gl.bindVertexArray(this.particleVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.particleVbo);
        gl.bufferData(gl.ARRAY_BUFFER, data.subarray(0, n * 7), gl.DYNAMIC_DRAW);
        gl.bindVertexArray(null);

        // Pass A: Background (Studio Sky + Subtle Caustics + Floor Grid + Chamber/Obstacle Wireframes)
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.bgFbo);
        gl.viewport(0, 0, t.w, t.h);
        gl.disable(gl.DEPTH_TEST);
        gl.depthMask(false);
        gl.disable(gl.BLEND);
        gl.useProgram(this.ssfrBgProgram);
        gl.uniform1f(gl.getUniformLocation(this.ssfrBgProgram, 'u_time'), animTime);
        gl.uniform3f(gl.getUniformLocation(this.ssfrBgProgram, 'u_water_color'), shallow[0], shallow[1], shallow[2]);
        gl.uniform1f(gl.getUniformLocation(this.ssfrBgProgram, 'u_light_mode'), lightMode);
        gl.bindVertexArray(this.quadVao);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindVertexArray(null);
        this.renderWireframes(solver, viewProjMat, !!opts.lightMode);

        const tanHalfFov = Math.tan(camera.fovY * 0.5);
        const depthScale = 16.0;
        const baseRad = solver.particleRadius * (solver.is2D ? 1.44 : 1.48);
        const fusedRad = baseRad * (1.0 + fusion * 0.46);

        // Pass B: Eye-Space Sphere Depth
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.depthFbo);
        gl.viewport(0, 0, t.w, t.h);
        gl.clearColor(0, 0, 0, 0);
        gl.clearDepth(1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.disable(gl.BLEND);
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);
        gl.depthMask(true);

        gl.useProgram(this.ssfrDepthProgram);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.ssfrDepthProgram, 'u_view'), false, viewMat.elements);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.ssfrDepthProgram, 'u_proj'), false, projMat.elements);
        gl.uniform1f(gl.getUniformLocation(this.ssfrDepthProgram, 'u_point_radius'), fusedRad);
        gl.uniform1f(gl.getUniformLocation(this.ssfrDepthProgram, 'u_viewport_height'), t.h);
        gl.uniform1f(gl.getUniformLocation(this.ssfrDepthProgram, 'u_tan_half_fov'), tanHalfFov);
        gl.uniform1f(gl.getUniformLocation(this.ssfrDepthProgram, 'u_fusion'), fusion);
        gl.uniform1f(gl.getUniformLocation(this.ssfrDepthProgram, 'u_depth_scale'), depthScale);

        gl.bindVertexArray(this.particleVao);
        gl.drawArrays(gl.POINTS, 0, n);

        // Pass C: Half-Resolution Additive Thickness
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.thickFbo);
        gl.viewport(0, 0, t.thickW, t.thickH);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.disable(gl.DEPTH_TEST);
        gl.depthMask(false);
        gl.enable(gl.BLEND);
        gl.blendEquation(gl.FUNC_ADD);
        gl.blendFunc(gl.ONE, gl.ONE);

        gl.useProgram(this.ssfrThickProgram);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.ssfrThickProgram, 'u_view'), false, viewMat.elements);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.ssfrThickProgram, 'u_proj'), false, projMat.elements);
        gl.uniform1f(gl.getUniformLocation(this.ssfrThickProgram, 'u_point_radius'), fusedRad * (1.0 + fusion * 0.28));
        gl.uniform1f(gl.getUniformLocation(this.ssfrThickProgram, 'u_viewport_height'), t.thickH);
        gl.uniform1f(gl.getUniformLocation(this.ssfrThickProgram, 'u_tan_half_fov'), tanHalfFov);
        const thickWeight = Math.max(0.012, Math.min(0.055, solver.particleRadius * 1.1));
        gl.uniform1f(gl.getUniformLocation(this.ssfrThickProgram, 'u_thick_weight'), thickWeight);
        gl.drawArrays(gl.POINTS, 0, n);

        // Pass D1..D4: Two-Iteration Separable À-Trous Edge-Preserving Bilateral Depth Blur
        gl.disable(gl.BLEND);
        gl.useProgram(this.ssfrBlurProgram);
        const wideStride = Math.max(2, Math.min(4, Math.round((t.h * fusedRad) / (Math.max(1.2, camera.distance) * tanHalfFov * 9.0))));
        const depthThresh = (solver.particleRadius * (3.2 + 9.0 * fusion)) / depthScale;
        gl.uniform1i(gl.getUniformLocation(this.ssfrBlurProgram, 't_depth'), 0);
        gl.uniform2f(gl.getUniformLocation(this.ssfrBlurProgram, 'u_texel_size'), 1.0 / t.w, 1.0 / t.h);
        gl.uniform1f(gl.getUniformLocation(this.ssfrBlurProgram, 'u_fusion'), fusion);
        gl.uniform1f(gl.getUniformLocation(this.ssfrBlurProgram, 'u_tap_stride'), 1.0);
        gl.uniform1f(gl.getUniformLocation(this.ssfrBlurProgram, 'u_depth_threshold'), depthThresh);

        gl.bindVertexArray(this.quadVao);
        gl.activeTexture(gl.TEXTURE0);

        // Iter 1 (stride 1 px: removes single-texel & parity noise)
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.blurTempFbo);
        gl.viewport(0, 0, t.w, t.h);
        gl.bindTexture(gl.TEXTURE_2D, t.depthTex);
        gl.uniform2f(gl.getUniformLocation(this.ssfrBlurProgram, 'u_direction'), 1.0, 0.0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        gl.bindFramebuffer(gl.FRAMEBUFFER, t.blurFinalFbo);
        gl.viewport(0, 0, t.w, t.h);
        gl.bindTexture(gl.TEXTURE_2D, t.blurTempTex);
        gl.uniform2f(gl.getUniformLocation(this.ssfrBlurProgram, 'u_direction'), 0.0, 1.0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        if (fusion > 0.08) {
            gl.uniform1f(gl.getUniformLocation(this.ssfrBlurProgram, 'u_tap_stride'), wideStride);

            gl.bindFramebuffer(gl.FRAMEBUFFER, t.blurTempFbo);
            gl.viewport(0, 0, t.w, t.h);
            gl.bindTexture(gl.TEXTURE_2D, t.blurFinalTex);
            gl.uniform2f(gl.getUniformLocation(this.ssfrBlurProgram, 'u_direction'), 1.0, 0.0);
            gl.drawArrays(gl.TRIANGLES, 0, 3);

            gl.bindFramebuffer(gl.FRAMEBUFFER, t.blurFinalFbo);
            gl.viewport(0, 0, t.w, t.h);
            gl.bindTexture(gl.TEXTURE_2D, t.blurTempTex);
            gl.uniform2f(gl.getUniformLocation(this.ssfrBlurProgram, 'u_direction'), 0.0, 1.0);
            gl.drawArrays(gl.TRIANGLES, 0, 3);
        }

        // Pass E: Final Optical Water Composite to Screen
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, fullW, fullH);
        gl.useProgram(this.ssfrCompositeProgram);

        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, t.bgTex);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, t.blurFinalTex);
        gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, t.depthTex);
        gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, t.thickTex);

        const cp = this.ssfrCompositeProgram;
        gl.uniform1i(gl.getUniformLocation(cp, 't_bg'), 0);
        gl.uniform1i(gl.getUniformLocation(cp, 't_smooth_depth'), 1);
        gl.uniform1i(gl.getUniformLocation(cp, 't_raw_depth'), 2);
        gl.uniform1i(gl.getUniformLocation(cp, 't_thickness'), 3);
        gl.uniform2f(gl.getUniformLocation(cp, 'u_texel_size'), 1.0 / t.w, 1.0 / t.h);
        gl.uniform1f(gl.getUniformLocation(cp, 'u_tan_half_fov'), tanHalfFov);
        gl.uniform1f(gl.getUniformLocation(cp, 'u_aspect'), t.w / t.h);
        gl.uniform1f(gl.getUniformLocation(cp, 'u_depth_scale'), depthScale);
        gl.uniform1f(gl.getUniformLocation(cp, 'u_fusion'), fusion);
        gl.uniform3f(gl.getUniformLocation(cp, 'u_shallow_color'), shallow[0], shallow[1], shallow[2]);
        gl.uniform3f(gl.getUniformLocation(cp, 'u_deep_color'), deep[0], deep[1], deep[2]);
        gl.uniform4f(gl.getUniformLocation(cp, 'u_optics'), ior, roughness, specular, metallic);
        gl.uniform1f(gl.getUniformLocation(cp, 'u_refraction'), refraction);
        gl.uniform1f(gl.getUniformLocation(cp, 'u_foam_amount'), foamAmount);
        const lx = 0.48, ly = 0.80, lz = 0.46;
        const invL = 1.0 / Math.hypot(lx, ly, lz);
        gl.uniform3f(gl.getUniformLocation(cp, 'u_light_dir'), lx * invL, ly * invL, lz * invL);
        gl.uniform1f(gl.getUniformLocation(cp, 'u_light_mode'), lightMode);

        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindVertexArray(null);
        gl.useProgram(null);
        gl.enable(gl.DEPTH_TEST);
        gl.depthMask(true);
    }

    pushSegment(idx, ax, ay, az, bx, by, bz) {
        if (idx + 6 > this.lineScratch.length) {
            const grown = new Float32Array(this.lineScratch.length * 2);
            grown.set(this.lineScratch);
            this.lineScratch = grown;
        }
        const s = this.lineScratch;
        s[idx] = ax; s[idx + 1] = ay; s[idx + 2] = az;
        s[idx + 3] = bx; s[idx + 4] = by; s[idx + 5] = bz;
        return idx + 6;
    }

    renderWireframes(solver, viewProjMat, lightMode = false) {
        const gl = this.gl;
        const bMin = solver.bounds.min;
        const bMax = solver.bounds.max;
        let idx = 0;

        if (!solver.is2D) {
            const yFloor = bMin.y - 0.001;
            const divs = 8;
            for (let i = 1; i < divs; i++) {
                const t = i / divs;
                const gx = bMin.x + (bMax.x - bMin.x) * t;
                const gz = bMin.z + (bMax.z - bMin.z) * t;
                idx = this.pushSegment(idx, gx, yFloor, bMin.z, gx, yFloor, bMax.z);
                idx = this.pushSegment(idx, bMin.x, yFloor, gz, bMax.x, yFloor, gz);
            }
        }
        const gridVerts = idx / 3;
        const wireStart = idx;

        if (solver.is2D) {
            idx = this.pushSegment(idx, bMin.x, bMin.y, 0, bMax.x, bMin.y, 0);
            idx = this.pushSegment(idx, bMax.x, bMin.y, 0, bMax.x, bMax.y, 0);
            idx = this.pushSegment(idx, bMax.x, bMax.y, 0, bMin.x, bMax.y, 0);
            idx = this.pushSegment(idx, bMin.x, bMax.y, 0, bMin.x, bMin.y, 0);

            if (solver.obstacleEnabled) {
                const ox = solver.obstaclePos.x;
                const oy = solver.obstaclePos.y;
                const r = solver.obstacleRadius;
                const segments = 36;
                for (let i = 0; i < segments; i++) {
                    const t0 = (i / segments) * Math.PI * 2;
                    const t1 = ((i + 1) / segments) * Math.PI * 2;
                    idx = this.pushSegment(
                        idx,
                        ox + Math.cos(t0) * r, oy + Math.sin(t0) * r, 0,
                        ox + Math.cos(t1) * r, oy + Math.sin(t1) * r, 0
                    );
                }
            }
        } else {
            idx = this.pushSegment(idx, bMin.x, bMin.y, bMin.z, bMax.x, bMin.y, bMin.z);
            idx = this.pushSegment(idx, bMax.x, bMin.y, bMin.z, bMax.x, bMax.y, bMin.z);
            idx = this.pushSegment(idx, bMax.x, bMax.y, bMin.z, bMin.x, bMax.y, bMin.z);
            idx = this.pushSegment(idx, bMin.x, bMax.y, bMin.z, bMin.x, bMin.y, bMin.z);

            idx = this.pushSegment(idx, bMin.x, bMin.y, bMax.z, bMax.x, bMin.y, bMax.z);
            idx = this.pushSegment(idx, bMax.x, bMin.y, bMax.z, bMax.x, bMax.y, bMax.z);
            idx = this.pushSegment(idx, bMax.x, bMax.y, bMax.z, bMin.x, bMax.y, bMax.z);
            idx = this.pushSegment(idx, bMin.x, bMax.y, bMax.z, bMin.x, bMin.y, bMax.z);

            idx = this.pushSegment(idx, bMin.x, bMin.y, bMin.z, bMin.x, bMin.y, bMax.z);
            idx = this.pushSegment(idx, bMax.x, bMin.y, bMin.z, bMax.x, bMin.y, bMax.z);
            idx = this.pushSegment(idx, bMax.x, bMax.y, bMin.z, bMax.x, bMax.y, bMax.z);
            idx = this.pushSegment(idx, bMin.x, bMax.y, bMin.z, bMin.x, bMax.y, bMax.z);

            if (solver.obstacleEnabled) {
                const o = solver.obstaclePos;
                const r = solver.obstacleRadius;
                const segments = 28;

                for (let i = 0; i < segments; i++) {
                    const t0 = (i / segments) * Math.PI * 2;
                    const t1 = ((i + 1) / segments) * Math.PI * 2;
                    const cos0 = Math.cos(t0) * r, sin0 = Math.sin(t0) * r;
                    const cos1 = Math.cos(t1) * r, sin1 = Math.sin(t1) * r;

                    idx = this.pushSegment(idx, o.x + cos0, o.y + sin0, o.z, o.x + cos1, o.y + sin1, o.z);
                    idx = this.pushSegment(idx, o.x + cos0, o.y, o.z + sin0, o.x + cos1, o.y, o.z + sin1);
                    idx = this.pushSegment(idx, o.x, o.y + cos0, o.z + sin0, o.x, o.y + cos1, o.z + sin1);
                }
            }
        }

        const wireVerts = (idx - wireStart) / 3;
        if (idx > 0) {
            gl.useProgram(this.lineProgram);
            gl.uniformMatrix4fv(this.uViewProj, false, viewProjMat.elements);

            gl.bindVertexArray(this.lineVao);
            gl.bindBuffer(gl.ARRAY_BUFFER, this.lineVbo);
            gl.bufferData(gl.ARRAY_BUFFER, this.lineScratch.subarray(0, idx), gl.DYNAMIC_DRAW);

            gl.disable(gl.DEPTH_TEST);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

            if (gridVerts > 0) {
                if (lightMode) {
                    gl.uniform4f(this.uLineColor, 0.34, 0.43, 0.56, 0.36);
                } else {
                    gl.uniform4f(this.uLineColor, 0.17, 0.24, 0.35, 0.45);
                }
                gl.drawArrays(gl.LINES, 0, gridVerts);
            }
            if (wireVerts > 0) {
                if (lightMode) {
                    gl.uniform4f(this.uLineColor, 0.11, 0.36, 0.70, 0.75);
                } else {
                    gl.uniform4f(this.uLineColor, 0.30, 0.66, 0.96, 0.80);
                }
                gl.drawArrays(gl.LINES, gridVerts, wireVerts);
            }
            gl.disable(gl.BLEND);
        }
    }

    renderParticles(solver, camera, viewMat, projMat, viewportHeight, opts = {}) {
        const gl = this.gl;
        const n = solver.particleCount;
        if (n === 0) return;

        const colorMode = opts.colorMode ?? 'material';
        const colorMapType = opts.colorMapType ?? 'ocean';
        const litSpheres = opts.litSpheres ?? true;
        const shallow = opts.shallowColor ?? [0.22, 0.74, 0.97];
        const deep = opts.deepColor ?? [0.05, 0.24, 0.54];
        const texMode = opts.texMode ?? 1;
        const texStrength = opts.texStrength ?? 0.28;
        const ior = opts.ior ?? 1.333;
        const roughness = opts.roughness ?? 0.045;
        const specular = opts.specular ?? 1.35;
        const refraction = opts.refraction ?? 0.42;
        const metallic = opts.metallic ?? 0.0;
        const foamAmount = opts.foamAmount ?? 0.35;
        const animTime = opts.animTime ?? 0.0;

        let cMode = 0;
        if (colorMode === 'pressure') cMode = 1;
        else if (colorMode === 'density') cMode = 2;
        else if (colorMode === 'material' || colorMode === 'single') cMode = 3;

        if (n > this.particleVertexCapacity) {
            this.particleVertexCapacity = n * 2;
            this.particleVertexData = new Float32Array(this.particleVertexCapacity * 7);
        }

        const data = this.particleVertexData;
        const pos = solver.positions;
        const vel = solver.velocities;
        const dens = solver.densities;
        const pres = solver.pressures;

        for (let i = 0; i < n; i++) {
            const i3 = i * 3;
            const i7 = i * 7;

            data[i7] = pos[i3];
            data[i7 + 1] = pos[i3 + 1];
            data[i7 + 2] = pos[i3 + 2];

            const vx = vel[i3], vy = vel[i3 + 1], vz = vel[i3 + 2];
            const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
            const speedT = Math.min(1.0, speed / 2.8);

            if (cMode === 3) {
                data[i7 + 3] = shallow[0];
                data[i7 + 4] = shallow[1];
                data[i7 + 5] = shallow[2];
            } else {
                let t = 0.0;
                if (cMode === 0) {
                    t = Math.min(1.0, speed / 2.5);
                } else if (cMode === 1) {
                    t = Math.min(1.0, pres[i] / (solver.restDensity * 0.45));
                } else {
                    const dev = Math.abs(dens[i] - solver.restDensity) / solver.restDensity;
                    t = Math.min(1.0, dev * 4.0);
                }
                const rgb = ColorMap.sample(colorMapType, t);
                data[i7 + 3] = rgb[0];
                data[i7 + 4] = rgb[1];
                data[i7 + 5] = rgb[2];
            }
            data[i7 + 6] = speedT;
        }

        gl.useProgram(this.particleProgram);

        gl.uniformMatrix4fv(this.uView, false, viewMat.elements);
        gl.uniformMatrix4fv(this.uProj, false, projMat.elements);
        gl.uniform1f(this.uPointRadius, solver.particleRadius * 1.28);
        gl.uniform1f(this.uViewportHeight, viewportHeight);
        gl.uniform1f(this.uTanHalfFov, Math.tan(camera.fovY * 0.5));

        const lx = 0.42, ly = 0.82, lz = 0.48;
        const invLen = 1.0 / Math.hypot(lx, ly, lz);
        gl.uniform3f(this.uLightDir, lx * invLen, ly * invLen, lz * invLen);

        gl.uniform1i(this.uLitSpheres, litSpheres ? 1 : 0);
        gl.uniform1i(this.uIs2D, solver.is2D ? 1 : 0);
        gl.uniform1i(this.uColorMode, cMode);
        gl.uniform1i(this.uTexMode, texMode | 0);
        gl.uniform1f(this.uTexStrength, texStrength);
        gl.uniform1f(this.uAnimTime, animTime);
        gl.uniform3f(this.uShallowColor, shallow[0], shallow[1], shallow[2]);
        gl.uniform3f(this.uDeepColor, deep[0], deep[1], deep[2]);
        gl.uniform4f(this.uOptics, ior, roughness, specular, metallic);
        gl.uniform1f(this.uRefraction, refraction);
        gl.uniform1f(this.uFoamAmount, foamAmount);
        gl.uniform1f(this.uLightMode, opts.lightMode ? 1.0 : 0.0);

        if (solver.is2D) {
            gl.disable(gl.DEPTH_TEST);
            gl.depthMask(false);
        } else {
            gl.enable(gl.DEPTH_TEST);
            gl.depthFunc(gl.LEQUAL);
            gl.depthMask(true);
        }

        gl.bindVertexArray(this.particleVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.particleVbo);
        gl.bufferData(gl.ARRAY_BUFFER, data.subarray(0, n * 7), gl.DYNAMIC_DRAW);

        gl.drawArrays(gl.POINTS, 0, n);

        gl.bindVertexArray(null);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        gl.useProgram(null);
    }
}
