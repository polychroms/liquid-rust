#version 300 es
precision highp float;
precision highp int;

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
uniform int u_lit_spheres;
uniform int u_color_mode;
uniform int u_color_map;

out vec4 fragColor;

float unpackDepthRGB(vec3 rgb) {
    return rgb.r + rgb.g / 255.0 + rgb.b / 65025.0;
}

vec3 uvToEyePos(vec2 uv, float normDepth) {
    float zEye = -normDepth * u_depth_scale;
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * (-zEye) * u_tan_half_fov * u_aspect,
                ndc.y * (-zEye) * u_tan_half_fov,
                zEye);
}

vec3 evalColorMap(float t, int cmap) {
    t = clamp(t, 0.0, 1.0);
    if (cmap == 1) {
        // Turbo
        vec3 c0 = vec3(0.19, 0.07, 0.23);
        vec3 c1 = vec3(0.12, 0.56, 0.90);
        vec3 c2 = vec3(0.64, 0.95, 0.26);
        vec3 c3 = vec3(0.95, 0.25, 0.09);
        if (t < 0.33) return mix(c0, c1, t / 0.33);
        if (t < 0.66) return mix(c1, c2, (t - 0.33) / 0.33);
        return mix(c2, c3, (t - 0.66) / 0.34);
    } else if (cmap == 2) {
        // Viridis
        vec3 c0 = vec3(0.27, 0.01, 0.33);
        vec3 c1 = vec3(0.19, 0.41, 0.56);
        vec3 c2 = vec3(0.12, 0.63, 0.53);
        vec3 c3 = vec3(0.99, 0.91, 0.14);
        if (t < 0.33) return mix(c0, c1, t / 0.33);
        if (t < 0.66) return mix(c1, c2, (t - 0.33) / 0.33);
        return mix(c2, c3, (t - 0.66) / 0.34);
    } else if (cmap == 3) {
        // Magma
        vec3 c0 = vec3(0.06, 0.02, 0.12);
        vec3 c1 = vec3(0.76, 0.18, 0.24);
        vec3 c2 = vec3(0.99, 0.64, 0.12);
        vec3 c3 = vec3(0.99, 0.96, 0.80);
        if (t < 0.33) return mix(c0, c1, t / 0.33);
        if (t < 0.66) return mix(c1, c2, (t - 0.33) / 0.33);
        return mix(c2, c3, (t - 0.66) / 0.34);
    }
    // Ocean default
    vec3 c0 = vec3(0.04, 0.18, 0.44);
    vec3 c1 = vec3(0.08, 0.55, 0.84);
    vec3 c2 = vec3(0.32, 0.88, 0.96);
    vec3 c3 = vec3(0.92, 0.98, 1.00);
    if (t < 0.33) return mix(c0, c1, t / 0.33);
    if (t < 0.66) return mix(c1, c2, (t - 0.33) / 0.33);
    return mix(c2, c3, (t - 0.66) / 0.34);
}

void main() {
    vec3 bgCol = texture(t_bg, v_uv).rgb;
    vec4 dSample = texture(t_smooth_depth, v_uv);
    vec3 thickSample = texture(t_thickness, v_uv).rgb;
    float thickness = max(thickSample.r, 0.0) * 2.85;
    float metricVal = (thickSample.r > 0.004) ? clamp(thickSample.g / thickSample.r, 0.0, 1.0) : 0.0;
    float speedEnergy = (thickSample.r > 0.004) ? clamp(thickSample.b / thickSample.r, 0.0, 1.0) : 0.0;

    vec3 shallowCol = u_shallow_color;
    vec3 deepCol = u_deep_color;
    if (u_color_mode != 3) {
        shallowCol = evalColorMap(metricVal, u_color_map);
        deepCol = shallowCol * 0.45;
    }

    // Intermediate-fusion glassy meniscus sheath around thin edges
    float sheathStr = smoothstep(0.05, 0.50, u_fusion) * (1.0 - smoothstep(0.88, 1.0, u_fusion));
    if (dSample.a < 0.5) {
        if (sheathStr > 0.01 && thickness > 0.06) {
            float halo = smoothstep(0.06, 0.28, thickness) * sheathStr;
            vec3 rimTint = mix(shallowCol, vec3(0.88, 0.98, 1.0), 0.45);
            fragColor = vec4(mix(bgCol, rimTint, halo * 0.42), 1.0);
            return;
        }
        fragColor = vec4(bgCol, 1.0);
        return;
    }

    float centerD = unpackDepthRGB(dSample.rgb);
    float rawD = unpackDepthRGB(texture(t_raw_depth, v_uv).rgb);
    vec3 eyePos = uvToEyePos(v_uv, centerD);

    // One-sided minimum-gradient eye-space surface normal reconstruction (silhouette-safe)
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
    float specMult = (u_lit_spheres == 1) ? max(u_optics.z, 0.0) : 0.0;
    float metallic = (u_color_mode == 3) ? clamp(u_optics.w, 0.0, 1.0) : 0.0;

    float f0Diel = pow((ior - 1.0) / (ior + 1.0), 2.0);
    vec3 f0 = mix(vec3(f0Diel), shallowCol, metallic);
    vec3 fresnel = f0 + (vec3(1.0) - f0) * pow(1.0 - ndotv, 4.4);

    float specPow = clamp(1.8 / (rough * rough + 0.003), 18.0, 180.0);
    vec3 spec = pow(ndoth, specPow) * mix(0.85, 1.35, u_fusion) * specMult
              * mix(vec3(1.0, 0.99, 0.97), shallowCol, metallic * 0.5);

    // Crevice AO between particles (fades out cleanly at high fusion)
    float ballVis = 1.0 - smoothstep(0.15, 0.92, u_fusion);
    float creviceAO = clamp(1.0 - max(0.0, rawD - centerD) * 180.0 * ballVis, 0.55, 1.0);

    // True chromatic screen-space background refraction + Beer-Lambert thickness transmittance
    float refrAmt = mix(0.014, 0.054, u_fusion) * clamp(u_refraction * 1.8, 0.0, 1.8) * (0.35 + min(thickness, 2.0));
    vec2 uvR = clamp(v_uv + normal.xy * (refrAmt * 0.94), vec2(0.001), vec2(0.999));
    vec2 uvG = clamp(v_uv + normal.xy * refrAmt, vec2(0.001), vec2(0.999));
    vec2 uvB = clamp(v_uv + normal.xy * (refrAmt * 1.06), vec2(0.001), vec2(0.999));
    vec3 refrBg = vec3(texture(t_bg, uvR).r, texture(t_bg, uvG).g, texture(t_bg, uvB).b);

    vec3 absorbCoeff = max(vec3(1.0) - shallowCol * 0.90, vec3(0.05)) * 1.35 + vec3(0.05, 0.025, 0.012);
    vec3 transmittance = exp(-absorbCoeff * thickness * 1.45);

    vec3 tintCol = mix(shallowCol, deepCol, clamp(1.0 - exp(-thickness * 1.05), 0.0, 1.0));
    vec3 deepTint = tintCol * (0.36 + 0.64 * ndotl) * creviceAO;
    vec3 shallowWater = mix(refrBg * transmittance + deepTint * (vec3(1.0) - transmittance) * 0.88, deepTint, metallic);

    vec3 reflDir = reflect(-viewDir, normal);
    float skyT = pow(clamp(reflDir.y * 0.5 + 0.5, 0.0, 1.0), 1.45);
    vec3 skyReflect = mix(vec3(0.09, 0.17, 0.30), vec3(0.76, 0.90, 1.0), skyT);
    float softbox = pow(clamp(dot(reflDir, normalize(vec3(-0.20, 0.82, 0.38))), 0.0, 1.0), 14.0);
    skyReflect += vec3(0.72, 0.84, 0.96) * softbox;
    skyReflect *= mix(vec3(1.0), shallowCol, metallic);
    float reflGain = (u_lit_spheres == 1) ? 1.0 : 0.25;

    float crestFoam = smoothstep(0.45, 0.95, speedEnergy) * smoothstep(0.04, 0.24, thickness)
                    * 0.55 * u_fusion * u_foam_amount * (1.0 - metallic);

    vec3 liquidCol = mix(shallowWater, skyReflect, clamp(fresnel * mix(0.45, 0.92, u_fusion) * reflGain, vec3(0.0), vec3(0.94)))
                   + spec * fresnel
                   + vec3(0.90, 0.97, 1.0) * crestFoam;

    vec3 solidBallCol = (deepTint + skyReflect * fresnel * 0.35 * reflGain) * creviceAO + spec * 0.65;
    fragColor = vec4(clamp(mix(solidBallCol, liquidCol, smoothstep(0.0, 0.65, u_fusion)), 0.0, 1.0), 1.0);
}
