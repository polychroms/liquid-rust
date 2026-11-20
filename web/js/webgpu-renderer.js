// WebGPU High-Performance Studio Particle Renderer for LiquidRust
// Zero-copy GPU rendering directly from the compute storage buffer with cached BindGroups,
// zero per-frame JS heap allocations, 3D studio floor grid, and glossy Fresnel shading.

import { PBF_RENDER_WGSL, WIREFRAME_WGSL, SSFR_WGSL, SSFR_POST_WGSL } from './pbf-shader.js?v=20261120a';
import { Mat4 } from './math.js?v=20261113b';

export class WebGpuRenderer {
    constructor(device, context, format) {
        this.device = device;
        this.context = context;
        this.format = format;

        this.depthFormat = 'depth24plus';
        this.depthTexture = null;
        this.isMobile = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);

        // Pre-allocated uniform & geometry scratch buffers (zero GC per frame)
        this.cBuf = new ArrayBuffer(320);
        this.cF32 = new Float32Array(this.cBuf);
        this.cU32 = new Uint32Array(this.cBuf);
        this.wBuf = new ArrayBuffer(80);
        this.wF32 = new Float32Array(this.wBuf);
        this.gBuf = new ArrayBuffer(80);
        this.gF32 = new Float32Array(this.gBuf);
        this.ssfrBuf = new ArrayBuffer(240);
        this.ssfrF32 = new Float32Array(this.ssfrBuf);
        this.viewProjScratch = new Mat4();
        this.wireScratch = new Float32Array(4096);

        // Cached bind groups & SSFR render targets
        this.cachedParticleBuf = null;
        this.cachedRenderBindGroup = null;
        this.cachedSsfrParticleBuf = null;
        this.ssfrDepthBG = null;
        this.ssfrThickBG = null;
        this.ssfrTargets = null;

        this.initPipelines();
    }

    initPipelines() {
        const device = this.device;

        // 1. Particle Render Pipeline
        const renderShader = device.createShaderModule({
            label: 'PBF Particle Render Shader',
            code: PBF_RENDER_WGSL,
        });

        this.cameraUniformBuf = device.createBuffer({
            label: 'Render Camera Uniform Buffer',
            size: 320,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.renderBindGroupLayout = device.createBindGroupLayout({
            label: 'Render BindGroupLayout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            ],
        });

        const renderPipelineLayout = device.createPipelineLayout({
            label: 'Particle Render Pipeline Layout',
            bindGroupLayouts: [this.renderBindGroupLayout],
        });

        this.particlePipeline = device.createRenderPipeline({
            label: 'Particle Render Pipeline',
            layout: renderPipelineLayout,
            vertex: {
                module: renderShader,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: renderShader,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-strip',
                cullMode: 'none',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less-equal',
                format: this.depthFormat,
            },
        });

        // 2. Wireframe & Studio Floor Grid Render Pipeline
        const wireShader = device.createShaderModule({
            label: 'Wireframe Render Shader',
            code: WIREFRAME_WGSL,
        });

        this.wireUniformBuf = device.createBuffer({
            label: 'Wireframe Uniform Buffer',
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.gridUniformBuf = device.createBuffer({
            label: 'Studio Floor Grid Uniform Buffer',
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.wireBindGroupLayout = device.createBindGroupLayout({
            label: 'Wireframe BindGroupLayout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            ],
        });

        const wirePipelineLayout = device.createPipelineLayout({
            label: 'Wireframe Pipeline Layout',
            bindGroupLayouts: [this.wireBindGroupLayout],
        });

        this.wireBindGroup = device.createBindGroup({
            label: 'Wireframe BindGroup',
            layout: this.wireBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.wireUniformBuf } },
            ],
        });

        this.gridBindGroup = device.createBindGroup({
            label: 'Studio Grid BindGroup',
            layout: this.wireBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.gridUniformBuf } },
            ],
        });

        this.wirePipeline = device.createRenderPipeline({
            label: 'Wireframe Line Pipeline',
            layout: wirePipelineLayout,
            vertex: {
                module: wireShader,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 12,
                    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
                }],
            },
            fragment: {
                module: wireShader,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                    },
                }],
            },
            primitive: {
                topology: 'line-list',
            },
            depthStencil: {
                depthWriteEnabled: false,
                depthCompare: 'always',
                format: this.depthFormat,
            },
        });

        this.wireVbo = null;
        this.gridVertexCount = 0;
        this.wireVertexCount = 0;

        // 3. 5-Pass Screen-Space Fluid Rendering (SSFR) + Translucent Refraction Pipelines
        const ssfrShader = device.createShaderModule({
            label: 'SSFR Geometry & Background Shader',
            code: SSFR_WGSL,
        });
        const ssfrPostShader = device.createShaderModule({
            label: 'SSFR Bilateral Blur & Refractive Composite Shader',
            code: SSFR_POST_WGSL,
        });

        this.ssfrSampler = device.createSampler({
            label: 'SSFR Linear Sampler',
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        });

        const uUsage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
        this.ssfrDepthBuf = device.createBuffer({ label: 'SSFR Main Uniform', size: 240, usage: uUsage });
        this.ssfrThickBuf = device.createBuffer({ label: 'SSFR Thick Uniform', size: 240, usage: uUsage });
        this.ssfrBlurH1Buf = device.createBuffer({ label: 'SSFR BlurH1 Uniform', size: 240, usage: uUsage });
        this.ssfrBlurV1Buf = device.createBuffer({ label: 'SSFR BlurV1 Uniform', size: 240, usage: uUsage });
        this.ssfrBlurH2Buf = device.createBuffer({ label: 'SSFR BlurH2 Uniform', size: 240, usage: uUsage });
        this.ssfrBlurV2Buf = device.createBuffer({ label: 'SSFR BlurV2 Uniform', size: 240, usage: uUsage });

        this.ssfrBgBGL = device.createBindGroupLayout({
            label: 'SSFR BG BindGroupLayout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            ],
        });
        this.ssfrSpriteBGL = device.createBindGroupLayout({
            label: 'SSFR Sprite BindGroupLayout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            ],
        });
        this.ssfrBlurBGL = device.createBindGroupLayout({
            label: 'SSFR Blur BindGroupLayout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
            ],
        });
        this.ssfrCompBGL = device.createBindGroupLayout({
            label: 'SSFR Composite BindGroupLayout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
            ],
        });

        this.ssfrBgBindGroup = device.createBindGroup({
            label: 'SSFR BG BindGroup',
            layout: this.ssfrBgBGL,
            entries: [{ binding: 0, resource: { buffer: this.ssfrDepthBuf } }],
        });

        this.ssfrBgPipeline = device.createRenderPipeline({
            label: 'SSFR Background Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.ssfrBgBGL] }),
            vertex: { module: ssfrShader, entryPoint: 'quad_vs' },
            fragment: { module: ssfrShader, entryPoint: 'bg_fs', targets: [{ format: this.format }] },
            primitive: { topology: 'triangle-list' },
            depthStencil: { depthWriteEnabled: false, depthCompare: 'always', format: this.depthFormat },
        });

        this.ssfrDepthPipeline = device.createRenderPipeline({
            label: 'SSFR Depth Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.ssfrSpriteBGL] }),
            vertex: { module: ssfrShader, entryPoint: 'sprite_vs' },
            fragment: { module: ssfrShader, entryPoint: 'depth_fs', targets: [{ format: 'r32float' }] },
            primitive: { topology: 'triangle-strip', cullMode: 'none' },
            depthStencil: { depthWriteEnabled: true, depthCompare: 'less-equal', format: this.depthFormat },
        });

        this.ssfrThickPipeline = device.createRenderPipeline({
            label: 'SSFR Thickness Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.ssfrSpriteBGL] }),
            vertex: { module: ssfrShader, entryPoint: 'sprite_vs' },
            fragment: {
                module: ssfrShader,
                entryPoint: 'thick_fs',
                targets: [{
                    format: 'rgba16float',
                    blend: {
                        color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                    },
                }],
            },
            primitive: { topology: 'triangle-strip', cullMode: 'none' },
        });

        this.ssfrBlurPipeline = device.createRenderPipeline({
            label: 'SSFR Bilateral Blur Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.ssfrBlurBGL] }),
            vertex: { module: ssfrPostShader, entryPoint: 'quad_vs' },
            fragment: { module: ssfrPostShader, entryPoint: 'blur_fs', targets: [{ format: 'r32float' }] },
            primitive: { topology: 'triangle-list' },
        });

        this.ssfrCompPipeline = device.createRenderPipeline({
            label: 'SSFR Refractive Composite Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.ssfrCompBGL] }),
            vertex: { module: ssfrPostShader, entryPoint: 'quad_vs' },
            fragment: { module: ssfrPostShader, entryPoint: 'composite_fs', targets: [{ format: this.format }] },
            primitive: { topology: 'triangle-list' },
        });
    }

    ensureDepthTexture(width, height) {
        if (!this.depthTexture || this.depthTexture.width !== width || this.depthTexture.height !== height) {
            if (this.depthTexture) this.depthTexture.destroy();
            this.depthTexture = this.device.createTexture({
                label: 'Render Depth Texture',
                size: [Math.max(1, width), Math.max(1, height)],
                format: this.depthFormat,
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });
        }
    }

    ensureSsfrTargets(width, height) {
        const scale = this.isMobile ? 0.75 : 1.0;
        const sw = Math.max(1, Math.round(width * scale));
        const sh = Math.max(1, Math.round(height * scale));
        const tw = Math.max(1, sw >> 1);
        const th = Math.max(1, sh >> 1);

        const t = this.ssfrTargets;
        if (t && t.w === width && t.h === height && t.sw === sw && t.sh === sh) {
            return t;
        }
        if (t) {
            t.bgTex.destroy();
            t.depthTex.destroy();
            t.ssfrDepthZ.destroy();
            t.blurTempTex.destroy();
            t.blurFinalTex.destroy();
            t.thickTex.destroy();
        }

        const device = this.device;
        const rtUsage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
        const bgTex = device.createTexture({ label: 'SSFR BG', size: [width, height], format: this.format, usage: rtUsage });
        const depthTex = device.createTexture({ label: 'SSFR Depth', size: [sw, sh], format: 'r32float', usage: rtUsage });
        const ssfrDepthZ = device.createTexture({ label: 'SSFR DepthZ', size: [sw, sh], format: this.depthFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT });
        const blurTempTex = device.createTexture({ label: 'SSFR BlurTemp', size: [sw, sh], format: 'r32float', usage: rtUsage });
        const blurFinalTex = device.createTexture({ label: 'SSFR BlurFinal', size: [sw, sh], format: 'r32float', usage: rtUsage });
        const thickTex = device.createTexture({ label: 'SSFR Thick', size: [tw, th], format: 'rgba16float', usage: rtUsage });

        const bgView = bgTex.createView();
        const depthView = depthTex.createView();
        const depthZView = ssfrDepthZ.createView();
        const blurTempView = blurTempTex.createView();
        const blurFinalView = blurFinalTex.createView();
        const thickView = thickTex.createView();

        const blurH1BG = device.createBindGroup({
            label: 'SSFR BlurH1 BindGroup',
            layout: this.ssfrBlurBGL,
            entries: [
                { binding: 0, resource: { buffer: this.ssfrBlurH1Buf } },
                { binding: 2, resource: depthView },
            ],
        });
        const blurV1BG = device.createBindGroup({
            label: 'SSFR BlurV1 BindGroup',
            layout: this.ssfrBlurBGL,
            entries: [
                { binding: 0, resource: { buffer: this.ssfrBlurV1Buf } },
                { binding: 2, resource: blurTempView },
            ],
        });
        const blurH2BG = device.createBindGroup({
            label: 'SSFR BlurH2 BindGroup',
            layout: this.ssfrBlurBGL,
            entries: [
                { binding: 0, resource: { buffer: this.ssfrBlurH2Buf } },
                { binding: 2, resource: blurFinalView },
            ],
        });
        const blurV2BG = device.createBindGroup({
            label: 'SSFR BlurV2 BindGroup',
            layout: this.ssfrBlurBGL,
            entries: [
                { binding: 0, resource: { buffer: this.ssfrBlurV2Buf } },
                { binding: 2, resource: blurTempView },
            ],
        });
        const compBG = device.createBindGroup({
            label: 'SSFR Composite BindGroup',
            layout: this.ssfrCompBGL,
            entries: [
                { binding: 0, resource: { buffer: this.ssfrDepthBuf } },
                { binding: 1, resource: this.ssfrSampler },
                { binding: 2, resource: bgView },
                { binding: 3, resource: blurFinalView },
                { binding: 4, resource: depthView },
                { binding: 5, resource: thickView },
            ],
        });

        this.ssfrTargets = {
            w: width, h: height, sw, sh, tw, th,
            bgTex, depthTex, ssfrDepthZ, blurTempTex, blurFinalTex, thickTex,
            bgView, depthView, depthZView, blurTempView, blurFinalView, thickView,
            blurH1BG, blurV1BG, blurH2BG, blurV2BG, compBG,
        };
        return this.ssfrTargets;
    }

    render(solver, camera, width, height, opts = {}) {
        if (width <= 0 || height <= 0) return;
        const device = this.device;
        this.ensureDepthTexture(width, height);

        const liquidSurface = opts.liquidSurface ?? true;
        const liquidFusion = opts.liquidFusion ?? 1.0;

        const aspect = width / height;
        const viewMat = camera.viewMatrix();
        const projMat = camera.projectionMatrix(aspect);
        this.viewProjScratch.elements.set(projMat.elements);
        const viewProjMat = this.viewProjScratch.multiply(viewMat);

        const lightMode = !!opts.lightMode;
        this.updateWireframeBuffer(solver, viewProjMat, lightMode);

        if (liquidSurface && liquidFusion > 0.005 && solver.particleCount > 0) {
            this.renderSsfr(solver, camera, viewMat, projMat, width, height, opts, liquidFusion);
            return;
        }

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

        const eye = camera.eyePosition();

        const f32 = this.cF32;
        const u32 = this.cU32;

        // 0..47: view, proj, view_proj matrices
        f32.set(viewMat.elements, 0);
        f32.set(projMat.elements, 16);
        f32.set(viewProjMat.elements, 32);

        // 48..51: eye_pos
        f32[48] = eye.x;
        f32[49] = eye.y;
        f32[50] = eye.z;
        f32[51] = 1.0;

        // 52..55: params [particle_radius, refraction, is_2d, lit_spheres]
        f32[52] = solver.particleRadius * 1.35;
        f32[53] = refraction;
        f32[54] = solver.is2D ? 1.0 : 0.0;
        f32[55] = litSpheres ? 1.0 : 0.0;

        // 56..59: light_dir [lx, ly, lz, rest_density]
        const lx = 0.42, ly = 0.82, lz = 0.48;
        const invLen = 1.0 / Math.hypot(lx, ly, lz);
        f32[56] = lx * invLen;
        f32[57] = ly * invLen;
        f32[58] = lz * invLen;
        f32[59] = solver.restDensity;

        // 60..63: color_mode [mode, colormap, tex_mode, pad]
        let cMode = 0;
        if (colorMode === 'pressure') cMode = 1;
        else if (colorMode === 'density') cMode = 2;
        else if (colorMode === 'material' || colorMode === 'single') cMode = 3;
        u32[60] = cMode;

        let cMap = 0;
        if (colorMapType === 'turbo') cMap = 1;
        else if (colorMapType === 'viridis') cMap = 2;
        else if (colorMapType === 'fire') cMap = 3;
        u32[61] = cMap;
        u32[62] = texMode | 0;
        u32[63] = 0;

        // 64..67: shallow_color [r, g, b, tex_strength]
        f32[64] = shallow[0];
        f32[65] = shallow[1];
        f32[66] = shallow[2];
        f32[67] = texStrength;

        // 68..71: deep_color [r, g, b, anim_time]
        f32[68] = deep[0];
        f32[69] = deep[1];
        f32[70] = deep[2];
        f32[71] = animTime;

        // 72..75: optics [ior, roughness, specular, metallic]
        f32[72] = ior;
        f32[73] = roughness;
        f32[74] = specular;
        f32[75] = metallic;

        // 76..79: foam_params [crest_foam_intensity, light_mode, 0, 0]
        f32[76] = foamAmount;
        f32[77] = lightMode ? 1.0 : 0.0;
        f32[78] = 0.0;
        f32[79] = 0.0;

        device.queue.writeBuffer(this.cameraUniformBuf, 0, this.cBuf);

        // Reuse BindGroup unless solver reallocated particleBuf
        if (this.cachedParticleBuf !== solver.particleBuf || !this.cachedRenderBindGroup) {
            this.cachedParticleBuf = solver.particleBuf;
            this.cachedRenderBindGroup = device.createBindGroup({
                label: 'Particle Render BindGroup',
                layout: this.renderBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.cameraUniformBuf } },
                    { binding: 1, resource: { buffer: solver.particleBuf } },
                ],
            });
        }

        const encoder = device.createCommandEncoder({ label: 'Render Encoder' });
        const clearRgb = lightMode
            ? { r: 230 / 255, g: 236 / 255, b: 245 / 255, a: 1.0 }
            : { r: 9 / 255, g: 12 / 255, b: 18 / 255, a: 1.0 };
        const pass = encoder.beginRenderPass({
            label: 'PBF Main Render Pass',
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: clearRgb,
                loadOp: 'clear',
                storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });

        // 1. Draw Studio Floor Grid & Chamber/Collider Wireframes
        if (this.wireVbo && (this.gridVertexCount > 0 || this.wireVertexCount > 0)) {
            pass.setPipeline(this.wirePipeline);
            pass.setVertexBuffer(0, this.wireVbo);
            if (this.gridVertexCount > 0) {
                pass.setBindGroup(0, this.gridBindGroup);
                pass.draw(this.gridVertexCount, 1, 0, 0);
            }
            if (this.wireVertexCount > 0) {
                pass.setBindGroup(0, this.wireBindGroup);
                pass.draw(this.wireVertexCount, 1, this.gridVertexCount, 0);
            }
        }

        // 2. Draw Glossy Particles
        if (solver.particleCount > 0) {
            pass.setPipeline(this.particlePipeline);
            pass.setBindGroup(0, this.cachedRenderBindGroup);
            pass.draw(4, solver.particleCount, 0, 0);
        }

        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    renderSsfr(solver, camera, viewMat, projMat, width, height, opts, fusion) {
        const device = this.device;
        const t = this.ensureSsfrTargets(width, height);

        if (this.cachedSsfrParticleBuf !== solver.particleBuf || !this.ssfrDepthBG || !this.ssfrThickBG) {
            this.cachedSsfrParticleBuf = solver.particleBuf;
            this.ssfrDepthBG = device.createBindGroup({
                label: 'SSFR Depth Sprite BindGroup',
                layout: this.ssfrSpriteBGL,
                entries: [
                    { binding: 0, resource: { buffer: this.ssfrDepthBuf } },
                    { binding: 1, resource: { buffer: solver.particleBuf } },
                ],
            });
            this.ssfrThickBG = device.createBindGroup({
                label: 'SSFR Thick Sprite BindGroup',
                layout: this.ssfrSpriteBGL,
                entries: [
                    { binding: 0, resource: { buffer: this.ssfrThickBuf } },
                    { binding: 1, resource: { buffer: solver.particleBuf } },
                ],
            });
        }

        const shallow = opts.shallowColor ?? [0.22, 0.74, 0.97];
        const deep = opts.deepColor ?? [0.05, 0.24, 0.54];
        const ior = opts.ior ?? 1.333;
        const roughness = opts.roughness ?? 0.045;
        const specular = opts.specular ?? 1.35;
        const refraction = opts.refraction ?? 0.42;
        const metallic = opts.metallic ?? 0.0;
        const foamAmount = opts.foamAmount ?? 0.35;
        const animTime = opts.animTime ?? 0.0;

        const baseRad = solver.particleRadius * (1.28 + 0.62 * fusion);
        const thickRad = solver.particleRadius * (1.55 + 0.78 * fusion);
        const thickWeight = Math.min(0.24, Math.max(0.035, 3.8 / Math.sqrt(Math.max(solver.particleCount, 120))));
        const camDist = Math.max(camera.distance ?? 3.2, 0.4);
        const tanHalfFov = Math.tan((camera.fovY ?? (45.0 * Math.PI / 180.0)) * 0.5);
        const wideStride = Math.max(2, Math.min(4, Math.round((t.sh * baseRad) / (camDist * tanHalfFov * 9.0))));
        const depthThresh = solver.particleRadius * (3.2 + 9.0 * fusion);
        const aspect = width / height;

        const lx = 0.42, ly = 0.82, lz = 0.48;
        const invL = 1.0 / Math.hypot(lx, ly, lz);

        const s = this.ssfrF32;
        s.set(viewMat.elements, 0);
        s.set(projMat.elements, 16);
        s[32] = baseRad;
        s[33] = fusion;
        s[34] = thickWeight;
        s[35] = animTime;
        s[36] = 1.0 / t.sw;
        s[37] = 1.0 / t.sh;
        s[38] = 1.0;
        s[39] = 0.0;
        s[40] = 1.0;
        s[41] = depthThresh;
        s[42] = tanHalfFov;
        s[43] = aspect;
        s[44] = shallow[0];
        s[45] = shallow[1];
        s[46] = shallow[2];
        s[47] = refraction;
        s[48] = deep[0];
        s[49] = deep[1];
        s[50] = deep[2];
        s[51] = foamAmount;
        s[52] = ior;
        s[53] = roughness;
        s[54] = specular;
        s[55] = metallic;
        s[56] = lx * invL;
        s[57] = ly * invL;
        s[58] = lz * invL;
        s[59] = opts.lightMode ? 1.0 : 0.0;

        device.queue.writeBuffer(this.ssfrDepthBuf, 0, this.ssfrBuf);
        device.queue.writeBuffer(this.ssfrBlurH1Buf, 0, this.ssfrBuf);

        s[38] = 0.0;
        s[39] = 1.0;
        device.queue.writeBuffer(this.ssfrBlurV1Buf, 0, this.ssfrBuf);

        s[38] = 1.0;
        s[39] = 0.0;
        s[40] = wideStride;
        device.queue.writeBuffer(this.ssfrBlurH2Buf, 0, this.ssfrBuf);

        s[38] = 0.0;
        s[39] = 1.0;
        device.queue.writeBuffer(this.ssfrBlurV2Buf, 0, this.ssfrBuf);

        s[32] = thickRad;
        device.queue.writeBuffer(this.ssfrThickBuf, 0, this.ssfrBuf);

        const encoder = device.createCommandEncoder({ label: 'SSFR Render Encoder' });

        // Pass A: Background gradient + animated floor caustics + studio grid & chamber wireframes
        {
            const bgClearRgb = opts.lightMode
                ? { r: 230 / 255, g: 236 / 255, b: 245 / 255, a: 1.0 }
                : { r: 9 / 255, g: 12 / 255, b: 18 / 255, a: 1.0 };
            const pass = encoder.beginRenderPass({
                label: 'SSFR Pass A - Background',
                colorAttachments: [{
                    view: t.bgView,
                    clearValue: bgClearRgb,
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
                depthStencilAttachment: {
                    view: this.depthTexture.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                },
            });
            pass.setPipeline(this.ssfrBgPipeline);
            pass.setBindGroup(0, this.ssfrBgBindGroup);
            pass.draw(3, 1, 0, 0);

            if (this.wireVbo && (this.gridVertexCount > 0 || this.wireVertexCount > 0)) {
                pass.setPipeline(this.wirePipeline);
                pass.setVertexBuffer(0, this.wireVbo);
                if (this.gridVertexCount > 0) {
                    pass.setBindGroup(0, this.gridBindGroup);
                    pass.draw(this.gridVertexCount, 1, 0, 0);
                }
                if (this.wireVertexCount > 0) {
                    pass.setBindGroup(0, this.wireBindGroup);
                    pass.draw(this.wireVertexCount, 1, this.gridVertexCount, 0);
                }
            }
            pass.end();
        }

        // Pass B: Analytical eye-space depth sprites (with dome flattening)
        {
            const pass = encoder.beginRenderPass({
                label: 'SSFR Pass B - Depth',
                colorAttachments: [{
                    view: t.depthView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
                depthStencilAttachment: {
                    view: t.depthZView,
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                },
            });
            pass.setPipeline(this.ssfrDepthPipeline);
            pass.setBindGroup(0, this.ssfrDepthBG);
            pass.draw(4, solver.particleCount, 0, 0);
            pass.end();
        }

        // Pass C: Half-resolution additive Beer-Lambert thickness & speed accumulation
        {
            const pass = encoder.beginRenderPass({
                label: 'SSFR Pass C - Thickness',
                colorAttachments: [{
                    view: t.thickView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(this.ssfrThickPipeline);
            pass.setBindGroup(0, this.ssfrThickBG);
            pass.draw(4, solver.particleCount, 0, 0);
            pass.end();
        }

        // Pass D1..D4: Two-iteration separable a-trous bilateral depth blur
        for (const [label, targetView, bg] of [
            ['SSFR Pass D1 - Blur H1', t.blurTempView, t.blurH1BG],
            ['SSFR Pass D2 - Blur V1', t.blurFinalView, t.blurV1BG],
            ...(fusion > 0.08 ? [
                ['SSFR Pass D3 - Blur H2', t.blurTempView, t.blurH2BG],
                ['SSFR Pass D4 - Blur V2', t.blurFinalView, t.blurV2BG],
            ] : []),
        ]) {
            const pass = encoder.beginRenderPass({
                label,
                colorAttachments: [{
                    view: targetView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(this.ssfrBlurPipeline);
            pass.setBindGroup(0, bg);
            pass.draw(3, 1, 0, 0);
            pass.end();
        }

        // Pass E: Fullscreen chromatic refraction + Beer-Lambert composite to swapchain
        {
            const pass = encoder.beginRenderPass({
                label: 'SSFR Pass E - Composite',
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 9 / 255, g: 12 / 255, b: 18 / 255, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(this.ssfrCompPipeline);
            pass.setBindGroup(0, t.compBG);
            pass.draw(3, 1, 0, 0);
            pass.end();
        }

        device.queue.submit([encoder.finish()]);
    }

    pushSegment(idx, ax, ay, az, bx, by, bz) {
        if (idx + 6 > this.wireScratch.length) {
            const grown = new Float32Array(this.wireScratch.length * 2);
            grown.set(this.wireScratch);
            this.wireScratch = grown;
        }
        const s = this.wireScratch;
        s[idx] = ax; s[idx + 1] = ay; s[idx + 2] = az;
        s[idx + 3] = bx; s[idx + 4] = by; s[idx + 5] = bz;
        return idx + 6;
    }

    updateWireframeBuffer(solver, viewProjMat, lightMode = false) {
        // Wireframe color uniform (crisp cobalt in light theme, studio cyan-blue in dark theme)
        this.wF32.set(viewProjMat.elements, 0);
        if (lightMode) {
            this.wF32[16] = 0.11;
            this.wF32[17] = 0.36;
            this.wF32[18] = 0.70;
            this.wF32[19] = 0.76;
        } else {
            this.wF32[16] = 0.30;
            this.wF32[17] = 0.66;
            this.wF32[18] = 0.96;
            this.wF32[19] = 0.78;
        }
        this.device.queue.writeBuffer(this.wireUniformBuf, 0, this.wBuf);

        let idx = 0;
        const bMin = solver.bounds.min;
        const bMax = solver.bounds.max;

        // Subtle 3D studio floor grid (when in 3D mode)
        if (!solver.is2D) {
            this.gF32.set(viewProjMat.elements, 0);
            if (lightMode) {
                this.gF32[16] = 0.34;
                this.gF32[17] = 0.43;
                this.gF32[18] = 0.56;
                this.gF32[19] = 0.36;
            } else {
                this.gF32[16] = 0.17;
                this.gF32[17] = 0.24;
                this.gF32[18] = 0.35;
                this.gF32[19] = 0.45;
            }
            this.device.queue.writeBuffer(this.gridUniformBuf, 0, this.gBuf);

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
        this.gridVertexCount = idx / 3;
        const wireStartIdx = idx;

        if (solver.is2D) {
            idx = this.pushSegment(idx, bMin.x, bMin.y, 0, bMax.x, bMin.y, 0);
            idx = this.pushSegment(idx, bMax.x, bMin.y, 0, bMax.x, bMax.y, 0);
            idx = this.pushSegment(idx, bMax.x, bMax.y, 0, bMin.x, bMax.y, 0);
            idx = this.pushSegment(idx, bMin.x, bMax.y, 0, bMin.x, bMin.y, 0);

            if (solver.obstacleEnabled) {
                const ox = solver.obstaclePos.x;
                const oy = solver.obstaclePos.y;
                const r = solver.obstacleRadius;
                const segs = 36;
                for (let i = 0; i < segs; i++) {
                    const t0 = (i / segs) * Math.PI * 2;
                    const t1 = ((i + 1) / segs) * Math.PI * 2;
                    idx = this.pushSegment(
                        idx,
                        ox + Math.cos(t0) * r, oy + Math.sin(t0) * r, 0,
                        ox + Math.cos(t1) * r, oy + Math.sin(t1) * r, 0
                    );
                }
            }
        } else {
            // Bottom face
            idx = this.pushSegment(idx, bMin.x, bMin.y, bMin.z, bMax.x, bMin.y, bMin.z);
            idx = this.pushSegment(idx, bMax.x, bMin.y, bMin.z, bMax.x, bMax.y, bMin.z);
            idx = this.pushSegment(idx, bMax.x, bMax.y, bMin.z, bMin.x, bMax.y, bMin.z);
            idx = this.pushSegment(idx, bMin.x, bMax.y, bMin.z, bMin.x, bMin.y, bMin.z);
            // Top face
            idx = this.pushSegment(idx, bMin.x, bMin.y, bMax.z, bMax.x, bMin.y, bMax.z);
            idx = this.pushSegment(idx, bMax.x, bMin.y, bMax.z, bMax.x, bMax.y, bMax.z);
            idx = this.pushSegment(idx, bMax.x, bMax.y, bMax.z, bMin.x, bMax.y, bMax.z);
            idx = this.pushSegment(idx, bMin.x, bMax.y, bMax.z, bMin.x, bMin.y, bMax.z);
            // Vertical pillars
            idx = this.pushSegment(idx, bMin.x, bMin.y, bMin.z, bMin.x, bMin.y, bMax.z);
            idx = this.pushSegment(idx, bMax.x, bMin.y, bMin.z, bMax.x, bMin.y, bMax.z);
            idx = this.pushSegment(idx, bMax.x, bMax.y, bMin.z, bMax.x, bMax.y, bMax.z);
            idx = this.pushSegment(idx, bMin.x, bMax.y, bMin.z, bMin.x, bMax.y, bMax.z);

            if (solver.obstacleEnabled) {
                const o = solver.obstaclePos;
                const r = solver.obstacleRadius;
                const segs = 28;
                for (let i = 0; i < segs; i++) {
                    const t0 = (i / segs) * Math.PI * 2;
                    const t1 = ((i + 1) / segs) * Math.PI * 2;
                    const c0 = Math.cos(t0) * r, s0 = Math.sin(t0) * r;
                    const c1 = Math.cos(t1) * r, s1 = Math.sin(t1) * r;
                    idx = this.pushSegment(idx, o.x + c0, o.y + s0, o.z, o.x + c1, o.y + s1, o.z);
                    idx = this.pushSegment(idx, o.x + c0, o.y, o.z + s0, o.x + c1, o.y, o.z + s1);
                    idx = this.pushSegment(idx, o.x, o.y + c0, o.z + s0, o.x, o.y + c1, o.z + s1);
                }
            }
        }

        this.wireVertexCount = (idx - wireStartIdx) / 3;
        const byteLen = idx * 4;
        if (byteLen === 0) return;

        if (!this.wireVbo || this.wireVbo.size < byteLen) {
            if (this.wireVbo) this.wireVbo.destroy();
            this.wireVbo = this.device.createBuffer({
                label: 'Wireframe VBO',
                size: Math.max(4096, byteLen * 2),
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
        }
        this.device.queue.writeBuffer(this.wireVbo, 0, this.wireScratch, 0, idx);
    }
}
