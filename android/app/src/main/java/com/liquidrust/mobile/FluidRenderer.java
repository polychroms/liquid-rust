package com.liquidrust.mobile;

import android.content.res.AssetManager;
import android.opengl.GLES30;
import android.opengl.GLSurfaceView;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Zero-allocation OpenGL ES 3.1/3.0 Studio Renderer for LiquidRust Android.
 * Pairs zero-copy SSBO instanced spherical billboards with dual 2D Early-Z / 3D FragDepth pipelines.
 */
public final class FluidRenderer implements GLSurfaceView.Renderer {

    private static final String TAG = "LiquidRustRenderer";

    public interface TelemetryListener {
        void onBackendReady(boolean isGpuCompute, String gpuName, float kernelH, int particles);
        void onTelemetry(int fps, int particles, float stepMs, float kernelH);
    }

    private static final class ParticleProg {
        final int prog;
        final int uView, uProj, uRadius, uInvRho0, uColorMode, uColorMap;
        final int uLightDir, uLitSpheres, uTexMode, uTexStrength, uAnimTime;
        final int uShallowColor, uDeepColor, uOptics, uRefraction, uFoamAmount, uLightMode;

        ParticleProg(int prog) {
            this.prog = prog;
            uView = GLES30.glGetUniformLocation(prog, "u_view");
            uProj = GLES30.glGetUniformLocation(prog, "u_proj");
            uRadius = GLES30.glGetUniformLocation(prog, "u_radius");
            uInvRho0 = GLES30.glGetUniformLocation(prog, "u_inv_rho0");
            uColorMode = GLES30.glGetUniformLocation(prog, "u_color_mode");
            uColorMap = GLES30.glGetUniformLocation(prog, "u_color_map");
            uLightDir = GLES30.glGetUniformLocation(prog, "u_light_dir");
            uLitSpheres = GLES30.glGetUniformLocation(prog, "u_lit_spheres");
            uTexMode = GLES30.glGetUniformLocation(prog, "u_tex_mode");
            uTexStrength = GLES30.glGetUniformLocation(prog, "u_tex_strength");
            uAnimTime = GLES30.glGetUniformLocation(prog, "u_anim_time");
            uShallowColor = GLES30.glGetUniformLocation(prog, "u_shallow_color");
            uDeepColor = GLES30.glGetUniformLocation(prog, "u_deep_color");
            uOptics = GLES30.glGetUniformLocation(prog, "u_optics");
            uRefraction = GLES30.glGetUniformLocation(prog, "u_refraction");
            uFoamAmount = GLES30.glGetUniformLocation(prog, "u_foam_amount");
            uLightMode = GLES30.glGetUniformLocation(prog, "u_light_mode");
        }
    }

    private final AssetManager assets;
    private final SimulationConfig cfg;
    private final Camera3D camera;
    private final TelemetryListener listener;

    private boolean useGpuCompute = false;
    private GpuPbfSolver gpuSolver;
    private CpuPbfSolver cpuSolver;

    private ParticleProg progParticle2D;
    private ParticleProg progParticle3D;

    private int progLine;
    private int uLineViewProj;
    private int uLineColor;

    // 5-Pass Screen-Space Fluid Rendering (SSFR) + Translucent Refraction Pipeline
    private boolean ssfrReady = false;
    private int progSsfrBg, uBgTime, uBgWaterColor, uBgIs2D, uBgLightMode;
    private int progSsfrDepth, uDepView, uDepProj, uDepRadius, uDepInvRho0, uDepColorMode, uDepFusion, uDepScale;
    private int progSsfrThick, uThkView, uThkProj, uThkRadius, uThkInvRho0, uThkColorMode, uThkWeight;
    private int progSsfrBlur, uBlurDepthTex, uBlurTexel, uBlurDir, uBlurFusion, uBlurStride, uBlurThresh;
    private int progSsfrComp, uCompBgTex, uCompSmoothTex, uCompRawTex, uCompThickTex;
    private int uCompTexel, uCompTanHalfFov, uCompAspect, uCompDepthScale, uCompFusion;
    private int uCompShallow, uCompDeep, uCompOptics, uCompRefraction, uCompFoam;
    private int uCompLightDir, uCompLitSpheres, uCompColorMode, uCompColorMap, uCompLightMode;

    private int quadVao;
    private int ssfrW = 0, ssfrH = 0, thickW = 0, thickH = 0;
    private int bgFbo, bgTex;
    private int depthFbo, depthTex, depthRb;
    private int blurTempFbo, blurTempTex;
    private int blurFinalFbo, blurFinalTex;
    private int thickFbo, thickTex;

    private int particleVao;
    private int lineVao;
    private int lineVbo;

    private final FloatBuffer lineFloatBuf;
    private final ByteBuffer lineByteBuf;
    private final float[] lineScratch = new float[2048];

    private float lightX, lightY, lightZ;
    private float animTime = 0.0f;
    private long lastFrameNs = 0L;
    private long telemetryWindowStartNs = 0L;
    private int frameCounter = 0;
    private float accumStepMs = 0.0f;
    private int stepCounter = 0;

    private int vpWidth = 1080;
    private int vpHeight = 1920;

    public FluidRenderer(AssetManager assets, SimulationConfig cfg, Camera3D camera, TelemetryListener listener) {
        this.assets = assets;
        this.cfg = cfg;
        this.camera = camera;
        this.listener = listener;

        float lx = 0.42f, ly = 0.82f, lz = 0.48f;
        float invL = 1.0f / (float) Math.sqrt(lx * lx + ly * ly + lz * lz);
        lightX = lx * invL;
        lightY = ly * invL;
        lightZ = lz * invL;

        lineByteBuf = ByteBuffer.allocateDirect(lineScratch.length * 4).order(ByteOrder.nativeOrder());
        lineFloatBuf = lineByteBuf.asFloatBuffer();
    }

    public void shutdown() {
        if (cpuSolver != null) {
            cpuSolver.shutdown();
        }
    }

    @Override
    public void onSurfaceCreated(GL10 unused, EGLConfig config) {
        int[] major = new int[1];
        int[] minor = new int[1];
        GLES30.glGetIntegerv(GLES30.GL_MAJOR_VERSION, major, 0);
        GLES30.glGetIntegerv(GLES30.GL_MINOR_VERSION, minor, 0);
        String gpuRenderer = GLES30.glGetString(GLES30.GL_RENDERER);
        if (gpuRenderer == null) gpuRenderer = "Mobile GPU";

        boolean supportsCompute = (major[0] > 3) || (major[0] == 3 && minor[0] >= 1);
        useGpuCompute = false;

        if (supportsCompute) {
            try {
                gpuSolver = new GpuPbfSolver(assets);
                useGpuCompute = true;
            } catch (Throwable t) {
                Log.w(TAG, "GLES 3.1 compute init failed, falling back to multi-core CPU solver", t);
                useGpuCompute = false;
            }
        }

        if (!useGpuCompute) {
            cpuSolver = new CpuPbfSolver();
        }

        progParticle2D = new ParticleProg(
            ShaderUtils.buildRenderProgram(assets, "shaders/particle.vert", "shaders/particle.frag", null)
        );
        progParticle3D = new ParticleProg(
            ShaderUtils.buildRenderProgram(assets, "shaders/particle.vert", "shaders/particle.frag", "#define MODE_3D 1")
        );

        progLine = ShaderUtils.buildRenderProgram(assets, "shaders/line.vert", "shaders/line.frag", null);
        uLineViewProj = GLES30.glGetUniformLocation(progLine, "u_view_proj");
        uLineColor = GLES30.glGetUniformLocation(progLine, "u_line_color");

        int[] vaos = new int[3];
        GLES30.glGenVertexArrays(3, vaos, 0);
        particleVao = vaos[0];
        lineVao = vaos[1];
        quadVao = vaos[2];

        // Compile 5-pass SSFR + Translucent Refraction shaders (falls back cleanly if unavailable)
        ssfrReady = false;
        ssfrW = 0;
        ssfrH = 0;
        bgFbo = depthFbo = blurTempFbo = blurFinalFbo = thickFbo = 0;
        bgTex = depthTex = blurTempTex = blurFinalTex = thickTex = depthRb = 0;
        try {
            progSsfrBg = ShaderUtils.buildRenderProgram(assets, "shaders/ssfr_quad.vert", "shaders/ssfr_bg.frag", null);
            uBgTime = GLES30.glGetUniformLocation(progSsfrBg, "u_time");
            uBgWaterColor = GLES30.glGetUniformLocation(progSsfrBg, "u_water_color");
            uBgIs2D = GLES30.glGetUniformLocation(progSsfrBg, "u_is_2d");
            uBgLightMode = GLES30.glGetUniformLocation(progSsfrBg, "u_light_mode");

            progSsfrDepth = ShaderUtils.buildRenderProgram(assets, "shaders/ssfr_sprite.vert", "shaders/ssfr_depth.frag", null);
            uDepView = GLES30.glGetUniformLocation(progSsfrDepth, "u_view");
            uDepProj = GLES30.glGetUniformLocation(progSsfrDepth, "u_proj");
            uDepRadius = GLES30.glGetUniformLocation(progSsfrDepth, "u_radius");
            uDepInvRho0 = GLES30.glGetUniformLocation(progSsfrDepth, "u_inv_rho0");
            uDepColorMode = GLES30.glGetUniformLocation(progSsfrDepth, "u_color_mode");
            uDepFusion = GLES30.glGetUniformLocation(progSsfrDepth, "u_fusion");
            uDepScale = GLES30.glGetUniformLocation(progSsfrDepth, "u_depth_scale");

            progSsfrThick = ShaderUtils.buildRenderProgram(assets, "shaders/ssfr_sprite.vert", "shaders/ssfr_thick.frag", null);
            uThkView = GLES30.glGetUniformLocation(progSsfrThick, "u_view");
            uThkProj = GLES30.glGetUniformLocation(progSsfrThick, "u_proj");
            uThkRadius = GLES30.glGetUniformLocation(progSsfrThick, "u_radius");
            uThkInvRho0 = GLES30.glGetUniformLocation(progSsfrThick, "u_inv_rho0");
            uThkColorMode = GLES30.glGetUniformLocation(progSsfrThick, "u_color_mode");
            uThkWeight = GLES30.glGetUniformLocation(progSsfrThick, "u_thick_weight");

            progSsfrBlur = ShaderUtils.buildRenderProgram(assets, "shaders/ssfr_quad.vert", "shaders/ssfr_blur.frag", null);
            uBlurDepthTex = GLES30.glGetUniformLocation(progSsfrBlur, "t_depth");
            uBlurTexel = GLES30.glGetUniformLocation(progSsfrBlur, "u_texel_size");
            uBlurDir = GLES30.glGetUniformLocation(progSsfrBlur, "u_direction");
            uBlurFusion = GLES30.glGetUniformLocation(progSsfrBlur, "u_fusion");
            uBlurStride = GLES30.glGetUniformLocation(progSsfrBlur, "u_tap_stride");
            uBlurThresh = GLES30.glGetUniformLocation(progSsfrBlur, "u_depth_threshold");

            progSsfrComp = ShaderUtils.buildRenderProgram(assets, "shaders/ssfr_quad.vert", "shaders/ssfr_composite.frag", null);
            uCompBgTex = GLES30.glGetUniformLocation(progSsfrComp, "t_bg");
            uCompSmoothTex = GLES30.glGetUniformLocation(progSsfrComp, "t_smooth_depth");
            uCompRawTex = GLES30.glGetUniformLocation(progSsfrComp, "t_raw_depth");
            uCompThickTex = GLES30.glGetUniformLocation(progSsfrComp, "t_thickness");
            uCompTexel = GLES30.glGetUniformLocation(progSsfrComp, "u_texel_size");
            uCompTanHalfFov = GLES30.glGetUniformLocation(progSsfrComp, "u_tan_half_fov");
            uCompAspect = GLES30.glGetUniformLocation(progSsfrComp, "u_aspect");
            uCompDepthScale = GLES30.glGetUniformLocation(progSsfrComp, "u_depth_scale");
            uCompFusion = GLES30.glGetUniformLocation(progSsfrComp, "u_fusion");
            uCompShallow = GLES30.glGetUniformLocation(progSsfrComp, "u_shallow_color");
            uCompDeep = GLES30.glGetUniformLocation(progSsfrComp, "u_deep_color");
            uCompOptics = GLES30.glGetUniformLocation(progSsfrComp, "u_optics");
            uCompRefraction = GLES30.glGetUniformLocation(progSsfrComp, "u_refraction");
            uCompFoam = GLES30.glGetUniformLocation(progSsfrComp, "u_foam_amount");
            uCompLightDir = GLES30.glGetUniformLocation(progSsfrComp, "u_light_dir");
            uCompLitSpheres = GLES30.glGetUniformLocation(progSsfrComp, "u_lit_spheres");
            uCompColorMode = GLES30.glGetUniformLocation(progSsfrComp, "u_color_mode");
            uCompColorMap = GLES30.glGetUniformLocation(progSsfrComp, "u_color_map");
            uCompLightMode = GLES30.glGetUniformLocation(progSsfrComp, "u_light_mode");

            ssfrReady = true;
        } catch (Throwable t) {
            Log.w(TAG, "SSFR shader init failed; using fast particle renderer", t);
            ssfrReady = false;
        }

        int[] vbos = new int[1];
        GLES30.glGenBuffers(1, vbos, 0);
        lineVbo = vbos[0];

        // Configure instanced particle VAO (attribute 0 = pos_rho, attribute 1 = vel_pres, divisor = 1)
        GLES30.glBindVertexArray(particleVao);
        GLES30.glEnableVertexAttribArray(0);
        GLES30.glVertexAttribDivisor(0, 1);
        GLES30.glEnableVertexAttribArray(1);
        GLES30.glVertexAttribDivisor(1, 1);
        GLES30.glBindVertexArray(0);

        // Configure wireframe VAO
        GLES30.glBindVertexArray(lineVao);
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, lineVbo);
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, lineScratch.length * 4, null, GLES30.GL_DYNAMIC_DRAW);
        GLES30.glEnableVertexAttribArray(0);
        GLES30.glVertexAttribPointer(0, 3, GLES30.GL_FLOAT, false, 12, 0);
        GLES30.glBindVertexArray(0);
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);

        reloadActiveSolver();
        camera.reset(cfg.is2D);

        lastFrameNs = System.nanoTime();
        telemetryWindowStartNs = lastFrameNs;

        if (listener != null) {
            listener.onBackendReady(useGpuCompute, gpuRenderer, getKernelH(), getParticleCount());
        }
    }

    @Override
    public void onSurfaceChanged(GL10 unused, int width, int height) {
        vpWidth = Math.max(1, width);
        vpHeight = Math.max(1, height);
        boolean portraitNow = (vpHeight >= vpWidth);
        if (portraitNow != cfg.isPortrait) {
            cfg.isPortrait = portraitNow;
            if (cfg.is2D) {
                reloadActiveSolver();
            }
        }
        camera.setViewport(vpWidth, vpHeight, cfg.is2D);
        GLES30.glViewport(0, 0, vpWidth, vpHeight);
    }

    private void reloadActiveSolver() {
        if (useGpuCompute) {
            gpuSolver.loadScene(cfg);
            camera.setChamberBounds(
                gpuSolver.minX, gpuSolver.maxX,
                gpuSolver.minY, gpuSolver.maxY,
                gpuSolver.minZ, gpuSolver.maxZ);
        } else {
            cpuSolver.loadScene(cfg);
            camera.setChamberBounds(
                cpuSolver.minX, cpuSolver.maxX,
                cpuSolver.minY, cpuSolver.maxY,
                cpuSolver.minZ, cpuSolver.maxZ);
        }
    }

    private int getParticleCount() {
        return useGpuCompute ? gpuSolver.getParticleCount() : cpuSolver.getParticleCount();
    }

    private float getKernelH() {
        return useGpuCompute ? gpuSolver.getSmoothingLength() : cpuSolver.getSmoothingLength();
    }

    @Override
    public void onDrawFrame(GL10 unused) {
        long nowNs = System.nanoTime();
        float frameDt = (nowNs - lastFrameNs) * 1e-9f;
        lastFrameNs = nowNs;
        if (frameDt <= 0.0f || frameDt > 0.1f) {
            frameDt = 1.0f / 60.0f;
        }

        if (cfg.resetCameraRequested.compareAndSet(true, false)) {
            camera.reset(cfg.is2D);
        }

        if (cfg.reloadSceneRequested.compareAndSet(true, false)) {
            boolean prev2D = useGpuCompute ? gpuSolver.is2D() : cpuSolver.is2D();
            reloadActiveSolver();
            if (prev2D != cfg.is2D) {
                camera.reset(cfg.is2D);
            }
        }

        boolean doStep = cfg.isPlaying || cfg.stepOnceRequested.compareAndSet(true, false);
        if (doStep) {
            long t0 = System.nanoTime();
            if (useGpuCompute) {
                gpuSolver.step(frameDt, cfg);
            } else {
                cpuSolver.step(frameDt, cfg);
            }
            float stepMs = (System.nanoTime() - t0) * 1e-6f;
            accumStepMs += stepMs;
            stepCounter++;
            animTime += Math.min(frameDt, 1.0f / 45.0f) * cfg.simSpeed;
        }

        boolean active2D = useGpuCompute ? gpuSolver.is2D() : cpuSolver.is2D();
        camera.updateMatrices(active2D);

        boolean useSsfr = ssfrReady && cfg.liquidSurface && cfg.liquidFusion > 0.005f && getParticleCount() > 0;
        if (useSsfr && ensureSsfrTargets()) {
            drawSsfr(active2D);
        } else {
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
            GLES30.glViewport(0, 0, vpWidth, vpHeight);
            GLES30.glDepthMask(true);
            if (cfg.lightTheme) {
                GLES30.glClearColor(232.0f / 255.0f, 238.0f / 255.0f, 246.0f / 255.0f, 1.0f);
            } else {
                // Material 3 Dark Neutral Slate (#0E141B)
                GLES30.glClearColor(14.0f / 255.0f, 20.0f / 255.0f, 27.0f / 255.0f, 1.0f);
            }
            GLES30.glClearDepthf(1.0f);
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT | GLES30.GL_DEPTH_BUFFER_BIT);

            drawWireframes(active2D);
            drawParticles(active2D);
        }

        frameCounter++;
        long elapsedNs = nowNs - telemetryWindowStartNs;
        if (elapsedNs >= 450_000_000L) {
            int fps = Math.round((frameCounter * 1_000_000_000.0f) / elapsedNs);
            float avgStep = (stepCounter > 0) ? (accumStepMs / stepCounter) : 0.0f;
            if (listener != null) {
                listener.onTelemetry(fps, getParticleCount(), avgStep, getKernelH());
            }
            frameCounter = 0;
            accumStepMs = 0.0f;
            stepCounter = 0;
            telemetryWindowStartNs = nowNs;
        }
    }

    private int pushSeg(int idx, float ax, float ay, float az, float bx, float by, float bz) {
        if (idx + 6 > lineScratch.length) return idx;
        lineScratch[idx]     = ax; lineScratch[idx + 1] = ay; lineScratch[idx + 2] = az;
        lineScratch[idx + 3] = bx; lineScratch[idx + 4] = by; lineScratch[idx + 5] = bz;
        return idx + 6;
    }

    private void drawWireframes(boolean active2D) {
        float minX = useGpuCompute ? gpuSolver.minX : cpuSolver.minX;
        float minY = useGpuCompute ? gpuSolver.minY : cpuSolver.minY;
        float minZ = useGpuCompute ? gpuSolver.minZ : cpuSolver.minZ;
        float maxX = useGpuCompute ? gpuSolver.maxX : cpuSolver.maxX;
        float maxY = useGpuCompute ? gpuSolver.maxY : cpuSolver.maxY;
        float maxZ = useGpuCompute ? gpuSolver.maxZ : cpuSolver.maxZ;

        int idx = 0;
        if (!active2D) {
            float yFloor = minY - 0.001f;
            int divs = 8;
            for (int i = 1; i < divs; i++) {
                float t = (float) i / divs;
                float gx = minX + (maxX - minX) * t;
                float gz = minZ + (maxZ - minZ) * t;
                idx = pushSeg(idx, gx, yFloor, minZ, gx, yFloor, maxZ);
                idx = pushSeg(idx, minX, yFloor, gz, maxX, yFloor, gz);
            }
        }
        int gridVerts = idx / 3;
        int wireStart = idx;

        if (active2D) {
            idx = pushSeg(idx, minX, minY, 0f, maxX, minY, 0f);
            idx = pushSeg(idx, maxX, minY, 0f, maxX, maxY, 0f);
            idx = pushSeg(idx, maxX, maxY, 0f, minX, maxY, 0f);
            idx = pushSeg(idx, minX, maxY, 0f, minX, minY, 0f);

            if (cfg.obstacleEnabled) {
                float ox = cfg.obstacleX, oy = cfg.obstacleY, r = cfg.obstacleR;
                int segs = 36;
                for (int i = 0; i < segs; i++) {
                    double a0 = (i * Math.PI * 2.0) / segs;
                    double a1 = ((i + 1) * Math.PI * 2.0) / segs;
                    idx = pushSeg(idx,
                        ox + (float) Math.cos(a0) * r, oy + (float) Math.sin(a0) * r, 0f,
                        ox + (float) Math.cos(a1) * r, oy + (float) Math.sin(a1) * r, 0f);
                }
            }
        } else {
            idx = pushSeg(idx, minX, minY, minZ, maxX, minY, minZ);
            idx = pushSeg(idx, maxX, minY, minZ, maxX, maxY, minZ);
            idx = pushSeg(idx, maxX, maxY, minZ, minX, maxY, minZ);
            idx = pushSeg(idx, minX, maxY, minZ, minX, minY, minZ);

            idx = pushSeg(idx, minX, minY, maxZ, maxX, minY, maxZ);
            idx = pushSeg(idx, maxX, minY, maxZ, maxX, maxY, maxZ);
            idx = pushSeg(idx, maxX, maxY, maxZ, minX, maxY, maxZ);
            idx = pushSeg(idx, minX, maxY, maxZ, minX, minY, maxZ);

            idx = pushSeg(idx, minX, minY, minZ, minX, minY, maxZ);
            idx = pushSeg(idx, maxX, minY, minZ, maxX, minY, maxZ);
            idx = pushSeg(idx, maxX, maxY, minZ, maxX, maxY, maxZ);
            idx = pushSeg(idx, minX, maxY, minZ, minX, maxY, maxZ);

            if (cfg.obstacleEnabled) {
                float ox = cfg.obstacleX, oy = cfg.obstacleY, r = cfg.obstacleR;
                int segs = 28;
                for (int i = 0; i < segs; i++) {
                    double a0 = (i * Math.PI * 2.0) / segs;
                    double a1 = ((i + 1) * Math.PI * 2.0) / segs;
                    float c0 = (float) Math.cos(a0) * r, s0 = (float) Math.sin(a0) * r;
                    float c1 = (float) Math.cos(a1) * r, s1 = (float) Math.sin(a1) * r;
                    idx = pushSeg(idx, ox + c0, oy + s0, 0f, ox + c1, oy + s1, 0f);
                    idx = pushSeg(idx, ox + c0, oy, s0, ox + c1, oy, s1);
                    idx = pushSeg(idx, ox, oy + c0, s0, ox, oy + c1, s1);
                }
            }
        }

        int wireVerts = (idx - wireStart) / 3;
        if (idx <= 0) return;

        lineFloatBuf.position(0);
        lineFloatBuf.put(lineScratch, 0, idx);
        lineByteBuf.position(0);

        GLES30.glUseProgram(progLine);
        GLES30.glUniformMatrix4fv(uLineViewProj, 1, false, camera.viewProjMatrix, 0);

        GLES30.glBindVertexArray(lineVao);
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, lineVbo);
        GLES30.glBufferSubData(GLES30.GL_ARRAY_BUFFER, 0, idx * 4, lineByteBuf);

        GLES30.glDisable(GLES30.GL_DEPTH_TEST);
        GLES30.glEnable(GLES30.GL_BLEND);
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA);

        if (gridVerts > 0) {
            if (cfg.lightTheme) {
                GLES30.glUniform4f(uLineColor, 0.34f, 0.43f, 0.56f, 0.36f);
            } else {
                GLES30.glUniform4f(uLineColor, 0.22f, 0.29f, 0.38f, 0.34f);
            }
            GLES30.glDrawArrays(GLES30.GL_LINES, 0, gridVerts);
        }
        if (wireVerts > 0) {
            if (cfg.lightTheme) {
                GLES30.glUniform4f(uLineColor, 0.10f, 0.36f, 0.70f, 0.62f);
            } else {
                GLES30.glUniform4f(uLineColor, 0.56f, 0.76f, 0.96f, 0.36f);
            }
            GLES30.glDrawArrays(GLES30.GL_LINES, gridVerts, wireVerts);
        }

        GLES30.glDisable(GLES30.GL_BLEND);
        GLES30.glBindVertexArray(0);
    }

    private void drawParticles(boolean active2D) {
        int n = getParticleCount();
        if (n <= 0) return;

        int posBufId = useGpuCompute ? gpuSolver.getPosBufferId() : cpuSolver.getPosBufferId();
        int velBufId = useGpuCompute ? gpuSolver.getVelBufferId() : cpuSolver.getVelBufferId();
        float pRad = (useGpuCompute ? gpuSolver.getParticleRadius() : cpuSolver.getParticleRadius()) * 1.35f;
        float invRho0 = useGpuCompute ? gpuSolver.getInvRestDensity() : cpuSolver.getInvRestDensity();

        ParticleProg p = active2D ? progParticle2D : progParticle3D;
        GLES30.glUseProgram(p.prog);

        GLES30.glUniformMatrix4fv(p.uView, 1, false, camera.viewMatrix, 0);
        GLES30.glUniformMatrix4fv(p.uProj, 1, false, camera.projMatrix, 0);
        GLES30.glUniform1f(p.uRadius, pRad);
        GLES30.glUniform1f(p.uInvRho0, invRho0);
        GLES30.glUniform1i(p.uColorMode, cfg.colorMode);
        GLES30.glUniform1i(p.uColorMap, cfg.colorMap);
        GLES30.glUniform3f(p.uLightDir, lightX, lightY, lightZ);
        GLES30.glUniform1i(p.uLitSpheres, cfg.litSpheres ? 1 : 0);
        GLES30.glUniform1i(p.uTexMode, cfg.texMode);
        GLES30.glUniform1f(p.uTexStrength, cfg.texStrength);
        GLES30.glUniform1f(p.uAnimTime, animTime);
        GLES30.glUniform3f(p.uShallowColor, cfg.shallowR, cfg.shallowG, cfg.shallowB);
        GLES30.glUniform3f(p.uDeepColor, cfg.deepR, cfg.deepG, cfg.deepB);
        GLES30.glUniform4f(p.uOptics, cfg.ior, cfg.roughness, cfg.specular, cfg.metallic);
        GLES30.glUniform1f(p.uRefraction, cfg.refraction);
        GLES30.glUniform1f(p.uFoamAmount, cfg.foamAmount);
        GLES30.glUniform1i(p.uLightMode, cfg.lightTheme ? 1 : 0);

        if (active2D) {
            GLES30.glDisable(GLES30.GL_DEPTH_TEST);
            GLES30.glDepthMask(false);
        } else {
            GLES30.glEnable(GLES30.GL_DEPTH_TEST);
            GLES30.glDepthFunc(GLES30.GL_LEQUAL);
            GLES30.glDepthMask(true);
        }

        GLES30.glBindVertexArray(particleVao);

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, posBufId);
        GLES30.glVertexAttribPointer(0, 4, GLES30.GL_FLOAT, false, 16, 0);

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, velBufId);
        GLES30.glVertexAttribPointer(1, 4, GLES30.GL_FLOAT, false, 16, 0);

        GLES30.glDrawArraysInstanced(GLES30.GL_TRIANGLE_STRIP, 0, 4, n);

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
        GLES30.glBindVertexArray(0);
    }

    private void deleteSsfrTargets() {
        int[] fbos = { bgFbo, depthFbo, blurTempFbo, blurFinalFbo, thickFbo };
        GLES30.glDeleteFramebuffers(fbos.length, fbos, 0);
        int[] texs = { bgTex, depthTex, blurTempTex, blurFinalTex, thickTex };
        GLES30.glDeleteTextures(texs.length, texs, 0);
        if (depthRb != 0) {
            int[] rbs = { depthRb };
            GLES30.glDeleteRenderbuffers(1, rbs, 0);
        }
        bgFbo = depthFbo = blurTempFbo = blurFinalFbo = thickFbo = 0;
        bgTex = depthTex = blurTempTex = blurFinalTex = thickTex = depthRb = 0;
        ssfrW = 0;
        ssfrH = 0;
    }

    private int[] createSsfrTarget(int w, int h, boolean withDepth, boolean nearest) {
        int[] tex = new int[1];
        GLES30.glGenTextures(1, tex, 0);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, tex[0]);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, w, h, 0,
                GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null);
        int filt = nearest ? GLES30.GL_NEAREST : GLES30.GL_LINEAR;
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, filt);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, filt);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);

        int[] fbo = new int[1];
        GLES30.glGenFramebuffers(1, fbo, 0);
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[0]);
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D, tex[0], 0);

        int rbId = 0;
        if (withDepth) {
            int[] rb = new int[1];
            GLES30.glGenRenderbuffers(1, rb, 0);
            rbId = rb[0];
            GLES30.glBindRenderbuffer(GLES30.GL_RENDERBUFFER, rbId);
            GLES30.glRenderbufferStorage(GLES30.GL_RENDERBUFFER, GLES30.GL_DEPTH_COMPONENT24, w, h);
            GLES30.glFramebufferRenderbuffer(GLES30.GL_FRAMEBUFFER, GLES30.GL_DEPTH_ATTACHMENT,
                    GLES30.GL_RENDERBUFFER, rbId);
        }

        int status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER);
        return new int[] { fbo[0], tex[0], rbId, status == GLES30.GL_FRAMEBUFFER_COMPLETE ? 1 : 0 };
    }

    private boolean ensureSsfrTargets() {
        // 0.75x resolution on mobile GPUs cuts fill-rate by ~44% while bilateral blur keeps edges silky crisp
        int targetW = Math.max(4, Math.round(vpWidth * 0.75f));
        int targetH = Math.max(4, Math.round(vpHeight * 0.75f));
        if (ssfrW == targetW && ssfrH == targetH && bgFbo != 0) {
            return true;
        }
        if (bgFbo != 0) {
            deleteSsfrTargets();
        }
        int tw = Math.max(2, targetW >> 1);
        int th = Math.max(2, targetH >> 1);

        int[] bg = createSsfrTarget(targetW, targetH, false, false);
        int[] dep = createSsfrTarget(targetW, targetH, true, true);
        int[] b1 = createSsfrTarget(targetW, targetH, false, true);
        int[] b2 = createSsfrTarget(targetW, targetH, false, true);
        int[] tk = createSsfrTarget(tw, th, false, false);

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0);
        GLES30.glBindRenderbuffer(GLES30.GL_RENDERBUFFER, 0);
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);

        bgFbo = bg[0]; bgTex = bg[1];
        depthFbo = dep[0]; depthTex = dep[1]; depthRb = dep[2];
        blurTempFbo = b1[0]; blurTempTex = b1[1];
        blurFinalFbo = b2[0]; blurFinalTex = b2[1];
        thickFbo = tk[0]; thickTex = tk[1];

        boolean ok = (bg[3] == 1 && dep[3] == 1 && b1[3] == 1 && b2[3] == 1 && tk[3] == 1);
        if (!ok) {
            Log.w(TAG, "SSFR FBO allocation incomplete; falling back to direct particles");
            deleteSsfrTargets();
            ssfrReady = false;
            return false;
        }

        ssfrW = targetW;
        ssfrH = targetH;
        thickW = tw;
        thickH = th;
        return true;
    }

    private void drawSsfr(boolean active2D) {
        int n = getParticleCount();
        if (n <= 0) return;

        int posBufId = useGpuCompute ? gpuSolver.getPosBufferId() : cpuSolver.getPosBufferId();
        int velBufId = useGpuCompute ? gpuSolver.getVelBufferId() : cpuSolver.getVelBufferId();
        float rawRadius = useGpuCompute ? gpuSolver.getParticleRadius() : cpuSolver.getParticleRadius();
        float invRho0 = useGpuCompute ? gpuSolver.getInvRestDensity() : cpuSolver.getInvRestDensity();

        float fusion = Math.max(0.0f, Math.min(1.0f, cfg.liquidFusion));
        float depthScale = 16.0f;
        float tanHalfFov = camera.getTanHalfFov();
        float aspect = (float) vpWidth / (float) Math.max(1, vpHeight);
        float camDist = Math.max(0.8f, camera.getDistance());

        // Slightly larger billboard radii as fusion increases bridge inter-particle gaps smoothly
        float baseRad = rawRadius * (active2D ? 1.48f : 1.44f);
        float fusedRad = baseRad * (1.0f + fusion * 0.46f);
        float thickRad = fusedRad * (1.0f + fusion * 0.28f);

        // Pass A: Studio Background + Animated Caustic Floor + Tank/Obstacle Wireframes
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, bgFbo);
        GLES30.glViewport(0, 0, ssfrW, ssfrH);
        GLES30.glDisable(GLES30.GL_DEPTH_TEST);
        GLES30.glDepthMask(false);
        GLES30.glDisable(GLES30.GL_BLEND);

        GLES30.glUseProgram(progSsfrBg);
        GLES30.glUniform1f(uBgTime, animTime);
        GLES30.glUniform3f(uBgWaterColor, cfg.shallowR, cfg.shallowG, cfg.shallowB);
        GLES30.glUniform1i(uBgIs2D, active2D ? 1 : 0);
        GLES30.glUniform1i(uBgLightMode, cfg.lightTheme ? 1 : 0);
        GLES30.glBindVertexArray(quadVao);
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3);
        GLES30.glBindVertexArray(0);

        drawWireframes(active2D);

        // Bind zero-copy GPU/CPU particle VAO once for both Pass B and Pass C
        GLES30.glBindVertexArray(particleVao);
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, posBufId);
        GLES30.glVertexAttribPointer(0, 4, GLES30.GL_FLOAT, false, 16, 0);
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, velBufId);
        GLES30.glVertexAttribPointer(1, 4, GLES30.GL_FLOAT, false, 16, 0);

        // Pass B: Analytical Eye-Space Sphere Depth (with view-Z flattening & RGB24 packing)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, depthFbo);
        GLES30.glViewport(0, 0, ssfrW, ssfrH);
        GLES30.glClearColor(0f, 0f, 0f, 0f);
        GLES30.glClearDepthf(1.0f);
        GLES30.glDepthMask(true);
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT | GLES30.GL_DEPTH_BUFFER_BIT);
        GLES30.glDisable(GLES30.GL_BLEND);
        GLES30.glEnable(GLES30.GL_DEPTH_TEST);
        GLES30.glDepthFunc(GLES30.GL_LEQUAL);

        GLES30.glUseProgram(progSsfrDepth);
        GLES30.glUniformMatrix4fv(uDepView, 1, false, camera.viewMatrix, 0);
        GLES30.glUniformMatrix4fv(uDepProj, 1, false, camera.projMatrix, 0);
        GLES30.glUniform1f(uDepRadius, fusedRad);
        GLES30.glUniform1f(uDepInvRho0, invRho0);
        GLES30.glUniform1i(uDepColorMode, cfg.colorMode);
        GLES30.glUniform1f(uDepFusion, fusion);
        GLES30.glUniform1f(uDepScale, depthScale);
        GLES30.glDrawArraysInstanced(GLES30.GL_TRIANGLE_STRIP, 0, 4, n);

        // Pass C: Half-Resolution Additive Optical Thickness + Speed/Metric Splatting
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, thickFbo);
        GLES30.glViewport(0, 0, thickW, thickH);
        GLES30.glClearColor(0f, 0f, 0f, 0f);
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
        GLES30.glDisable(GLES30.GL_DEPTH_TEST);
        GLES30.glDepthMask(false);
        GLES30.glEnable(GLES30.GL_BLEND);
        GLES30.glBlendEquation(GLES30.GL_FUNC_ADD);
        GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE);

        float thickWeight = active2D
            ? 0.14f
            : Math.max(0.014f, Math.min(0.075f, 1.5f / (float) Math.sqrt(Math.max(n, 80))));

        GLES30.glUseProgram(progSsfrThick);
        GLES30.glUniformMatrix4fv(uThkView, 1, false, camera.viewMatrix, 0);
        GLES30.glUniformMatrix4fv(uThkProj, 1, false, camera.projMatrix, 0);
        GLES30.glUniform1f(uThkRadius, thickRad);
        GLES30.glUniform1f(uThkInvRho0, invRho0);
        GLES30.glUniform1i(uThkColorMode, cfg.colorMode);
        GLES30.glUniform1f(uThkWeight, thickWeight);
        GLES30.glDrawArraysInstanced(GLES30.GL_TRIANGLE_STRIP, 0, 4, n);

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);

        // Pass D1..D4: Two-Iteration Separable À-Trous Edge-Preserving Bilateral Depth Blur
        GLES30.glDisable(GLES30.GL_BLEND);
        GLES30.glUseProgram(progSsfrBlur);
        float wideStride = Math.max(2.0f, Math.min(4.0f,
            Math.round((ssfrH * fusedRad) / (camDist * tanHalfFov * 9.0f))));
        float depthThresh = (rawRadius * (3.2f + 9.0f * fusion)) / depthScale;

        GLES30.glUniform1i(uBlurDepthTex, 0);
        GLES30.glUniform2f(uBlurTexel, 1.0f / ssfrW, 1.0f / ssfrH);
        GLES30.glUniform1f(uBlurFusion, fusion);
        GLES30.glUniform1f(uBlurStride, 1.0f);
        GLES30.glUniform1f(uBlurThresh, depthThresh);

        GLES30.glBindVertexArray(quadVao);
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0);

        // Iter 1: 1px consecutive horizontal + vertical
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, blurTempFbo);
        GLES30.glViewport(0, 0, ssfrW, ssfrH);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTex);
        GLES30.glUniform2f(uBlurDir, 1.0f, 0.0f);
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3);

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, blurFinalFbo);
        GLES30.glViewport(0, 0, ssfrW, ssfrH);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, blurTempTex);
        GLES30.glUniform2f(uBlurDir, 0.0f, 1.0f);
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3);

        if (fusion > 0.08f) {
            GLES30.glUniform1f(uBlurStride, wideStride);

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, blurTempFbo);
            GLES30.glViewport(0, 0, ssfrW, ssfrH);
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, blurFinalTex);
            GLES30.glUniform2f(uBlurDir, 1.0f, 0.0f);
            GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3);

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, blurFinalFbo);
            GLES30.glViewport(0, 0, ssfrW, ssfrH);
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, blurTempTex);
            GLES30.glUniform2f(uBlurDir, 0.0f, 1.0f);
            GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3);
        }

        // Pass E: Fullscreen Chromatic Refractive + Beer-Lambert Liquid Composite to Screen
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        GLES30.glViewport(0, 0, vpWidth, vpHeight);
        GLES30.glUseProgram(progSsfrComp);

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, bgTex);
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, blurFinalTex);
        GLES30.glActiveTexture(GLES30.GL_TEXTURE2);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTex);
        GLES30.glActiveTexture(GLES30.GL_TEXTURE3);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, thickTex);

        GLES30.glUniform1i(uCompBgTex, 0);
        GLES30.glUniform1i(uCompSmoothTex, 1);
        GLES30.glUniform1i(uCompRawTex, 2);
        GLES30.glUniform1i(uCompThickTex, 3);
        GLES30.glUniform2f(uCompTexel, 1.0f / ssfrW, 1.0f / ssfrH);
        GLES30.glUniform1f(uCompTanHalfFov, tanHalfFov);
        GLES30.glUniform1f(uCompAspect, aspect);
        GLES30.glUniform1f(uCompDepthScale, depthScale);
        GLES30.glUniform1f(uCompFusion, fusion);
        GLES30.glUniform3f(uCompShallow, cfg.shallowR, cfg.shallowG, cfg.shallowB);
        GLES30.glUniform3f(uCompDeep, cfg.deepR, cfg.deepG, cfg.deepB);
        GLES30.glUniform4f(uCompOptics, cfg.ior, cfg.roughness, cfg.specular, cfg.metallic);
        GLES30.glUniform1f(uCompRefraction, cfg.refraction);
        GLES30.glUniform1f(uCompFoam, cfg.foamAmount);
        GLES30.glUniform3f(uCompLightDir, lightX, lightY, lightZ);
        GLES30.glUniform1i(uCompLitSpheres, cfg.litSpheres ? 1 : 0);
        GLES30.glUniform1i(uCompColorMode, cfg.colorMode);
        GLES30.glUniform1i(uCompColorMap, cfg.colorMap);
        GLES30.glUniform1i(uCompLightMode, cfg.lightTheme ? 1 : 0);

        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3);

        // Clean state restore
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0);
        GLES30.glBindVertexArray(0);
        GLES30.glUseProgram(0);
        GLES30.glEnable(GLES30.GL_DEPTH_TEST);
        GLES30.glDepthMask(true);
    }
}
