import bpy
import os
import shutil
import sys
import tempfile
import time

def run_blender_case(res: int, frames: int, with_mesh: bool):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames
    scene.render.fps = 30

    cache_dir = tempfile.mkdtemp(prefix=f"blender_fluid_{res}_")
    try:
        # Match LiquidRust 3D Dam Break domain: 1.60m (X) x 0.90m (Y) x 1.20m (Z)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.60))
        domain = bpy.context.active_object
        domain.name = "FluidDomain"
        domain.scale = (1.60, 0.90, 1.20)
        bpy.ops.object.transform_apply(scale=True)

        bpy.ops.object.modifier_add(type="FLUID")
        dom_mod = domain.modifiers["Fluid"]
        dom_mod.fluid_type = "DOMAIN"
        dom = dom_mod.domain_settings
        dom.domain_type = "LIQUID"
        dom.resolution_max = res
        dom.cache_type = "MODULAR"
        dom.cache_directory = cache_dir
        dom.cache_frame_start = 1
        dom.cache_frame_end = frames
        dom.use_mesh = False
        dom.cache_data_format = "OPENVDB"
        dom.cache_mesh_format = "BOBJECT"

        # Match LiquidRust DamBreak fluid block: X[-0.8..-0.15] -> center -0.475 (w=0.65),
        # Y[-0.45..0.45] -> center 0 (d=0.90), Z[0..0.85] -> center 0.425 (h=0.85)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-0.475, 0.0, 0.425))
        flow = bpy.context.active_object
        flow.name = "WaterColumn"
        flow.scale = (0.65, 0.90, 0.85)
        bpy.ops.object.transform_apply(scale=True)

        bpy.ops.object.modifier_add(type="FLUID")
        flow_mod = flow.modifiers["Fluid"]
        flow_mod.fluid_type = "FLOW"
        flow_mod.flow_settings.flow_type = "LIQUID"
        flow_mod.flow_settings.flow_behavior = "GEOMETRY"

        bpy.context.view_layer.objects.active = domain
        domain.select_set(True)

        t0 = time.perf_counter()
        bpy.ops.fluid.bake_data()
        t_data = time.perf_counter() - t0

        # Sample particle count at frame 15 from evaluated depsgraph
        scene.frame_set(min(15, frames))
        deps = bpy.context.evaluated_depsgraph_get()
        eval_dom = domain.evaluated_get(deps)
        p_count = 0
        for ps in eval_dom.particle_systems:
            p_count += len(ps.particles)

        t_mesh = 0.0
        verts = 0
        polys = 0
        if with_mesh:
            dom.use_mesh = True
            tm0 = time.perf_counter()
            bpy.ops.fluid.bake_mesh()
            t_mesh = time.perf_counter() - tm0
            scene.frame_set(min(30, frames))
            deps = bpy.context.evaluated_depsgraph_get()
            eval_dom = domain.evaluated_get(deps)
            mesh_data = eval_dom.to_mesh()
            verts = len(mesh_data.vertices)
            polys = len(mesh_data.polygons)
            eval_dom.to_mesh_clear()

        data_files = len(os.listdir(os.path.join(cache_dir, "data"))) if os.path.isdir(os.path.join(cache_dir, "data")) else 0
        print(
            f"BLENDER_RESULT res={res} frames={frames} data_files={data_files} with_mesh={with_mesh} "
            f"particles={p_count} bake_data_s={t_data:.3f} bake_mesh_s={t_mesh:.3f} "
            f"total_s={t_data + t_mesh:.3f} ms_per_frame={(t_data + t_mesh) * 1000.0 / frames:.2f} "
            f"verts={verts} polys={polys}",
            flush=True,
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

if __name__ == "__main__":
    # Parse args after `--`
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    res = int(argv[0]) if len(argv) > 0 else 64
    frames = int(argv[1]) if len(argv) > 1 else 90
    with_mesh = (argv[2].lower() == "true") if len(argv) > 2 else False
    run_blender_case(res, frames, with_mesh)
