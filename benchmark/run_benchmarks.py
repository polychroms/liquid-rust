#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER_SCRIPT = os.path.join(HERE, "bench_blender.py")
OUT_JSON = os.path.join(HERE, "bench_results.json")

results = []

def run_liquid_rust(label, quality, backend, surface_mesh=False):
    cmd = [
        "liquid-rust",
        "--headless",
        "--preset", "dam-break",
        "--quality", quality,
        "--time", "3.0",
        "--fps", "30",
    ]
    if "CPU" in backend:
        cmd.append("--cpu")
    obj_dir = None
    if surface_mesh:
        obj_dir = tempfile.mkdtemp(prefix=f"lr_obj_{quality}_")
        cmd.extend(["--surface-mesh", "--export-obj", obj_dir])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        out = proc.stdout + "\n" + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + "\n" + (exc.stderr or "")
        rc = -1
    finally:
        if obj_dir:
            shutil.rmtree(obj_dir, ignore_errors=True)

    p_m = re.search(r"Spawned (\d+) particles", out)
    t_m = re.search(r"Total wall time:\s+([0-9.]+)s", out)
    phys_m = re.search(r"Avg physics step:\s+([0-9.]+) ms/frame", out)
    mesh_m = re.search(r"Avg mesh \+ rasterize:\s+([0-9.]+) ms/frame", out)
    thru_m = re.search(r"Throughput:\s+(\d+) particle-steps/sec", out)

    particles = int(p_m.group(1)) if p_m else 0
    total_s = float(t_m.group(1)) if t_m else 0.0
    phys_ms = float(phys_m.group(1)) if phys_m else 0.0
    mesh_ms = float(mesh_m.group(1)) if mesh_m else 0.0
    thru = int(thru_m.group(1)) if thru_m else 0

    note_suffix = ""
    if rc != 0:
        note_suffix = f" [exit={rc}: wgpu queue drop timeout]" if "timed out while waiting" in out else f" [exit={rc}]"

    rec = {
        "simulator": f"LiquidRust ({backend})",
        "preset_or_res": f"{quality}" + (" + OBJ mesh" if surface_mesh else ""),
        "mode": "Physics + Mesh" if surface_mesh else "Physics Only",
        "backend": backend,
        "frames": 90,
        "particles": particles,
        "physics_s": round(phys_ms * 90.0 / 1000.0, 3),
        "mesh_s": round(mesh_ms * 90.0 / 1000.0, 3),
        "total_s": round(total_s, 3),
        "ms_per_frame": round(total_s * 1000.0 / 90.0, 2),
        "particle_steps_per_sec": thru,
        "mesh_verts": None,
        "mesh_polys": None,
        "notes": f"SPH/PBF Dam Break 1.60x0.90x1.20m ({backend})" + (" + SurfaceMesh OBJ export" if surface_mesh else "") + note_suffix,
    }
    print(f"[LiquidRust] {rec['preset_or_res']} | {backend} | rc={rc} | particles={particles} | phys={rec['physics_s']}s | mesh={rec['mesh_s']}s | total={total_s:.3f}s", flush=True)
    if total_s > 0:
        results.append(rec)

def run_blender(res, frames=90):
    proc = subprocess.run(
        ["blender", "-b", "--factory-startup", "--python", BLENDER_SCRIPT, "--", str(res), str(frames), "true"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout + "\n" + proc.stderr
    m = re.search(
        r"BLENDER_RESULT res=(\d+) frames=(\d+) data_files=(\d+) with_mesh=(\w+) particles=(\d+) "
        r"bake_data_s=([0-9.]+) bake_mesh_s=([0-9.]+) total_s=([0-9.]+) ms_per_frame=([0-9.]+) "
        r"verts=(\d+) polys=(\d+)",
        out,
    )
    if not m:
        raise RuntimeError(f"Missing BLENDER_RESULT for res={res}:\n{out[-1500:]}")
    particles = int(m.group(5))
    data_s = float(m.group(6))
    mesh_s = float(m.group(7))
    total_s = float(m.group(8))
    verts = int(m.group(10))
    polys = int(m.group(11))

    # Store both Physics Only (bake_data) and Physics + Mesh (bake_data + bake_mesh) rows cleanly
    r_phys = {
        "simulator": "Blender 5.3.0 (Mantaflow)",
        "preset_or_res": f"res={res} (Data Only)",
        "mode": "Physics Only",
        "backend": "CPU (24-thr)",
        "frames": frames,
        "particles": particles,
        "physics_s": round(data_s, 3),
        "mesh_s": 0.0,
        "total_s": round(data_s, 3),
        "ms_per_frame": round(data_s * 1000.0 / frames, 2),
        "particle_steps_per_sec": round(particles * frames / max(data_s, 1e-6)),
        "mesh_verts": 0,
        "mesh_polys": 0,
        "notes": f"Mantaflow FLIP OpenVDB bake_data(), data_files={m.group(3)}",
    }
    r_mesh = {
        "simulator": "Blender 5.3.0 (Mantaflow)",
        "preset_or_res": f"res={res} (Data + Mesh)",
        "mode": "Physics + Mesh",
        "backend": "CPU (24-thr)",
        "frames": frames,
        "particles": particles,
        "physics_s": round(data_s, 3),
        "mesh_s": round(mesh_s, 3),
        "total_s": round(total_s, 3),
        "ms_per_frame": round(total_s * 1000.0 / frames, 2),
        "particle_steps_per_sec": round(particles * frames / max(total_s, 1e-6)),
        "mesh_verts": verts,
        "mesh_polys": polys,
        "notes": f"Mantaflow FLIP bake_data() + bake_mesh(), verts={verts}, polys={polys}",
    }
    print(f"[Blender] res={res} | files={m.group(3)} | particles={particles} | data={data_s:.3f}s | mesh={mesh_s:.3f}s | total={total_s:.3f}s", flush=True)
    results.extend([r_phys, r_mesh])

for q in ["medium", "high", "ultra", "cinematic"]:
    run_liquid_rust(q, q, "Vulkan GPU", surface_mesh=False)

for q in ["medium", "high", "ultra", "cinematic"]:
    run_liquid_rust(q, q, "CPU Rayon", surface_mesh=False)

for q in ["high", "ultra"]:
    run_liquid_rust(q, q, "Vulkan GPU", surface_mesh=True)

for res in [32, 64, 96]:
    run_blender(res, frames=90)

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"SAVED {len(results)} rows -> {OUT_JSON}", flush=True)
