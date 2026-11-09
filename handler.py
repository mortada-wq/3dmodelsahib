"""RunPod Serverless worker: one concept image in, one GLB out.

Model: microsoft/TRELLIS.2-4B (MIT). Image-to-3D only — Sahib supplies the
concept image from the FLUX design pipeline, which is what keeps the brand
colours and silhouette. Text-to-3D is deliberately not offered here.

The worker returns a compact GLB: decimated mesh, 1K WebP textures, PBR
channels baked by TRELLIS.2. No preview render is produced — that path pulls
in the renderers and the NVIDIA non-commercial-adjacent stack, and the client
renders the GLB with model-viewer instead.

Input (job["input"]):
    image_url     https URL or data:image/... (required unless image_base64)
    image_base64  raw base64, optional alternative to image_url
    seed          int, default random
    pipeline_type "512" | "1024_cascade" | "1536_cascade", default 1024_cascade
    texture_size  512..2048, default 1024
    decimation    5000..150000 triangles, default 30000
    upload_url    optional presigned PUT; when set the GLB is uploaded there
                  and only the URL is returned (keeps the RunPod payload small)

Output:
    { glb_b64 | glb_url, bytes, tris, ms, seed, pipeline_type,
      texture_size, decimation }
"""

import base64
import io
import os
import random
import tempfile
import time

import o_voxel
import requests
import runpod
import torch
from PIL import Image

MODEL_ID = os.environ.get("TRELLIS_MODEL_ID", "microsoft/TRELLIS.2-4B")
# A network volume keeps the 15 GB checkpoint between cold starts; without
# one, fall back to the container's own cache instead of writing into a
# missing /runpod-volume directory.
_VOLUME = "/runpod-volume"
HF_HOME = os.environ.get(
    "HF_HOME",
    f"{_VOLUME}/huggingface-cache" if os.path.isdir(_VOLUME) else "/root/.cache/huggingface",
)
CACHE_DIR = os.path.join(HF_HOME, "hub")
LOW_VRAM = os.environ.get("TRELLIS_LOW_VRAM", "1") != "0"
PIPELINE_TYPES = {"512", "1024_cascade", "1536_cascade"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024

_pipeline = None


def _cached_model_path() -> str | None:
    """RunPod network volumes mount at /runpod-volume; if the model was
    downloaded on a previous cold start it is already in the HF cache."""
    snapshots = os.path.join(CACHE_DIR, f"models--{MODEL_ID.replace('/', '--')}", "snapshots")
    if not os.path.isdir(snapshots):
        return None
    for name in sorted(os.listdir(snapshots)):
        candidate = os.path.join(snapshots, name)
        if os.path.isfile(os.path.join(candidate, "pipeline.json")):
            return candidate
    return None


def load_pipeline():
    """Load once per worker, not once per job."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    source = _cached_model_path() or MODEL_ID
    started = time.time()
    print(f"[trellis] loading {source} (low_vram={LOW_VRAM})", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(source)
    # The checkpoint's pipeline.json defaults low_vram to true. On a 48 GB
    # card turning it off keeps every stage resident and removes the CPU
    # round-trips; on 24 GB it must stay on.
    pipeline.low_vram = LOW_VRAM
    pipeline.cuda()
    torch.set_grad_enabled(False)
    _pipeline = pipeline
    print(f"[trellis] ready in {time.time() - started:.1f}s", flush=True)
    return pipeline


def _decode_image(payload: dict) -> Image.Image:
    raw_b64 = payload.get("image_base64")
    if raw_b64:
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        data = base64.b64decode(raw_b64)
    else:
        url = payload.get("image_url") or payload.get("image")
        if not url:
            raise ValueError("image_url (or image_base64) is required")
        if url.startswith("data:"):
            data = base64.b64decode(url.split(",", 1)[1])
        else:
            if not url.startswith(("https://", "http://")):
                raise ValueError("image_url must be http(s) or a data URL")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.content
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("reference image exceeds 20 MB")
    image = Image.open(io.BytesIO(data))
    return image.convert("RGBA")


def _clamp_int(value, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _pipeline_type(value) -> str:
    return value if value in PIPELINE_TYPES else "1024_cascade"


def handler(job: dict) -> dict:
    payload = job.get("input") or {}
    pipeline = load_pipeline()

    image = _decode_image(payload)
    seed = _clamp_int(payload.get("seed"), 0, 2**31 - 1, random.randint(0, 2**31 - 1))
    pipeline_type = _pipeline_type(payload.get("pipeline_type"))
    texture_size = _clamp_int(payload.get("texture_size"), 512, 2048, 1024)
    decimation = _clamp_int(payload.get("decimation"), 5000, 150000, 30000)
    upload_url = payload.get("upload_url")

    started = time.time()
    print(
        f"[trellis] job start seed={seed} type={pipeline_type} "
        f"tex={texture_size} tris={decimation}",
        flush=True,
    )

    mesh = pipeline.run(image, seed=seed, pipeline_type=pipeline_type)[0]

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as handle:
        path = handle.name
    try:
        glb.export(path, extension_webp=True)
        with open(path, "rb") as handle:
            data = handle.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    elapsed_ms = int((time.time() - started) * 1000)
    tris = int(len(mesh.faces))
    result = {
        "bytes": len(data),
        "tris": tris,
        "ms": elapsed_ms,
        "seed": seed,
        "pipeline_type": pipeline_type,
        "texture_size": texture_size,
        "decimation": decimation,
    }

    if upload_url:
        response = requests.put(
            upload_url,
            data=data,
            headers={"Content-Type": "model/gltf-binary"},
            timeout=120,
        )
        response.raise_for_status()
        result["glb_url"] = upload_url
    else:
        result["glb_b64"] = base64.b64encode(data).decode("ascii")

    torch.cuda.empty_cache()
    print(
        f"[trellis] done tris={tris} bytes={len(data)} ms={elapsed_ms}",
        flush=True,
    )
    return result


print("[trellis] booting worker", flush=True)
load_pipeline()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
