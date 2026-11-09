# RunPod worker — TRELLIS.2 image to 3D

This repository is the deployment source for the Sahib 3D worker. It is a
copy of [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) with a
root `Dockerfile` and `handler.py` added, because RunPod builds from the repo
root and expects `Dockerfile` there.

## Deploy

1. RunPod → Serverless → New endpoint → **GitHub repo** → `mortada-wq/3dmodelsahib`.
2. Branch `main`, Dockerfile path `/Dockerfile`.
3. GPU: **48 GB (A6000 / A40)**. 24 GB works only at 512³–1024³.
4. Active workers `0`, max workers `2`, idle timeout `5`, execution timeout
   `300000` ms, concurrency `1`.
5. Container disk `30 GB`; attach a `60 GB` network volume at `/runpod-volume`.
6. Environment:
   - `HF_HOME=/runpod-volume/huggingface-cache`
   - `TRELLIS_LOW_VRAM=1` (set `0` on 48 GB for faster jobs)
7. Build. The first build takes 40–70 minutes, mostly `flash-attn`.

The first cold start downloads ~15 GB of weights to the volume and takes
several minutes. Later cold starts load from the volume in about 1–2 minutes.

## What the worker does

Image → mesh only. Sahib renders the concept image with FLUX and sends it
here; there is deliberately no text encoder and no second image-generation
step. The result is a compact GLB: decimated mesh, 1K WebP textures, PBR
channels. No preview render is produced.

For 3D logos and Arabic lettering this is the wrong path — extrude the vector
so the letterforms stay exact.

## API

```bash
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_url": "https://sahib.li/api/v1/media/blob/<id>",
      "seed": 42,
      "pipeline_type": "1024_cascade",
      "texture_size": 1024,
      "decimation": 30000
    }
  }'
```

Poll `GET /v2/$ENDPOINT_ID/status/<job_id>` until `COMPLETED`.

| Field | Default | Notes |
| --- | --- | --- |
| `image_url` | required | https URL or `data:image/...`; `image_base64` also accepted |
| `seed` | random | same image + seed = same mesh |
| `pipeline_type` | `1024_cascade` | `512`, `1024_cascade`, `1536_cascade` |
| `texture_size` | 1024 | clamped 512–2048 |
| `decimation` | 30000 | clamped 5k–150k triangles |
| `upload_url` | unset | presigned PUT; returns `glb_url` instead of inline base64 |

Output: `glb_b64` (or `glb_url`), `bytes`, `tris`, `ms`, `seed`,
`pipeline_type`, `texture_size`, `decimation`.

## Cost

RunPod serverless 48 GB is $1.22/hr: about **$0.03** warm per model and
**$0.07** cold. A permanently warm worker is ~$890/month, so keep active
workers at 0 until volume justifies it.

## Notes

- TRELLIS.2 is MIT. The GLB export path imports `nvdiffrast` (NVIDIA Source
  Code License, 1-way commercial); re-check those terms before redistributing
  the image itself.
- `o-voxel/third_party/eigen` is a git submodule that RunPod's clone does not
  populate; the Dockerfile fetches the pinned Eigen commit explicitly.
- If the GPU cannot run flash-attn, build with `--build-arg ATTN_BACKEND=xformers`
  and set `ATTN_BACKEND=xformers` on the endpoint.
