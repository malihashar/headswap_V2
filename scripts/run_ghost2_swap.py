#!/usr/bin/env python3
"""Run GHOST 2.0 head swap with multi-person face selection.

Expects GHOST2_ROOT (clone of ai-forever/ghost-2.0) to be set up via
scripts/setup_ghost2_colab.sh. Source = identity donor, target = body/scene.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _select_face(dets, policy: str, index: int):
    if not dets:
        raise RuntimeError("No face detected in target/body image.")
    policy = (policy or "largest").strip().lower()
    if policy == "index":
        i = int(index)
        if i < 0 or i >= len(dets):
            raise RuntimeError(f"BODY_FACE_INDEX={i} out of range (n={len(dets)})")
        return dets[i]
    if policy == "rightmost":
        return max(dets, key=lambda d: float(d.bbox[0] + d.bbox[2]) / 2.0)
    if policy == "leftmost":
        return min(dets, key=lambda d: float(d.bbox[0] + d.bbox[2]) / 2.0)
    # largest
    return max(
        dets,
        key=lambda d: float((d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghost-root", type=Path, default=None)
    parser.add_argument("--source", type=Path, required=True, help="Identity face/head")
    parser.add_argument("--target", type=Path, required=True, help="Body / scene")
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--face-policy", default="largest")
    parser.add_argument("--face-index", type=int, default=0)
    parser.add_argument("--use-kandi", action="store_true")
    parser.add_argument("--output-long-side", type=int, default=0)
    args = parser.parse_args()

    ghost_root = Path(
        args.ghost_root
        or os.environ.get("GHOST2_ROOT")
        or "/content/ghost-2.0"
    ).resolve()
    if not ghost_root.is_dir():
        raise SystemExit(f"GHOST2_ROOT not found: {ghost_root}")

    os.chdir(ghost_root)
    sys.path.insert(0, str(ghost_root))

    import cv2
    import numpy as np
    import torch
    import onnxruntime as ort
    from PIL import Image
    from insightface.app import FaceAnalysis
    from omegaconf import OmegaConf
    from torchvision import transforms
    from torchvision.transforms.functional import rgb_to_grayscale

    from src.utils.crops import copy_head_back, norm_crop, wide_crop_face
    from src.utils.inference import normalize_and_torch
    from src.utils.inpainter import LamaInpainter
    from src.utils.preblending import calc_pseudo_target_bg, post_inpainting
    from train_aligner import AlignerModule
    from train_blender import BlenderModule
    from repos.stylematte.stylematte.models import StyleMatte

    cfg_a = OmegaConf.load(ghost_root / "configs" / "aligner.yaml")
    cfg_b = OmegaConf.load(ghost_root / "configs" / "blender.yaml")
    ckpt_a = ghost_root / "aligner_checkpoints" / "aligner_1020_gaze_final.ckpt"
    ckpt_b = ghost_root / "blender_checkpoints" / "blender_lama.ckpt"

    print(f"→ Loading Aligner from {ckpt_a}", flush=True)
    aligner = AlignerModule(cfg_a)
    aligner.load_state_dict(torch.load(ckpt_a, map_location="cpu"), strict=False)
    aligner.eval().cuda()

    print(f"→ Loading Blender from {ckpt_b}", flush=True)
    blender = BlenderModule(cfg_b)
    blender.load_state_dict(
        torch.load(ckpt_b, map_location="cpu")["state_dict"], strict=False
    )
    blender.eval().cuda()

    inpainter = LamaInpainter()
    app = FaceAnalysis(providers=["CUDAExecutionProvider"], allowed_modules=["detection"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    pipe = None
    if args.use_kandi:
        from diffusers import AutoPipelineForInpainting

        pipe = AutoPipelineForInpainting.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint",
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()

    segment_model = StyleMatte()
    segment_model.load_state_dict(
        torch.load(
            ghost_root
            / "repos"
            / "stylematte"
            / "stylematte"
            / "checkpoints"
            / "stylematte_synth.pth",
            map_location="cpu",
        )
    )
    segment_model = segment_model.cuda().eval()

    parsings_session = ort.InferenceSession(
        str(ghost_root / "weights" / "segformer_B5_ce.onnx"),
        providers=[("CUDAExecutionProvider", {})],
    )
    input_name = parsings_session.get_inputs()[0].name
    output_names = [o.name for o in parsings_session.get_outputs()]
    mean = np.array([0.51315393, 0.48064056, 0.46301059])[None, :, None, None]
    std = np.array([0.21438347, 0.20799829, 0.20304542])[None, :, None, None]

    def infer_parsing(img):
        return torch.tensor(
            parsings_session.run(
                output_names,
                {
                    input_name: (
                        ((img[:, [2, 1, 0], ...] / 2 + 0.5).cpu().detach().numpy() - mean)
                        / std
                    ).astype(np.float32)
                },
            )[0],
            device="cuda",
            dtype=torch.float32,
        )

    def calc_mask(img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).cuda()
        if img.max() > 1.0:
            img = img / 255.0
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        input_t = normalize(img).unsqueeze(0).float()
        with torch.no_grad():
            out = segment_model(input_t)
        return out[0][0]

    def process_img(img_path: Path, *, target: bool = False, face_det=None):
        full_frames = cv2.imread(str(img_path))
        if full_frames is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        # Optional long-side resize for target body.
        if target and args.output_long_side and args.output_long_side > 0:
            h, w = full_frames.shape[:2]
            long_side = max(h, w)
            if long_side != args.output_long_side:
                scale = float(args.output_long_side) / float(long_side)
                full_frames = cv2.resize(
                    full_frames,
                    (int(round(w * scale)), int(round(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
        dets = app.get(full_frames)
        if face_det is None:
            face_det = _select_face(dets, "largest", 0)
        else:
            # Re-match selected box after possible resize via IoU on current dets.
            face_det = _select_face(dets, args.face_policy, args.face_index)
        kps = face_det["kps"]
        wide = wide_crop_face(full_frames, kps, return_M=target)
        if target:
            wide, M = wide
        arc = norm_crop(full_frames, kps)
        mask = calc_mask(wide)
        arc = normalize_and_torch(arc)
        wide = normalize_and_torch(wide)
        if target:
            return wide, arc, mask, full_frames, M, len(dets)
        return wide, arc, mask, len(dets)

    print(
        f"→ Faces: policy={args.face_policy} index={args.face_index}",
        flush=True,
    )
    wide_source, arc_source, mask_source, n_src = process_img(args.source)
    wide_target, arc_target, mask_target, full_frames, M, n_tgt = process_img(
        args.target, target=True
    )
    print(f"✓ source_faces={n_src} target_faces={n_tgt}", flush=True)

    wide_source = wide_source.unsqueeze(1)
    arc_source = arc_source.unsqueeze(1)

    X_dict = {
        "source": {
            "face_arc": arc_source,
            "face_wide": wide_source * mask_source,
            "face_wide_mask": mask_source,
        },
        "target": {
            "face_arc": arc_target,
            "face_wide": wide_target * mask_target,
            "face_wide_mask": mask_target,
        },
    }

    with torch.no_grad():
        output = aligner(X_dict)

    target_parsing = infer_parsing(wide_target)
    pseudo_norm_target = calc_pseudo_target_bg(wide_target, target_parsing)
    soft_mask = calc_mask(
        ((output["fake_rgbs"] * output["fake_segm"])[0, [2, 1, 0], :, :] + 1) / 2
    )[None]
    new_source = output["fake_rgbs"] * soft_mask[:, None, ...] + pseudo_norm_target * (
        1 - soft_mask[:, None, ...]
    )

    blender_input = {
        "face_source": new_source,
        "gray_source": rgb_to_grayscale(new_source[0][[2, 1, 0], ...]).unsqueeze(0),
        "face_target": wide_target,
        "mask_source": infer_parsing(output["fake_rgbs"] * output["fake_segm"]),
        "mask_target": target_parsing,
        "mask_source_noise": None,
        "mask_target_noise": None,
        "alpha_source": soft_mask,
    }
    output_b = blender(blender_input, inpainter=inpainter)
    np_output = np.uint8(
        (output_b["oup"][0].detach().cpu().numpy().transpose((1, 2, 0))[:, :, ::-1] / 2 + 0.5)
        * 255
    )
    result = copy_head_back(np_output, full_frames[..., ::-1], M)
    if args.use_kandi and pipe is not None:
        result = post_inpainting(result, output, full_frames, M, infer_parsing, pipe)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(args.save_path)
    print(f"✓ saved {args.save_path} size={result.shape[1]}x{result.shape[0]}", flush=True)


if __name__ == "__main__":
    main()
