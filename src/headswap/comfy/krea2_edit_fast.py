"""Quality-preserving speedups for comfyui-krea2edit's per-step forward.

The stock ``krea2_edit_forward`` recomputes *every* diffusion step:

* source latent → patch tokens (``m.first``)
* text-fusion + text-MLP on the (unchanging) conditioning
* RoPE freqs for the full [text|refs|target] layout
* dense ``attn_bias`` for ``ref_boost``

None of that depends on the denoising timestep. Only ``tgt_img``, ``tvec``,
and the DiT blocks do. Caching the static tensors removes repeated GPU/Python
work without changing numerics (same tensors reused).

Also: ``ref_boost != 1`` builds an attention logit bias. That forces Comfy's
masked attention path (FlashAttention often cannot apply arbitrary bias),
which is expected for identity fidelity — we do **not** disable ref_boost.
"""
from __future__ import annotations

import sys
from typing import Any, Callable


_INSTALLED = False


def install_krea2_edit_static_cache() -> dict[str, Any]:
    """
    Monkey-patch ``krea2_edit_forward`` in the loaded custom-node module.

    Safe to call multiple times. Returns status dict.
    """
    global _INSTALLED
    info: dict[str, Any] = {"installed": False, "module": None}
    if _INSTALLED:
        info["installed"] = True
        info["already"] = True
        return info

    target_mod = None
    for mod in list(sys.modules.values()):
        fn = getattr(mod, "krea2_edit_forward", None)
        if fn is None:
            continue
        if getattr(mod, "_headswap_edit_cache", False):
            target_mod = mod
            break
        # Prefer the real custom-node module (has _fit_src + ModelPatch class).
        if hasattr(mod, "_fit_src") and hasattr(mod, "Krea2EditModelPatch"):
            target_mod = mod
            break
        if target_mod is None:
            target_mod = mod

    if target_mod is None or not hasattr(target_mod, "krea2_edit_forward"):
        info["error"] = "krea2_edit_forward_not_loaded"
        return info

    if getattr(target_mod, "_headswap_edit_cache", False):
        _INSTALLED = True
        info["installed"] = True
        info["already"] = True
        info["module"] = getattr(target_mod, "__name__", None)
        return info

    orig: Callable = target_mod.krea2_edit_forward
    _to_4d = target_mod._to_4d
    _fit_src = target_mod._fit_src
    _imgids = target_mod._imgids
    _imgids_offset = target_mod._imgids_offset
    _ref_attn_bias = target_mod._ref_attn_bias

    # Cache keyed by geometry + conditioning identity; cleared when H/W or src ids change.
    cache: dict[str, Any] = {"key": None, "payload": None}

    def _src_key(src_latent) -> tuple:
        src_list = src_latent if isinstance(src_latent, (list, tuple)) else [src_latent]
        keys = []
        for sl in src_list:
            t = _to_4d(sl)
            keys.append((id(t), tuple(t.shape), str(t.dtype), str(t.device)))
        return tuple(keys)

    def cached_forward(
        m,
        x,
        timesteps,
        context,
        src_latent,
        transformer_options,
        ref_boost=1.0,
        ref_boost_a=1.0,
        ref_boost_mask=None,
        ref_native=False,
        pos_mode="anchor",
    ):
        import comfy.ldm.common_dit
        import torch
        from comfy.ldm.flux.layers import timestep_embedding
        from einops import rearrange

        patch = m.patch
        temporal = x.ndim == 5
        if temporal:
            b5, c5, t5, h5, w5 = x.shape
            x = _to_4d(x)
        bs, c, H_orig, W_orig = x.shape

        x = comfy.ldm.common_dit.pad_to_patch_size(
            x, (patch, patch), padding_mode="replicate"
        )
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        # Timestep-invariant cache key (conditioning + refs + geometry + boosts).
        try:
            ctx_ptr = int(context.data_ptr())
        except Exception:
            ctx_ptr = id(context)
        ctx_id = (
            id(context),
            tuple(context.shape) if hasattr(context, "shape") else None,
            ctx_ptr,
        )
        key = (
            id(m),
            H,
            W,
            h_,
            w_,
            pos_mode,
            bool(ref_native),
            float(ref_boost),
            float(ref_boost_a),
            id(ref_boost_mask) if ref_boost_mask is not None else None,
            _src_key(src_latent),
            ctx_id,
        )

        payload = cache["payload"] if cache["key"] == key else None
        if payload is None:
            src_list = (
                src_latent if isinstance(src_latent, (list, tuple)) else [src_latent]
            )
            srcs = []
            for sl in src_list:
                src = _to_4d(sl).to(device=x.device, dtype=x.dtype)
                if src.shape[0] != bs:
                    src = src[:1].expand(bs, *src.shape[1:])
                if (not ref_native) and src.shape[-2:] != (H, W):
                    src = _fit_src(src, H, W).to(x.dtype)
                srcs.append(
                    comfy.ldm.common_dit.pad_to_patch_size(
                        src, (patch, patch), padding_mode="replicate"
                    )
                )
            src_grids = [(s_.shape[-2] // patch, s_.shape[-1] // patch) for s_ in srcs]

            ctx = m._unpack_context(context)
            src_imgs = [
                m.first(
                    rearrange(
                        s_,
                        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                        ph=patch,
                        pw=patch,
                    )
                )
                for s_ in srcs
            ]
            ctx = m.txtfusion(ctx, mask=None, transformer_options=transformer_options)
            ctx = m.txtmlp(ctx)

            txtlen = ctx.shape[1]
            # tgtlen depends on current x grid — known from h_,w_
            tgtlen = h_ * w_
            srclen = sum(si.shape[1] for si in src_imgs)

            if pos_mode == "stride1" and ref_native:
                ref_ids = [
                    _imgids_offset(bs, i + 1, gh, gw, h_, w_, x.device)
                    for i, (gh, gw) in enumerate(src_grids)
                ]
            else:
                ref_ids = [
                    _imgids(bs, i + 1, gh, gw, x.device)
                    for i, (gh, gw) in enumerate(src_grids)
                ]
            pos = torch.cat(
                [torch.zeros(bs, txtlen, 3, device=x.device, dtype=torch.float32)]
                + ref_ids
                + [_imgids(bs, 0, h_, w_, x.device)],
                dim=1,
            )
            freqs = m.pe_embedder(pos)

            attn_bias = None
            if ref_boost != 1.0 or ref_boost_a != 1.0:
                boosts = [ref_boost_a] * (len(src_imgs) - 1) + [ref_boost]
                # dtype follows activations; use x.dtype as stand-in until combined exists
                attn_bias = _ref_attn_bias(
                    boosts,
                    ref_boost_mask,
                    txtlen,
                    [si.shape[1] for si in src_imgs],
                    tgtlen,
                    src_grids,
                    x.device,
                    x.dtype,
                )

            payload = {
                "context": ctx,
                "src_imgs": src_imgs,
                "src_grids": src_grids,
                "txtlen": txtlen,
                "srclen": srclen,
                "freqs": freqs,
                "attn_bias": attn_bias,
            }
            cache["key"] = key
            cache["payload"] = payload

        context_c = payload["context"]
        src_imgs = payload["src_imgs"]
        txtlen = payload["txtlen"]
        srclen = payload["srclen"]
        freqs = payload["freqs"]
        attn_bias = payload["attn_bias"]

        tgt_img = m.first(
            rearrange(
                x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch
            )
        )
        t = m.tmlp(
            timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype)
        )
        tvec = m.tproj(t)

        combined = torch.cat([context_c] + src_imgs + [tgt_img], dim=1)
        # If bias was built with a different dtype than combined, cast once (rare).
        if attn_bias is not None and attn_bias.dtype != combined.dtype:
            attn_bias = attn_bias.to(dtype=combined.dtype)
            payload["attn_bias"] = attn_bias

        for block in m.blocks:
            combined = block(
                combined,
                tvec,
                freqs,
                attn_bias,
                transformer_options=transformer_options,
            )

        final = m.last(combined, t)
        out = final[:, txtlen + srclen : txtlen + srclen + tgt_img.shape[1], :]
        out = rearrange(
            out,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_,
            w=w_,
            ph=patch,
            pw=patch,
            c=m.channels,
        )
        out = out[:, :, :H_orig, :W_orig]
        if temporal:
            out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
        return out

    cached_forward._headswap_orig = orig  # type: ignore[attr-defined]
    target_mod.krea2_edit_forward = cached_forward
    target_mod._headswap_edit_cache = True
    _INSTALLED = True
    info["installed"] = True
    info["module"] = getattr(target_mod, "__name__", None)
    print(f"[krea2] static edit-forward cache installed on {info['module']}")
    return info


def clear_krea2_edit_static_cache() -> None:
    """Drop cached tensors (e.g. between images). Patch stays installed."""
    for mod in list(sys.modules.values()):
        fn = getattr(mod, "krea2_edit_forward", None)
        if fn is None:
            continue
        # Reach into closure cache if present
        cell = getattr(fn, "__closure__", None)
        if not cell:
            continue
        for c in cell:
            try:
                obj = c.cell_contents
            except ValueError:
                continue
            if isinstance(obj, dict) and "payload" in obj and "key" in obj:
                obj["key"] = None
                obj["payload"] = None
