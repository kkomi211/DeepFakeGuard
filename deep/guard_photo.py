import os, glob, argparse, math, io
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp

mpfm = mp.solutions.face_mesh
mpcon = mp.solutions.face_mesh_connections

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_bgr(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # PIL 우선 디코딩 (HEIC 등 호환성 향상)
        with open(path, "rb") as f:
            data = f.read()
        try:
            im = Image.open(io.BytesIO(data)).convert("RGB")
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        except Exception:
            raise FileNotFoundError(f"cannot decode image: {path}")
    return img

def save_bgr(path: str, bgr: np.ndarray):
    ext = os.path.splitext(path)[1].lower()
    encode_ext = ".png" if ext not in [".jpg", ".jpeg", ".png", ".webp"] else ext
    ok, buf = cv2.imencode(encode_ext, bgr)
    if not ok: raise RuntimeError(f"encode failed: {path}")
    with open(path, "wb") as f: f.write(buf.tobytes())

def ids_from(conns):
    s=set()
    for a,b in conns: s.add(a); s.add(b)
    return sorted(list(s))

IDS_LIPS  = ids_from(mpcon.FACEMESH_LIPS)
IDS_LEYE  = ids_from(mpcon.FACEMESH_LEFT_EYE)
IDS_REYE  = ids_from(mpcon.FACEMESH_RIGHT_EYE)
IDS_OVAL  = ids_from(mpcon.FACEMESH_FACE_OVAL)
# 일부 버전에서 FACEMESH_NOSE가 없을 수 있음. 없으면 None.
IDS_NOSE  = getattr(mpcon, "FACEMESH_NOSE", None)
IDS_NOSE  = ids_from(IDS_NOSE) if IDS_NOSE is not None else None

def lm_to_xy(lms, W, H, ids):
    return np.array([(int(lms[i].x*W), int(lms[i].y*H)) for i in ids], np.int32)

def fill_hull(mask, pts):
    if pts is None or len(pts) < 3: return
    hull = cv2.convexHull(pts)
    cv2.fillPoly(mask, [hull], 255)

def build_mask(bgr: np.ndarray, lms, parts=("lips","eyes","nose","chin"),
               dilate_px=4, feather=1.2) -> np.ndarray:
    H,W = bgr.shape[:2]
    mask = np.zeros((H,W), np.uint8)

    if "lips" in parts:
        fill_hull(mask, lm_to_xy(lms, W, H, IDS_LIPS))
    if "eyes" in parts:
        fill_hull(mask, lm_to_xy(lms, W, H, IDS_LEYE))
        fill_hull(mask, lm_to_xy(lms, W, H, IDS_REYE))
    if "nose" in parts:
        if IDS_NOSE is not None:
            fill_hull(mask, lm_to_xy(lms, W, H, IDS_NOSE))
        else:
            # 대체: 두 눈 중앙과 입 상부 사이 영역을 코로 근사
            eyes = np.vstack([lm_to_xy(lms, W, H, IDS_LEYE),
                              lm_to_xy(lms, W, H, IDS_REYE)])
            lips = lm_to_xy(lms, W, H, IDS_LIPS)
            eye_center = np.mean(eyes, axis=0)
            lip_top = lips[lips[:,1].argmin()]
            rough = np.array([eye_center,
                              [int(eye_center[0]-0.08*W), int((eye_center[1]+lip_top[1])/2)],
                              [int(eye_center[0]+0.08*W), int((eye_center[1]+lip_top[1])/2)],
                              lip_top], np.int32)
            fill_hull(mask, rough)
    if "chin" in parts:
        oval = lm_to_xy(lms, W, H, IDS_OVAL)
        if len(oval) >= 3:
            ythr = np.quantile(oval[:,1], 0.6)
            chin_pts = oval[oval[:,1] >= ythr]
            fill_hull(mask, chin_pts)

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k, 1)

    m = mask.astype(np.float32)/255.0
    if feather > 0:
        m = cv2.GaussianBlur(m, (0,0), feather)
    return m

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.PSNR(a, b))

def chroma_jitter_in_mask(bgr: np.ndarray, mask01: np.ndarray, amp=3.0, rng=None):
    if rng is None: rng = np.random.default_rng()
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV).astype(np.float32)
    y,u,v = yuv[...,0], yuv[...,1], yuv[...,2]
    du = rng.uniform(-amp, amp)
    dv = rng.uniform(-amp, amp)
    u = np.clip(u + du*mask01, 0, 255)
    v = np.clip(v + dv*mask01, 0, 255)
    out = cv2.cvtColor(np.stack([y,u,v], -1).astype(np.uint8), cv2.COLOR_YUV2BGR)
    return out

def lowfreq_sine_tex(H, W, amp_px=1.5, fx=0.012, fy=0.016, phase=0.0):
    yy, xx = np.mgrid[0:H,0:W]
    s = amp_px*np.sin(2*np.pi*(fx*xx + fy*yy) + phase)
    return s.astype(np.float32)

def micro_grid_tex(H, W, step=6, strength=1.2):
    g = np.zeros((H,W), np.float32)
    g[:, ::step] = strength
    g[::step, :] += strength
    return g

def apply_texture(bgr: np.ndarray, mask01: np.ndarray,
                  amp_lf=1.2, fx=0.01, fy=0.016,
                  grid_step=6, grid_strength=1.0, phase=0.0):
    H,W = bgr.shape[:2]
    lf = lowfreq_sine_tex(H,W,amp_px=amp_lf, fx=fx, fy=fy, phase=phase)[:,:,None]
    hg = micro_grid_tex(H,W,step=grid_step, strength=grid_strength)[:,:,None]
    tex = lf + hg
    out = np.clip(bgr.astype(np.float32) + tex*mask01[:,:,None], 0, 255).astype(np.uint8)
    return out

def micro_warp(bgr: np.ndarray, mask01: np.ndarray, amp_px=1.2, freq=0.008, phase=0.0):
    H,W = bgr.shape[:2]
    yy,xx = np.mgrid[0:H,0:W]
    dx = amp_px*np.sin(2*np.pi*(freq*yy)+phase) * mask01
    map_x = (xx + dx).astype(np.float32)
    map_y = yy.astype(np.float32)
    return cv2.remap(bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def protect_image(bgr: np.ndarray, parts, strength="balanced", seed=1337,
                  dilate=4, feather=1.2, psnr_min=45.0):
    H,W = bgr.shape[:2]
    face = mpfm.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1,
                         min_detection_confidence=0.5)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = face.process(rgb)
    face.close()
    if not res.multi_face_landmarks:
        return bgr, (None, {"note":"no-face"})

    lms = res.multi_face_landmarks[0].landmark
    mask = build_mask(bgr, lms, parts=parts, dilate_px=dilate, feather=feather)
    rng = np.random.default_rng(seed)

    if strength == "stealth":
        chroma_amp = 2.0
        lf_amp     = 0.8
        grid_step  = 7
        grid_str   = 0.6
        warp_amp   = 0.8
    elif strength == "strong":
        chroma_amp = 4.0
        lf_amp     = 1.8
        grid_step  = 5
        grid_str   = 1.6
        warp_amp   = 1.8
    else:  # balanced
        chroma_amp = 3.0
        lf_amp     = 1.2
        grid_step  = 6
        grid_str   = 1.0
        warp_amp   = 1.2

    out = bgr.copy()
    out = chroma_jitter_in_mask(out, mask, amp=chroma_amp, rng=rng)
    out = apply_texture(out, mask, amp_lf=lf_amp, fx=0.01, fy=0.016,
                        grid_step=grid_step, grid_strength=grid_str,
                        phase=rng.uniform(0, 2*np.pi))
    out = micro_warp(out, mask, amp_px=warp_amp, freq=0.008,
                     phase=rng.uniform(0, 2*np.pi))

    cur_psnr = psnr(bgr, out)
    if cur_psnr < psnr_min:
        scale = max(0.4, min(1.0, (cur_psnr/psnr_min)))
        if scale < 1.0:
            out2 = bgr.copy()
            out2 = chroma_jitter_in_mask(out2, mask, amp=chroma_amp*scale, rng=rng)
            out2 = apply_texture(out2, mask, amp_lf=lf_amp*scale, fx=0.01, fy=0.016,
                                 grid_step=grid_step, grid_strength=grid_str*scale,
                                 phase=rng.uniform(0, 2*np.pi))
            out2 = micro_warp(out2, mask, amp_px=warp_amp*scale, freq=0.008,
                              phase=rng.uniform(0, 2*np.pi))
            out = out2
            cur_psnr = psnr(bgr, out)

    meta = {
        "psnr": round(cur_psnr, 4),
        "size": {"h": H, "w": W},
        "parts": list(parts),
        "params": {
            "preset": strength, "dilate": dilate, "feather": feather,
            "chroma_amp": chroma_amp, "lf_amp": lf_amp,
            "grid_step": grid_step, "grid_strength": grid_str,
            "warp_amp": warp_amp, "seed": seed, "psnr_min": psnr_min
        }
    }
    return out, (mask, meta)

def overlay_heatmap(base_bgr: np.ndarray, other_bgr: np.ndarray, alpha=0.55):
    diff = cv2.absdiff(base_bgr, other_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return cv2.addWeighted(heat, alpha, base_bgr, 1-alpha, 0)

def list_images(inp, exts):
    if os.path.isdir(inp):
        paths=[]
        for e in exts:
            paths += glob.glob(os.path.join(inp, f"*{e}"))
        return sorted(paths)
    return [inp]

def main():
    ap = argparse.ArgumentParser(description="Still-image deepfake defense (lips/eyes/nose/chin masking + chroma jitter + midband texture + micro-warp)")
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--outdir", required=True, help="output folder")
    ap.add_argument("--parts", nargs="+", default=["lips","eyes","nose","chin"], help="target parts: lips eyes nose chin")
    ap.add_argument("--preset", choices=["stealth","balanced","strong"], default="balanced")
    ap.add_argument("--dilate", type=int, default=4)
    ap.add_argument("--feather", type=float, default=1.2)
    ap.add_argument("--psnr-min", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--exts", nargs="+", default=[".jpg",".png",".jpeg",".webp"])
    args = ap.parse_args()

    ensure_dir(args.outdir)
    paths = list_images(args.input, args.exts)
    if len(paths)==0:
        print("no images found"); return

    for p in paths:
        try:
            bgr = load_bgr(p)
            out, (mask, meta) = protect_image(
                bgr, parts=set(args.parts), strength=args.preset,
                seed=args.seed, dilate=args.dilate, feather=args.feather,
                psnr_min=args.psnr_min
            )
            base = os.path.splitext(os.path.basename(p))[0]
            save_bgr(os.path.join(args.outdir, f"{base}_guard.png"), out)
            if mask is not None:
                save_bgr(os.path.join(args.outdir, f"{base}_mask.png"),
                         (mask*255).astype(np.uint8))
                ov = overlay_heatmap(bgr, out, alpha=0.55)
                save_bgr(os.path.join(args.outdir, f"{base}_overlay.png"), ov)
            with open(os.path.join(args.outdir, f"{base}_meta.txt"), "w", encoding="utf-8") as f:
                f.write(str(meta))
            print(f"[ok] {p} -> {base}_guard.png  psnr={meta['psnr']}")
        except Exception as e:
            print(f"[error] {p}: {e}")

if __name__ == "__main__":
    main()
