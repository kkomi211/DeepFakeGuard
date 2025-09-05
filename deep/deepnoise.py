import io
import os
import zipfile
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

# Mediapipe (얼굴 랜드마크)
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_conns = mp.solutions.face_mesh_connections

# SSIM (선택)
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

app = FastAPI(title="Deepfake-Protection API", version="1.0.0")

# --------- FaceMesh Detector (전역 1회 생성) ---------
def build_facemesh():
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,          # 눈/입 디테일 향상
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
DETECTOR = build_facemesh()

# --------- 유틸 ----------
def read_upload_to_bgr(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    img_array = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")
    return bgr

def ids_from_connections(conns):
    s = set()
    for a, b in conns:
        s.add(a); s.add(b)
    return sorted(list(s))

IDS_LEFT_EYE   = ids_from_connections(mp_conns.FACEMESH_LEFT_EYE)
IDS_RIGHT_EYE  = ids_from_connections(mp_conns.FACEMESH_RIGHT_EYE)
IDS_LIPS       = ids_from_connections(mp_conns.FACEMESH_LIPS)
IDS_FACE_OVAL  = ids_from_connections(mp_conns.FACEMESH_FACE_OVAL)

def landmarks_to_xy(landmarks, W, H, ids: List[int]) -> np.ndarray:
    pts = [(int(landmarks[i].x * W), int(landmarks[i].y * H)) for i in ids]
    return np.array(pts, dtype=np.int32)

def fill_convex(mask: np.ndarray, pts: np.ndarray):
    if pts is None or len(pts) < 3: return
    hull = cv2.convexHull(pts)
    cv2.fillPoly(mask, [hull], 255)

def build_facial_mask_from_landmarks(bgr: np.ndarray,
                                     landmarks,
                                     dilate_px: int = 5,
                                     feather_sigma: float = 1.0) -> np.ndarray:
    """눈/입/턱/귀 근사 마스크 생성. 반환: float HxW [0,1]"""
    H, W = bgr.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    # 눈/입
    fill_convex(mask, landmarks_to_xy(landmarks, W, H, IDS_LEFT_EYE))
    fill_convex(mask, landmarks_to_xy(landmarks, W, H, IDS_RIGHT_EYE))
    fill_convex(mask, landmarks_to_xy(landmarks, W, H, IDS_LIPS))

    # 턱: FACE_OVAL 하단 40% 근사
    oval_pts = landmarks_to_xy(landmarks, W, H, IDS_FACE_OVAL)
    if len(oval_pts) >= 3:
        y_thr = np.quantile(oval_pts[:, 1], 0.6)
        chin_pts = oval_pts[oval_pts[:, 1] >= y_thr]
        fill_convex(mask, chin_pts)

        # 귀: 좌/우 극점 근사
        x_left_thr  = np.quantile(oval_pts[:, 0], 0.15)
        x_right_thr = np.quantile(oval_pts[:, 0], 0.85)
        left_edge  = oval_pts[oval_pts[:, 0] <= x_left_thr]
        right_edge = oval_pts[oval_pts[:, 0] >= x_right_thr]
        fill_convex(mask, left_edge)
        fill_convex(mask, right_edge)

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k, iterations=1)

    mask_f = mask.astype(np.float32) / 255.0
    if feather_sigma > 0:
        mask_f = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=feather_sigma)
    return mask_f  # [0,1]

def gaussian_lowpass_3(noise: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0: return noise
    out = np.empty_like(noise, dtype=np.float32)
    for c in range(noise.shape[2]):
        out[:, :, c] = cv2.GaussianBlur(noise[:, :, c], (0,0), sigmaX=sigma)
    return out

def compute_psnr(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    return float(cv2.PSNR(img1_bgr, img2_bgr))

def compute_ssim(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> Optional[float]:
    if not HAS_SKIMAGE:
        return None
    a = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    b = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return float(ssim(a, b, channel_axis=2, data_range=1.0))

def overlay_heatmap_on_image(
    base_bgr: np.ndarray,
    diff_bgr: np.ndarray,
    alpha: float = 0.55,
    mask01: np.ndarray = None,
    clip_low_percentile: float = 2.0,    # 1~5 권장: 하위값 잘라 대비↑
    clip_high_percentile: float = 98.0,  # 95~99 권장
    gamma: float = 0.7,                  # 0.5~0.8: 밝은 변화 더 강조
    blur_sigma: float = 1.0,             # 0~2: 점상→면상으로 보이게
    reinforce_edge: bool = False         # True면 경계도 살짝 강조
):
    """
    마스크는 '대상 픽셀을 고르는 용도'로만 사용 (스케일 계산에만 반영).
    실제 히트맵은 마스크로 곱하지 않아 내부 전체에 변화가 드러남.
    """
    # 1) 스칼라 차이
    gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2) 스케일(퍼센타일) 계산: 마스크 내부 기준
    roi = gray[(mask01 > 0.05)] if mask01 is not None else gray.reshape(-1)
    if roi.size == 0:
        roi = gray.reshape(-1)
    lo = np.percentile(roi, clip_low_percentile)
    hi = np.percentile(roi, clip_high_percentile)
    if hi <= lo: hi = lo + 1.0

    # 3) 정규화 + 감마
    norm = np.clip((gray - lo) / (hi - lo), 0, 1)
    if gamma and gamma > 0:
        norm = np.power(norm, gamma)

    # 4) 약한 블러로 점상 노이즈를 면적처럼 보이게
    if blur_sigma and blur_sigma > 0:
        norm = cv2.GaussianBlur(norm, (0, 0), blur_sigma)

    # 5) (선택) 경계 가시성 강화
    if reinforce_edge and mask01 is not None:
        edge = cv2.Laplacian(mask01.astype(np.float32), cv2.CV_32F)
        edge = np.abs(edge)
        edge = cv2.GaussianBlur(edge, (0, 0), 1.0)
        edge = np.clip(edge * 2.0, 0, 1)  # 가중
        norm = np.clip(norm + 0.15 * edge, 0, 1)

    heat = (norm * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (base_bgr.shape[1], base_bgr.shape[0]))

    # 6) 마스크 밖은 색 입히지 않기 (합성 시 마스크로 컷)
    if mask01 is not None:
        m3 = np.repeat((mask01 > 0.05)[..., None], 3, axis=2).astype(np.uint8) * 255
        heat = cv2.bitwise_and(heat, heat, mask=m3[...,0])

    return cv2.addWeighted(heat, alpha, base_bgr, 1 - alpha, 0)




# UAP 캐시: 해상도별 고정 노이즈(픽셀 스케일, [-eps, eps])
UAP_CACHE = {}  # key=(H,W), value=np.ndarray(H,W,3)

def get_uap(H, W, eps_px: float, seed: int = 1337, lowpass_sigma: float = 0.0) -> np.ndarray:
    key = (H, W, round(eps_px, 6), round(lowpass_sigma, 6), seed)
    if key in UAP_CACHE:
        return UAP_CACHE[key]
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-eps_px, eps_px, size=(H, W, 3)).astype(np.float32)
    noise = gaussian_lowpass_3(noise, lowpass_sigma)
    UAP_CACHE[key] = noise
    return noise

# ---------- API ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/protect/face-mask-noise")
def protect_face_mask_noise(
    file: UploadFile = File(...),

    # 노이즈/마스크 파라미터
    eps: float = Query(12.0, ge=0.0, le=64.0, description="노이즈 세기 (픽셀 기준, ±eps)"),
    feather: float = Query(1.0, ge=0.0, le=5.0, description="마스크 경계 블러 sigma"),
    dilate: int = Query(5, ge=0, le=25, description="마스크 팽창 커널 크기"),
    lowpass: float = Query(1.0, ge=0.0, le=5.0, description="노이즈 저주파화 sigma"),
    use_uap: bool = Query(True, description="해상도별 고정 노이즈(유니버설) 사용"),
    seed: int = Query(1337, description="use_uap=True일 때 UAP 시드"),

    # 반환 옵션
    return_mask: bool = Query(True),
    return_overlay: bool = Query(True),
    return_zip: bool = Query(True, description="Zip으로 이미지 묶어서 반환"),
):
    # 1) 입력 로드
    bgr = read_upload_to_bgr(file)
    H, W = bgr.shape[:2]

    # 2) 랜드마크 → 마스크
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = DETECTOR.process(rgb)
    if not res.multi_face_landmarks:
        raise HTTPException(status_code=422, detail="얼굴을 찾지 못했습니다.")
    lm = res.multi_face_landmarks[0].landmark
    mask_f = build_facial_mask_from_landmarks(bgr, lm, dilate_px=dilate, feather_sigma=feather)  # [0,1]
    M = mask_f[:, :, None].astype(np.float32)

    # 3) 노이즈 생성(픽셀 스케일)
    if use_uap:
        noise = get_uap(H, W, eps_px=eps, seed=seed, lowpass_sigma=lowpass)
    else:
        rng = np.random.default_rng()
        noise = rng.uniform(-eps, eps, size=(H, W, 3)).astype(np.float32)
        noise = gaussian_lowpass_3(noise, lowpass)

    # 4) 합성
    noisy = np.clip(bgr.astype(np.float32) + noise * M, 0, 255).astype(np.uint8)

    # 5) 메트릭
    psnr = compute_psnr(bgr, noisy)
    ssim_val = compute_ssim(bgr, noisy)

    # 6) 응답
    if return_zip:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 결과 이미지
            _, enc_noisy = cv2.imencode(".png", noisy)
            zf.writestr("noisy.png", enc_noisy.tobytes())

            # 마스크/오버레이 선택 반환
            if return_mask:
                mask_png = (mask_f * 255).astype(np.uint8)
                _, enc_mask = cv2.imencode(".png", mask_png)
                zf.writestr("mask.png", enc_mask.tobytes())

            if return_overlay:
                diff = cv2.absdiff(bgr, noisy)
                overlay = overlay_heatmap_on_image(
                    base_bgr=bgr,
                    diff_bgr=diff,
                    alpha=0.55,
                    mask01=mask_f,              # ★ 마스크 전달
                    clip_low_percentile=2.0,    # 1~5 권장
                    clip_high_percentile=98.0,  # 95~99 권장
                    gamma=0.7,
                    blur_sigma=1.0,
                    reinforce_edge=False
                )
                _, enc_overlay = cv2.imencode(".png", overlay)
                zf.writestr("overlay.png", enc_overlay.tobytes())


            # 메타(JSON) 파일
            meta = {
                "psnr": round(psnr, 4),
                "ssim": round(ssim_val, 6) if ssim_val is not None else None,
                "height": H, "width": W,
                "params": {
                    "eps": eps, "feather": feather, "dilate": dilate,
                    "lowpass": lowpass, "use_uap": use_uap, "seed": seed
                }
            }
            zf.writestr("meta.json", str(meta).encode("utf-8"))

        buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="deepfake_protect_result.zip"'}
        return StreamingResponse(buf, media_type="application/zip", headers=headers)

    # JSON-only (이미지 바이너리 없이 메트릭만)
    return JSONResponse({
        "psnr": psnr,
        "ssim": ssim_val,
        "height": H,
        "width": W,
        "params": {"eps": eps, "feather": feather, "dilate": dilate, "lowpass": lowpass,
                   "use_uap": use_uap, "seed": seed}
    })