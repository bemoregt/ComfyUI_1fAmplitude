import numpy as np
import torch


class DFT1fAmplitudeNode:
    """
    DFT 분석 실험 노드 (1/f Amplitude + Original Phase → IFFT)

    처리 과정:
    1. 입력 이미지에 FFT 적용
    2. 원본 위상(phase) 추출
    3. 진폭을 1/f 노이즈로 교체 (각 주파수 성분의 크기 = 1 / sqrt(fx² + fy²))
    4. IFFT로 역변환하여 출력
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dc_value": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "DC 성분(직류) 값. 출력 영상의 평균 밝기에 영향을 줍니다.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("1f_phase_image",)
    FUNCTION = "process"
    CATEGORY = "image/frequency"
    DESCRIPTION = "원본 위상 + 1/f 진폭으로 역푸리에 변환합니다. 위상이 구조 정보를 담고 있음을 시각적으로 확인할 수 있습니다."

    def _build_one_over_f_mask(self, H: int, W: int) -> np.ndarray:
        """주파수 격자에서 1/f 진폭 마스크를 생성합니다."""
        fy = np.fft.fftfreq(H)  # shape (H,)
        fx = np.fft.fftfreq(W)  # shape (W,)
        FX, FY = np.meshgrid(fx, fy)  # 각각 (H, W)
        freq_mag = np.sqrt(FX ** 2 + FY ** 2)

        # DC 성분(0,0) 분리 처리
        dc_mask = freq_mag == 0.0
        freq_mag[dc_mask] = 1.0          # 나눗셈 오류 방지용 임시 값

        one_over_f = 1.0 / freq_mag
        one_over_f[dc_mask] = 0.0        # DC 성분은 별도로 제어
        return one_over_f

    def process(self, image: torch.Tensor, dc_value: float):
        # image: (B, H, W, C), float32, [0, 1]
        B, H, W, C = image.shape
        one_over_f = self._build_one_over_f_mask(H, W)  # (H, W)

        results = []
        for b in range(B):
            img_np = image[b].cpu().numpy()  # (H, W, C)
            channels = []

            for c in range(C):
                channel = img_np[:, :, c].astype(np.float64)

                # FFT → 위상 추출
                fft = np.fft.fft2(channel)
                phase = np.angle(fft)

                # 1/f 진폭 × 원본 위상으로 새 스펙트럼 구성
                # DC는 dc_value를 역변환 후 평균 밝기로 사용
                new_fft = one_over_f * np.exp(1j * phase)
                new_fft[0, 0] = dc_value * H * W  # DC 성분 직접 설정

                # IFFT → 실수부 취득
                reconstructed = np.real(np.fft.ifft2(new_fft))

                # [0, 1] 범위로 정규화
                r_min, r_max = reconstructed.min(), reconstructed.max()
                if r_max > r_min:
                    reconstructed = (reconstructed - r_min) / (r_max - r_min)
                else:
                    reconstructed = np.full_like(reconstructed, dc_value)

                channels.append(reconstructed)

            result = np.stack(channels, axis=-1).astype(np.float32)  # (H, W, C)
            results.append(result)

        output = torch.from_numpy(np.stack(results, axis=0))  # (B, H, W, C)
        return (output,)


# ─── 추가 노드: Random Phase (비교용) ────────────────────────────────────────

class DFTRandomPhaseNode:
    """
    DFT 분석 실험 노드 (Original Amplitude + Random Phase → IFFT)

    원본 진폭은 유지하되 위상을 랜덤화합니다.
    위상이 파괴되면 영상 구조도 붕괴됨을 확인할 수 있습니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2 ** 31 - 1,
                        "tooltip": "랜덤 위상 생성 시드",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("random_phase_image",)
    FUNCTION = "process"
    CATEGORY = "image/frequency"
    DESCRIPTION = "원본 진폭 + 랜덤 위상으로 역푸리에 변환합니다. 위상 파괴 시 구조가 붕괴됨을 보여줍니다."

    def process(self, image: torch.Tensor, seed: int):
        B, H, W, C = image.shape
        rng = np.random.default_rng(seed)

        results = []
        for b in range(B):
            img_np = image[b].cpu().numpy()
            channels = []

            for c in range(C):
                channel = img_np[:, :, c].astype(np.float64)
                fft = np.fft.fft2(channel)
                amplitude = np.abs(fft)

                # 랜덤 위상 생성 [-π, π)
                random_phase = rng.uniform(-np.pi, np.pi, size=(H, W))

                new_fft = amplitude * np.exp(1j * random_phase)
                reconstructed = np.real(np.fft.ifft2(new_fft))

                r_min, r_max = reconstructed.min(), reconstructed.max()
                if r_max > r_min:
                    reconstructed = (reconstructed - r_min) / (r_max - r_min)
                else:
                    reconstructed = np.zeros_like(reconstructed)

                channels.append(reconstructed.astype(np.float32))

            results.append(np.stack(channels, axis=-1))

        output = torch.from_numpy(np.stack(results, axis=0))
        return (output,)


# ─── 노드 등록 ────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "DFT1fAmplitude": DFT1fAmplitudeNode,
    "DFTRandomPhase": DFTRandomPhaseNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DFT1fAmplitude": "DFT: 1/f Amplitude + Original Phase",
    "DFTRandomPhase": "DFT: Original Amplitude + Random Phase",
}
