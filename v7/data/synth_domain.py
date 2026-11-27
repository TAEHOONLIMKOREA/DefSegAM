"""v7 — 합성 도메인을 v6 의 데이터 흐름에 붙이는 부분 (PLAN_v7 §3).

v6 의 것을 그대로 쓴다: 세트 읽기 (`enumerate_samples_v6`), RAM 에 올리기 (`GTPool`), 증강 (`augment_v6`), 추출 식 (`compute_domain_sample_weights`).
여기서 더하는 것은 셋이다.

  1. 합성 세트를 합성 도메인의 표본으로 읽는다 — 세트는 사람 GT 데이터셋과 같은 구성이다
     (`visible/{0,1}/{id}.png`, `annotations/{id}.npy`, `index.csv`, `classes.json`).
  2. 합성 표본에는 Copy-Paste 를 걸지 않는 Dataset (합성 레이어에 진짜 조각을 붙이면 흔적이 결함 자리에 생긴다).
  3. 추출 가중: 합성 도메인의 몫을 떼어 놓고, 사람 GT 쪽은 v6 의 식 그대로.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ...common.data.image_utils import normalize_image
from ...v6.data.augmentation import augment_v6
from ...v6.data.data_gt_v6 import DefSegGTDatasetV6, SampleSpecV6, enumerate_samples_v6
from ...v6.data.sampler_helpers import compute_domain_sample_weights
from .. import config_v7 as cfg


def load_synth_samples(root: Path, legacy_keys: set[str]) -> list[SampleSpecV6]:
    """합성 세트 → 합성 도메인의 표본 목록. 사람 GT 와 이름이 겹치거나 그룹 이름이 규칙을 벗어나면 멈춘다."""
    samples = enumerate_samples_v6(root, domain=cfg.DOMAIN_SYNTH)
    bad = sorted({s.group for s in samples if not s.group.startswith(cfg.SYNTH_GROUP_PREFIX)})
    if bad:
        raise RuntimeError(
            f"FATAL: 합성 세트 {root.name} 의 process_code 가 '{cfg.SYNTH_GROUP_PREFIX}' 로 시작하지 않는다: {bad[:5]}"
        )
    dup = sorted({s.key for s in samples} & legacy_keys)
    if dup:
        raise RuntimeError(f"FATAL: 합성 세트 {root.name} 의 표본 이름이 사람 GT 와 겹친다: {dup[:5]}")
    return samples


def synth_set_info(root: Path, n_layers: int) -> dict:
    """어느 세트로 학습했는지 — 이름, 장 수, manifest 의 해시 (같은 이름의 세트가 다시 만들어졌는지 가리는 값)."""
    info = {"name": root.name, "root": str(root), "n_layers": int(n_layers), "manifest_sha256": None, "purpose": None,
            "valid_folds": None}
    man = root / "manifest.json"
    if man.is_file():
        raw = man.read_bytes()
        info["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        try:
            doc = json.loads(raw.decode("utf-8"))
            info["purpose"] = doc.get("purpose")
            # 이 세트를 넣어도 되는 판정 fold — 그 fold 의 검증 레이어를 생성 모델이 보지 않았다 (genaug 의 write_set 이 적는다)
            info["valid_folds"] = doc.get("valid_folds")
        except Exception:
            pass
    return info


class DefSegDatasetV7(DefSegGTDatasetV6):
    """v6 의 Dataset + 합성 표본에는 Copy-Paste 를 걸지 않는다 + 배치에 `is_synth`.

    사람 GT 표본에서는 v6 와 같은 계산이다 (같은 난수, 같은 증강, 같은 Copy-Paste).
    """

    def __getitem__(self, j: int) -> dict:
        i = self.indices[j]
        spec = self.pool.samples[i]
        is_synth = spec.domain == cfg.DOMAIN_SYNTH
        img0, img1, ann = self.pool.i0[i], self.pool.i1[i], self.pool.ann[i]
        if self.training:
            img0, img1, ann = augment_v6(
                img0, img1, ann, training=True,
                rng=self._get_rng(), cp_sampler=None if is_synth else self.cp_sampler,
            )
        return {
            "img0": torch.from_numpy(normalize_image(img0)),
            "img1": torch.from_numpy(normalize_image(img1)),
            "label": torch.from_numpy(ann.astype(np.int64)),
            "group": spec.group,
            "is_synth": torch.tensor(is_synth),
        }


def compute_weights_v7(
    pix: np.ndarray,
    groups: list[str],
    domains: list[str],
    synth_share: float,
    mix: float,
    max_repeat: float,
    diva_max: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """학습 표본의 추출 가중 (합 = 1) 과 도메인별 몫.

    합성 표본이 없으면 v6 의 함수를 그대로 부른다 (같은 값). 있으면:
      p_i = (1 − synth_share) × [v6 의 식으로 낸 사람 GT 쪽 가중]      (사람 GT 표본)
      p_i = synth_share / N_synth                                      (합성 표본 — 고르게)
    사람 GT 쪽의 식 (그룹 균등 + 클래스 균등, 표본당 상한) 과 그 통계에는 합성 표본이 들어가지 않는다.
    """
    dom = np.array(domains)
    is_syn = dom == cfg.DOMAIN_SYNTH
    if not is_syn.any():
        return compute_domain_sample_weights(pix, groups, domains, mix=mix, max_repeat=max_repeat, diva_max=diva_max)
    if not (0.0 < synth_share < 1.0):
        raise ValueError(f"합성 세트를 줬는데 몫이 0–1 사이가 아니다: {synth_share}")
    real = np.where(~is_syn)[0]
    w_real, shares_real = compute_domain_sample_weights(
        np.asarray(pix)[real], [groups[i] for i in real], [domains[i] for i in real],
        mix=mix, max_repeat=max_repeat, diva_max=diva_max,
    )
    p = np.zeros(len(domains), dtype=np.float64)
    p[real] = (1.0 - synth_share) * w_real
    p[is_syn] = synth_share / int(is_syn.sum())
    shares = {d: (1.0 - synth_share) * s for d, s in shares_real.items()}
    shares[cfg.DOMAIN_SYNTH] = float(synth_share)
    return p / p.sum(), shares
