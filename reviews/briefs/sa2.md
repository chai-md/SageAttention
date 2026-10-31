# BRIEF — SageAttention 2.2.0 (SA2) の GB10 帯域最適化（担当: Claude Code / 2 回目）

作業ディレクトリ: `/home/chai/worktrees/sa-claude-sa2`（ブランチ `review-claude-sa2`）
担当範囲: **`csrc/` と `sageattention/` のみ**（`sageattention3_blackwell/` は 1 回目で完了済み。触らない）。

## 0. 前提

| 項目 | 実測値 |
|---|---|
| GPU | NVIDIA GB10, sm_121, **48 SM**, L2 24 MB |
| DRAM | **218–222 GB/s**（copy 実測。公称 273） |
| bf16 GEMM | 77.4 TFLOPS |
| 計測 | CUDA event。`ncu` は `ERR_NVGPUCTRPERM` で使用不可 |
| ビルド | C++20, `arch=compute_121a,code=sm_121a` |

## 1. 既に計測済みの事実（再測定しない。ここから先だけを探す）

`sageattn()`（sm_121 ディスパッチ = per_warp int8 QK + fp8 PV, `pv_accum_dtype="fp32+fp16"`）の内訳
（bf16, H=24, D=128, L=32768, 合計 87.7 ms / SDPA 比 1.89x）:

| ステージ | ms | 実効帯域 | 状態 |
|---|---:|---:|---|
| `k.mean(dim=2)` | 0.85 | 234 GB/s | ピーク。パス削減以外に余地なし |
| `per_warp_int8(q,k)` | 2.57 | 224 GB/s | ピーク |
| `per_channel_fp8(v)`（2 パスに改良済み） | 2.56 | amax 189 / transpose+quant 219 GB/s | ほぼピーク |
| アテンションカーネル | 79.7 | 演算 ~161 TOPS | 帯域律速ではない（後述） |

* V 量子化は 3 パス → 2 パスに既に変更済み（`ChannelAmaxKernel` + `TransposePadPermuteQuantKernel`、
  旧経路とビット一致）。
* `per_channel_fp8` の転置カーネルは `CTA_SIZE=64`（= 1024 スレッド、これ以上は 1024 制限に当たる）。
  `CTA_SIZE=32` も試したが有意差なし。
* **試して失敗した実験**: アテンションカーネルの `CTA_K=128`（`sm89_..._f16_..._inst_buf.cu`）。
  コンパイルは通るが **出力が壊れる**（cos 0.31 / NaN）。smem の swizzle と permutation が
  `CTA_K=64, V_SMEM_STRIDE=CTA_K` 前提で組まれているため。**この方向は深追いしない**。
* カーネルは 226 レジスタ/スレッド、2 CTA/SM（8 warps/SM）。RO アキュムレータが 128 float/thread で
  これが占有率的な上限を作っている。
* `kv` の K/V 読みは L2（24 MB）にほぼ収まる（1 ヘッド K int8 + V fp8 = 8.4 MB @L=32k）。

## 2. あなたのタスク

1. **帯域監査（コードリーディング）**: `sageattn()` 1 回の呼び出しで発生する全テンソルの read/write を
   列挙し、**まだ避けられるパス**を `REPORT.md` に表で書く（テンソル・サイズ・回数・回避案・見込み ms）。
   候補として検証してほしい観点:
   - `sageattn_qk_int8_pv_fp8_cuda` の Python 側: `km_broadcast` / `torch.empty` / `.contiguous()` /
     `o[..., :head_dim_og]` などの小さくない中間テンソルや暗黙コピー
   - `per_warp_int8_cuda` の内部で Q と K がそれぞれ 1 カーネルずつ走る構成（統合で launch 以外に
     何か減るか）、`BLKQ=128/WARPQ=32/BLKK=64` が 48 SM で端数ブロックを出していないか
   - V の 2 パス経路で、amax カーネル（189 GB/s）を 220 GB/s に近づけられるか（1 列 1 スレッド +
     atomicMax の構成。ベクトル化・ブロック縮約・`__ldg` など）
   - dtype 変換・pad・slice による暗黙コピー（`head_dim < 64/128` のパディング経路を含む）
2. **実装**: 見込みの大きいものから 1〜2 件。**必ず自分でビルドして計測**する。
3. **報告**: `REPORT.md` に「監査表 / 実装 / before-after / 却下案と理由」。

## 3. ビルドと計測（このコマンドだけを使う）

**`uv pip install` / `pip install` は禁止**（共有 venv を壊す）。`build_ext --inplace` のみ。

```bash
cd /home/chai/worktrees/sa-claude-sa2
MAX_JOBS=6 TORCH_CUDA_ARCH_LIST=12.1 /home/chai/ComfyUI/.venv/bin/python setup.py build_ext --inplace

cd /tmp
PYTHONPATH=/home/chai/worktrees/sa-claude-sa2 \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-claude-sa2/bench/bench_gb10.py --suite sa2 --stages
PYTHONPATH=/home/chai/worktrees/sa-claude-sa2 \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-claude-sa2/bench/bench_v_quant_ab.py
PYTHONPATH=/home/chai/worktrees/sa-claude-sa2 \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-claude-sa2/bench/verify_v_quant.py
PYTHONPATH=/home/chai/worktrees/sa-claude-sa2 \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-claude-sa2/bench/smoke_test.py
```

ベースライン: L=4096 0.80 ms / L=8192 6.67 ms / L=16384 23.41 ms / **L=32768 87.74 ms**。

## 4. 受け入れ条件

* `smoke_test.py` 全通過（SA2 の cos ≥ 0.9992）。
* V 量子化に触った場合は `verify_v_quant.py` がビット一致のまま。
* **L=32768 が 87.7 ms 未満**、または L=8192 が 6.67 ms 未満。
* `git commit`（ブランチ `review-claude-sa2`）し、`git diff gb10-sm121 --stat` を REPORT に貼る。

## 5. 触ってはいけないもの

* `sageattention3_blackwell/`（1 回目で完了済み）
* `bench/*.py` の既存ファイル（追加は `bench/sa2_extra.py` など別名で）
* 共有 venv へのインストール
* アテンションカーネルのタイル形状変更（上記のとおり壊れる）
