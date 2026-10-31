# BRIEF — SageAttention 2.2.0 の GB10 帯域最適化（担当: Antigravity / agy）

作業ディレクトリ: `/home/chai/worktrees/sa-agy`（ブランチ `review-agy`、`gb10-sm121` から分岐）
あなたの担当は **SA2 側（`csrc/`, `sageattention/`）のみ**。`sageattention3_blackwell/` は触らない。

## 0. 前提（GB10 の帯域モデル）

| 項目 | 実測値 |
|---|---|
| GPU | NVIDIA GB10, sm_121, **48 SM**, L2 **24 MB** |
| DRAM 帯域 | **218–222 GB/s**（公称 273、copy 実測） |
| bf16 GEMM | 77.4 TFLOPS |
| 計測 | CUDA event。`ncu` は `ERR_NVGPUCTRPERM` で**使用不可** |
| 環境 | torch 2.15.0+cu132 / CUDA 13.0 / python 3.13 (`/home/chai/ComfyUI/.venv/bin/python`) |

**帯域が狭いことが全て**。1 要素あたりのパス数（read/write 回数）を減らすのが唯一のレバー。
すでに 220 GB/s に張り付いているステージは、パスを減らす以外に速くならない。

## 1. 既に実装済み（再発明しない）

* ビルドは C++20 + `sm_121a`。
* `per_channel_fp8(smooth_v=False)`: V の量子化を 3 パス → 2 パスに変更済み
  （`ChannelAmaxKernel` + `TransposePadPermuteQuantKernel` を追加、旧経路とはビット一致）。
* sm_121 のディスパッチ（`per_warp` + `pv_accum_dtype="fp32+fp16"`）は上流のまま。
  実測では `per_warp+fp32` が僅差（1–4%）で最速だが、上流既定を尊重して据え置き。

現状（bf16, H=24, D=128, L=32768, 合計 **87.7 ms**）:

| ステージ | ms | 帯域効率 |
|---|---:|---|
| アテンションカーネル（int8 QK + fp8 PV） | 79.7 | 演算 ~161 TOPS（int8 ピークの 6–7 割） |
| `k.mean(dim=2)` | 0.85 | 234 GB/s = ほぼピーク |
| `per_warp_int8(q,k)` | 2.57 | 224 GB/s = ほぼピーク |
| `per_channel_fp8(v)` | 2.56 | 2 パス、146 GB/s（まだ余裕あり） |

## 2. あなたのタスク

1. **帯域監査**: `sageattn()`（sm_121 ディスパッチ）1 回の呼び出しについて、
   Q/K/V の read/write を全て列挙し「避けられるパス」を `REPORT.md` に表で書く
   （テンソル・サイズ・回数・回避案・見込み ms）。
   特に次を検証してほしい:
   - `per_warp_int8` の Q/K 量子化カーネルのグリッド/ブロック設定（BLKQ=128, WARPQ=32, BLKK=64）が
     GB10 の 48 SM に合っているか（無駄な端数ブロック、占有率的に損していないか）
   - `km.repeat_interleave`（GQA 時）や `km` の dtype 変換など、小さくない中間テンソル
   - 追加した `TransposePadPermuteQuantKernel` の `CTA_SIZE=64` が最適か（32/128 を試す）
   - アテンションカーネルの K/V ロード `cp_async::PrefetchMode::kNoPrefetch` を `kPrefetch` にした場合
   - 量子化カーネルの `BLOCK_SIZE`（64/128/256）の帯域への影響
2. **実装**: 効果の見込めるものを 1〜2 件実装し、**自分でビルドして計測**する。
3. **報告**: `REPORT.md` に「監査表 / 実装 / before-after / 却下案と理由」。

## 3. ビルドと計測（このコマンドだけを使う）

**絶対に `uv pip install` / `pip install` を使わないこと**（共有 venv を壊す）。
`setup.py build_ext --inplace` で自分の worktree だけにビルドする。

```bash
# ビルド（~1.5 分）
cd /home/chai/worktrees/sa-agy
MAX_JOBS=6 TORCH_CUDA_ARCH_LIST=12.1 /home/chai/ComfyUI/.venv/bin/python setup.py build_ext --inplace

# 計測（PYTHONPATH で worktree のビルドを使う）
cd /tmp
PYTHONPATH=/home/chai/worktrees/sa-agy \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-agy/bench/bench_gb10.py --suite sa2 --stages

# V 量子化だけの A/B（20 iter）
PYTHONPATH=/home/chai/worktrees/sa-agy \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-agy/bench/bench_v_quant_ab.py

# 数値検証（V 量子化の新旧ビット一致 / 全体スモーク）
PYTHONPATH=/home/chai/worktrees/sa-agy \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-agy/bench/verify_v_quant.py
PYTHONPATH=/home/chai/worktrees/sa-agy \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-agy/bench/smoke_test.py
```

ベースライン（bf16, SDPA 比）: L=4096 0.80 ms / L=8192 6.67 ms / L=16384 23.41 ms / **L=32768 87.74 ms**。

## 4. 受け入れ条件（これを満たさない変更は revert して報告）

* `smoke_test.py` が全ケース通過（cos ≥ 0.9992 for SA2）。
* `verify_v_quant.py` がビット一致のまま（V 量子化に触った場合）。
* **L=32768 が 87.7 ms を下回る**か、L=8192 が 6.67 ms を下回ること。
* 変更は `git commit`（ブランチ `review-agy`）し、`git diff gb10-sm121 --stat` を REPORT に貼る。

## 5. 触ってはいけないもの

* `sageattention3_blackwell/`（SA3 側 = 別担当）
* `bench/*.py` の既存ファイル（追加は `bench/sa2_extra.py` のように別名で）
* 共有 venv（`/home/chai/ComfyUI/.venv`）へのインストール
* GPU を長時間占有する計測（1 回のベンチは ~1 分。並行して他エージェントが動いている）

## 6. 参考

* `GB10_PORTING.md`（技術レポート、既存の計測値と判断理由）
* `HANDOFF.md`（環境の落とし穴）
