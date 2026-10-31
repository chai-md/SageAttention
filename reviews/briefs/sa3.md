# BRIEF — SageAttention3 (FP4) の GB10 帯域最適化（担当: Claude Code）

作業ディレクトリ: `/home/chai/worktrees/sa-claude`（ブランチ `review-claude`、`gb10-sm121` から分岐）
あなたの担当は **`sageattention3_blackwell/` 配下のみ**。SA2 側（`csrc/`, `sageattention/`）は触らない。

## 0. 前提（GB10 の帯域モデル）

| 項目 | 実測値 |
|---|---|
| GPU | NVIDIA GB10, sm_121, **48 SM**, L2 **24 MB** |
| DRAM 帯域 | **218–222 GB/s**（公称 273、copy 実測） |
| bf16 GEMM | 77.4 TFLOPS（cuBLAS 8192³） |
| 計測 | CUDA event。`ncu` は `ERR_NVGPUCTRPERM` で**使用不可** |
| 環境 | torch 2.15.0+cu132 / CUDA 13.0 / python 3.13 (`/home/chai/ComfyUI/.venv/bin/python`) |

**帯域が狭いことが全て**。1 要素あたりのパス数（read/write 回数）を減らすのが唯一のレバー。
すでに 220 GB/s に張り付いているステージは、パスを減らす以外に速くならない。

## 1. 既に実装済み（再発明しない）

* `launch.h`: グリッドの 170 SM ハードコード → デバイス問い合わせ（GB10=48）。`SAGE3_NUM_SM` で上書き可。
* FP4 量子化カーネルに `mean` 引数を追加し、`q-qm` / `k-km` の実体化をやめた（`preprocess_qkv`）。
* `delta_s` から `qm·km`（キー方向の定数 = softmax 不変）を削除。

現状の内訳（bf16, H=24, L=16384, D=128, 合計 **21.9 ms**）:

| ステージ | ms | 備考 |
|---|---:|---|
| FP4 アテンションカーネル | 16.1 | 全体の 74% |
| `delta_s` matmul (bf16) | 0.92 | `qm @ k^T` |
| `delta_s` → fp32 変換 | **1.31** | `.to(torch.float32)` だけ |
| `k.mean` | 0.46 | |
| `qm`（128 トークン平均） | 0.43 | |
| quant Q/K/V | 1.69 | 3 カーネル |

## 2. あなたのタスク

1. **帯域監査**: 1 回の `sageattn3_blackwell()` 呼び出しで走る全テンソルの read/write を列挙し、
   「避けられるパス」を特定して `REPORT.md` に表で書く（テンソル・サイズ・回数・回避案・見込み ms）。
2. **実装**: 上位 1〜2 件を実装し、**自分でビルドして計測**する。第一候補は
   **`delta_s` の bf16 化**（`.float()` 1.31 ms + カーネル側 fp32 タイル read の半減で計 ~1.7 ms の見込み）:
   - `blackwell/kernel_traits.h` の `ArrayEngine<float>` / `SmemLayoutDS`
   - `blackwell/mainloop_tma_ws.h` の `add_delta_s`（`recast<float4>` での読み出し）
   - `blackwell/api.cu` の `delta_s` 用 TMA ディスクリプタと `params.delta_s_ptr` の扱い
   - `sageattn3/api.py` の `.to(torch.float32)` を外す
   代替候補: `qm` 計算の量子化カーネル内融合（-0.43 ms 見込み）。
3. **報告**: `REPORT.md` に「監査表 / 実装した変更 / before-after 実測 / 却下した案と理由」を書く。

## 3. ビルドと計測（このコマンドだけを使う）

**絶対に `uv pip install` / `pip install` を使わないこと**（共有 venv を壊す）。
`setup.py build_ext --inplace` で自分の worktree だけにビルドする。CUTLASS は `csrc/cutlass` に
シンボリックリンク済み。

```bash
# ビルド（初回 ~6 分、.cu 変更ごとに再実行）
cd /home/chai/worktrees/sa-claude/sageattention3_blackwell
MAX_JOBS=6 /home/chai/ComfyUI/.venv/bin/python setup.py build_ext --inplace

# 計測（PYTHONPATH で worktree のビルドを使う）
cd /tmp
PYTHONPATH=/home/chai/worktrees/sa-claude/sageattention3_blackwell \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-claude/bench/bench_gb10.py --suite sa3

# ステージ内訳
PYTHONPATH=/home/chai/worktrees/sa-claude/sageattention3_blackwell \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-claude/bench/bench_sa3_stages.py
```

ベースライン（bf16, SDPA 比）: L=4096 0.67 ms / **L=8192 6.85 ms** / **L=16384 21.94 ms**、
cos 0.9818（fp32 SDPA リファレンス比）。

## 4. 受け入れ条件（これを満たさない変更は revert して報告）

* 出力の `cos` が **0.9810 以上**（現状 0.9818、L=4096/8192/16384 で確認）。
* 既存の API シグネチャ `sageattn3_blackwell(q, k, v, is_causal=..., per_block_mean=...)` を維持。
* 入力テンソルを in-place で書き換えない（現状の非破壊性を保つ）。
* **L=16384 が 21.9 ms を下回る**こと（1 つでも改善を入れた場合）。
* 変更は `git commit`（ブランチ `review-claude`）し、`git diff gb10-sm121 --stat` を REPORT に貼る。

## 5. 触ってはいけないもの

* `csrc/`, `sageattention/`, `setup.py`（SA2 側 = 別担当）
* `bench/*.py` の既存ファイル（追加したい場合は `bench/sa3_extra.py` のように別名で）
* 共有 venv（`/home/chai/ComfyUI/.venv`）へのインストール
* GPU を長時間占有する計測（1 回のベンチは ~1 分。並行して他エージェントが動いている）

## 6. 参考

* `GB10_PORTING.md`（このブランチの技術レポート、既存の計測値と判断理由）
* `HANDOFF.md`（環境の落とし穴）
