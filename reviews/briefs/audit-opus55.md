# BRIEF — GB10 対応の第三者レビュー（Claude Opus 5.5 / 監査モード）

作業ディレクトリ: `/home/chai/worktrees/sa-opus55`（ブランチ `review-opus55`、`gb10-sm121` = `f40986f` から分岐）
これは**実装ラウンドではなく監査**。読む・測る・指摘するのが主目的。修正は「明らかに安全で小さい」ものだけ。

## 0. 対象と現状

SageAttention 2.2.0 + SageAttention3 を DGX Spark (GB10, sm_121, 48 SM, L2 24 MB, DRAM 実测 220 GB/s) に
ポーティングし、帯域最適化を 3 ラウンド入れた状態。インストール済み（`/home/chai/ComfyUI/.venv`）。

実測（bf16, H=24, D=128, non-causal, SDPA 比）:

| L | SA2 | SA3 |
|---:|---:|---:|
| 8 192 | 6.45 ms (1.63x) | 6.18 ms (1.73x) |
| 16 384 | 22.57 ms (1.86x) | 20.29 ms (2.03x) |
| 32 768 | 87.12 ms (1.89x) | 73.39 ms (2.26x) |

内訳（L=16k）: SA2 = attn 20.5 / quantK+mean 1.66 / V quant 1.06 ms。
SA3 = attn 16.0 / delta_s mm 0.94 / quant 1.7 / k.mean 0.44 ms。
アテンションカーネルは SA2 161 TFLOPS（int8+fp8）、SA3 206 TFLOPS（NVFP4）で、GB10 の dense FP4 公称
500 TFLOPS の 41%。

## 1. レビューしてほしいこと（優先度順）

### A. 正しさ・リスク監査（最重要）

1. **`csrc/fused/fused.cu` の `L2ReduceQuantKernel`**（K: mean+int8、V: amax+fp8 を 1 launch に融合）
   - atomic チケットで reduce/quant を自己割当し、quant 側が `ready[bh]` を spin-wait する設計。
     **デッドロックしないという主張が本当に成立するか**（CTA の起動順・occupancy・
     `__launch_bounds__` 1024 スレッド・smem 32 KB での resident CTA 数から）
   - 決定性（fold 順は固定か）、`torch::zeros` の workspace を毎回作るコスト、
     multi-stream / MPS / CUDA graph capture 下での安全性
   - `lag=1` の境界（num_bh が小さい、L が red_tokens より短い、R=1 のとき）
2. **SA3 の `delta_s` 16-bit 化**（`kernel_traits.h` / `mainloop_tma_ws.h` / `api.cu`）
   - TMA ディスクリプタ・smem アライン・`TmaTransactionBytesK` の整合、fp16 と bf16 の両方
   - L が 128 の倍数でない場合のパディング領域の扱い
3. **`mean_out` のカーネル内平均**（`fp4_quantization_4d.cu`）
   - 端数ブロック（0 埋め）の平均が旧経路と一致するか、`head_dim=64` で縮約が正しいか
4. 上記以外で、私が見落としているバグ・未定義動作・数値の罠

### B. 性能の残り（測って裏を取ってほしい）

5. SA3 が FP4 dense ピークの 41% に留まる理由の**分解**（MMA / softmax / P→FP4 変換 / スケール処理）。
   手元の計測で切り分けられるなら数値で。カーネルは演算律速という主張（`delta_s` の読みを半減しても
   16.10→16.07 ms だった）が正しいか。
6. SA2 の 64% との差は何か。4bit 化で相対的にオーバーヘッドが増えるという説明の妥当性。
7. 「次に効く 1 手」を、見積り（ms）付きで 3 つまで。パイプライン段数・kBlockN・warp specialization の
   どれが効きそうか、根拠つきで。

### C. 任意（時間が余ったら）

8. 明らかに安全で小さい改善が 1 つだけあるなら実装して計測（ビルドは `build_ext --inplace`、
   計測は `PYTHONPATH` 経由）。数値が変わらないこと（cos）を必ず確認。

## 2. 環境とコマンド

```bash
# ビルド（SA2: ~1.5 分 / SA3: ~6 分）。共有 venv への install は禁止。
cd /home/chai/worktrees/sa-opus55
MAX_JOBS=6 TORCH_CUDA_ARCH_LIST=12.1 /home/chai/ComfyUI/.venv/bin/python setup.py build_ext --inplace
cd /home/chai/worktrees/sa-opus55/sageattention3_blackwell
MAX_JOBS=6 /home/chai/ComfyUI/.venv/bin/python setup.py build_ext --inplace   # CUTLASS は symlink 済み

# 計測（必ずリポジトリ外から。PYTHONPATH で worktree のビルドを使う）
cd /tmp
PYTHONPATH=/home/chai/worktrees/sa-opus55 \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-opus55/bench/bench_gb10.py --suite sa2 --stages
PYTHONPATH=/home/chai/worktrees/sa-opus55/sageattention3_blackwell \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-opus55/bench/bench_gb10.py --suite sa3
PYTHONPATH=/home/chai/worktrees/sa-opus55/sageattention3_blackwell \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-opus55/bench/bench_sa3_stages.py
PYTHONPATH=/home/chai/worktrees/sa-opus55 \
  /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-opus55/bench/stress_l2_fused.py
```

既存の資料: `GB10_PORTING.md`（設計判断と実測）、`HANDOFF.md`（落とし穴）、`reviews/`（前 2 ラウンドの
レポート）。**前のラウンドの指摘を繰り返さないこと**。

## 3. 制約

- `ncu` は `ERR_NVGPUCTRPERM` で使えない（CUDA event と A/B で測る）
- ベンチは 1 回 ~1 分。実行間のばらつき ±2-3% あるので、比較は同一プロセスか交互 A/B で
- 共有 venv（`/home/chai/ComfyUI/.venv`）への install 禁止
- 入力テンソルを in-place で壊さない（現状の非破壊性を維持）
- 修正した場合は `git commit`（ブランチ `review-opus55`）

## 4. 提出物

`REVIEW.md` に:
1. 重大度つき指摘リスト（Critical / High / Medium / Low、`file:line` 付き）
2. 性能分解の実測（表）
3. 「次に効く 1 手」3 つ（見積り ms 付き）
4. 実施した修正（あれば）と before/after
5. 確認したが問題なかった項目（"no finding" も明記）
