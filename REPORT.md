# REPORT — SageAttention3 (FP4) GB10 帯域最適化（review-claude）

対象: `sageattention3_blackwell/` のみ。ビルド 1 回（`setup.py build_ext --inplace`、~2 分）、計測はすべて自分で実施。

## 結果サマリ

| shape (bf16, non-causal) | before SA3 ms | after SA3 ms | Δ | cos before | cos after |
|---|---:|---:|---:|---:|---:|
| (1, 8, 4096, 128)   | 0.680  | 0.630  | −7.4 % | 0.98186 | 0.98186 |
| (1, 24, 8192, 128)  | 6.490  | 5.893  | −9.2 % | 0.98180 | 0.98180 |
| (1, 24, 16384, 128) | 21.732 | **19.913** | **−1.82 ms (−8.4 %)** | 0.98185 | 0.98185 |

`bench/bench_gb10.py --suite sa3` の出力。before は同じ worktree の HEAD (`0c8b090`) のビルドで、変更前に測った値。
cos は**全サイズで before と同じ値**: どちらの変更も、カーネルに渡る値をビット単位で変えない（理由は後述）。
SDPA 比は 1.90x → **2.10x**（16k）。

受け入れ条件:
- cos ≥ 0.9810: ✅（4k/8k/16k とも 0.9818）
- API シグネチャ `sageattn3_blackwell(q, k, v, is_causal=..., per_block_mean=...)` は変更なし ✅
- 入力を in-place で書き換えない: ✅（`bench/sa3_extra.py` で bf16/fp16 × per_block_mean on/off × L∈{1000, 4096} について `torch.equal` で確認）
- L=16384 で 21.9 ms 未満: ✅ 19.91 ms

## 1. 帯域監査（H=24, L=16384, D=128, bf16, 1 回の `sageattn3_blackwell()` 呼び出し）

テンソルサイズ: Q/K/V/O bf16 = 各 100.7 MB、`delta_s` [1,24,128,16384] = bf16 100.7 MB / fp32 201.3 MB、
FP4 packed = 各 25.2 MB、fp8 scale = 各 3.1 MB。1 パス 100 MB は 220 GB/s で ≈ 0.46 ms。

| # | ステージ | read | write | 計測 ms (before) | 避けられるか / 案 | 見込み | 状態 |
|---|---|---|---|---:|---|---:|---|
| 1 | `k.mean(dim=-2)` | K 100.7 MB | – | 0.46 | km はスムージング用で、softmax に効かない（`delta_s` から km 項は削除済み）。正確な平均である必要はなく、近似で良い。トークンの間引き（`k[:, :, ::16]`）で推定すれば 1/16 になる | −0.4 | 未実装（量子化品質が実データの分布に依存するので、実モデルでの検証が必要） |
| 2 | pad_128 | – | – | 0 | L%128==0 なら `.contiguous()` は何もしない | 0 | – |
| 3 | `qm`（128 トークンのブロック平均） | Q 100.7 MB | 0.8 MB | 0.43 | **Q 量子化カーネル内で計算する**（各 CTA がちょうど 1 グループ = 128 トークンを担当） | −0.43 | ✅ **実装** |
| 4 | `delta_s = qm @ kᵀ` (bf16 GEMM) | K 100.7 MB | 100.7 MB | 0.90 | K の読みは #1, #7 と重複（K を 3 回読んでいる）。K 量子化カーネルに融合すれば K の読みを 1 回減らせる | −0.45 | 未実装（理由は §4） |
| 5 | `delta_s.to(float32)` | 100.7 MB | 201.3 MB | **1.31**（再計測 1.38） | **カーネルに bf16/fp16 のまま渡す**。GEMM の出力はもともと bf16 に丸められているので、fp32 化は情報を増やさない | −1.31 | ✅ **実装** |
| 6 | quant Q (`x−qm` 融合) | Q 100.7 MB | 28.3 MB | 0.56 | 最低限の 1 パス | 0 | – |
| 7 | quant K permute (`x−km` 融合) | K 100.7 MB | 28.3 MB | 0.57 | 最低限の 1 パス（#4 を融合する先の候補） | 0 | – |
| 8 | quant V transpose | V 100.7 MB | 28.3 MB | 0.57–0.62 | 最低限の 1 パス | 0 | – |
| 9 | FP4 attention | Q/K/V fp4+sf 85 MB（K/V は L2 経由で再読）、`delta_s` 201→**101 MB** | O 100.7 MB + LSE 1.6 MB | 16.10 → 16.07 | 実効 ≈ 205 TFLOPS(FP4)。`delta_s` の read を半分にしても時間が変わらない → DRAM 律速ではない | ~0 | ✅（#5 に付随） |
| 10 | `o[:, :, :QL].contiguous()` | – | – | 0 | QL が 128 の倍数なら何もしない | 0 | – |

まとめ: 16k で避けられる DRAM パスは #3 と #5（計 ~1.7 ms）で、今回はこの 2 つを消した。
残りで避けられるのは K の重複読み（#1 と #4、計 ~0.9 ms）。カーネル（#9）は帯域では律速されていない。

## 2. 実装した変更

### A. `delta_s` を 16-bit のまま扱う（第一候補）
- `sageattn3/api.py`: `delta_s.to(torch.float32)` を削除（`.contiguous()` だけ残す）。
- `blackwell/kernel_traits.h`: `ElementDS = ElementOut`（入力が bf16 なら `bfloat16_t`、fp16 なら `half_t`）。`smem_ds` もこの型に変更。
- `blackwell/mainloop_tma_ws.h`: `TMA_DS` / `Arguments::ptr_ds` / `TmaTransactionBytesK` を `ElementDS` に変更。
  `add_delta_s` は 16 B の smem 読み 1 回（8 要素）を `NumericArrayConverter<float, ElementDS, 8>` で
  float4 × 2 に変換する。各スレッドの列の割り当て（32 列グループごとに `8·(tid%4)` から 8 列）は fp32 版の
  `float4` インデックス `2·(tid%4)+8i, +1` とまったく同じ。
- `blackwell/launch.h`: `delta_s_ptr` のキャストを `Kernel_traits::ElementDS const*` に変更。
- `blackwell/api.cu`: `delta_s` の dtype が出力の dtype と一致するかの `TORCH_CHECK` を追加（fp32 が渡されたときに黙って壊れないように）。
- 数値: GEMM の出力が bf16/fp16 → fp32 → カーネル内 fp32 だったものが、bf16/fp16 → カーネル内 fp32 になるだけなので、
  加算される値は同一。cos が全サイズで一致することでも確認した。
- smem は 1 ステージあたり kBlockN×2 B 減る（今回はステージ数の見直しはしていない）。

### B. `qm` を Q 量子化カーネル内で計算する（代替候補）
- `quantization/fp4_quantization_4d.cu`: `scaled_fp4_quant_kernel` に `mean_out` 引数を追加。
  非 null のとき、warp shuffle（同じ列のスレッド同士、xor `NUM_THREADS_PER_TOKEN..16`）→
  smem `[NUM_WARPS][head_dim]`（D=128 で 16 KB）→ 先頭 head_dim スレッドが合計、の順でブロック内の列和を取る。
  平均を **T（bf16/fp16）に丸めてから** `mean_out` に書き、同じ丸め済みの値を減算する。
  これでホスト側の `delta_s` GEMM が使う qm と、カーネルが減算する qm が一致する。
- ホスト API: `scaled_fp4_quant(..., mean=None, mean_out=None)`（両方の指定は不可、`tensor_layout==1` 限定）。
  permute / trans の各カーネルは既定の `nullptr` のままで、変更はない。
- `sageattn3/api.py`: `per_block_mean=True` の経路は `scale_and_quant_fp4_block_mean(q)` → qm → `qm @ kᵀ` の順にした。
  `per_block_mean=False` は従来どおり `preprocess_qkv` を使う。`preprocess_qkv` は残してある（`bench_sa3_stages.py` が使用）。
- 検証（`bench/sa3_extra.py`）: 融合した qm と `torch.mean` の差は 0（bf16/fp16 とも）。FP4 のビットと fp8 scale は
  `scale_and_quant_fp4(q, qm)` と `torch.equal` で一致。
- コスト: quantQ+qm 0.557 ms ≒ 平均を入力で受け取る quantQ 0.562 ms → ブロック内リダクションの追加コストは誤差の範囲。

### ステージ内訳（after, 16k, `bench_sa3_stages.py` + `bench/sa3_extra.py`）
k.mean 0.44 | quantQ+qm 0.56 | delta_s mm 0.87–0.90 | quantK 0.56 | quantV 0.57 | attn 16.07 | end-to-end 19.2–19.9 ms。
（`bench_sa3_stages.py` の "delta_s->f32" 列は、`preprocess_qkv` が bf16 を返すようになったので、消したキャストの
コスト（1.38 ms）をそのまま測っている。before ではすでに fp32 だったテンソルを変換していたため ~1 µs と表示されていた。）

## 3. ラグのある L について（参考、今回の変更とは無関係）
L=1000（パディングあり）の cos は ≈0.979（bf16/fp16）。venv にインストール済みの元の SA3 でも同程度
（0.9787 / 0.9794）なので、パディング由来の既存の挙動で、今回の変更による劣化ではない。
受け入れ対象の L=4096/8192/16384 はすべて ≥0.9818。

## 4. 却下・保留した案と理由
| 案 | 理由 |
|---|---|
| `delta_s` をさらに fp8 にする | カーネル側で read を半分にしても速くならないことを確認した（#9）。GEMM の write 0.2 ms を削れるだけなのに、数値が変わる |
| `delta_s` を attention カーネル内で計算する（`qm·kᵀ`） | カーネル内の K は FP4 の `(k−km)` しかなく、精度を保つには bf16 の K を m ブロックごとに再読する必要がある（L2 24 MB に収まらない）。カーネルは演算律速なので、追加の MMA も効く |
| `delta_s` GEMM を K 量子化カーネルに融合（K の読みを −1） | 見込み −0.45 ms。ただし 128 グループ × D の内積を CUDA コアでやると ~6 GFLOP（~0.4 ms）で利得が相殺されるおそれがあり、tensor core を使うには mma の再設計が要る。ビルド 3 回の予算では見送り。次の候補 |
| `km` をトークンの間引きで推定（−0.4 ms） | 数学的には任意のシフトが softmax で相殺されるので正しい。ただし量子化品質は実データの分布次第で、乱数ベンチでは検証できない。実モデルでの検証を前提とした次の候補 |
| smem が減った分でパイプラインのステージを増やす | カーネルは帯域律速ではないので効果は薄いと判断。未計測 |

## 5. `git diff gb10-sm121 --stat`
```
 REPORT.md                                          | (this file)
 bench/sa3_extra.py                                 | (new: correctness + stage split)
 sageattention3_blackwell/sageattn3/api.py          | 53 +++++++++++----
 .../sageattn3/blackwell/api.cu                     |  2 +
 .../sageattn3/blackwell/kernel_traits.h            |  7 +-
 .../sageattn3/blackwell/launch.h                   |  2 +-
 .../sageattn3/blackwell/mainloop_tma_ws.h          | 22 ++++---
 .../sageattn3/quantization/fp4_quantization_4d.cu  | 76 ++++++++++++++++++++--
```
（ソース 6 ファイル: 132 insertions(+), 30 deletions(-)。コミット後の正確な stat は `git diff gb10-sm121 --stat` で確認できる）

## 再現
```bash
cd /home/chai/worktrees/sa-claude/sageattention3_blackwell && MAX_JOBS=6 /home/chai/ComfyUI/.venv/bin/python setup.py build_ext --inplace
cd /tmp && PYTHONPATH=/home/chai/worktrees/sa-claude/sageattention3_blackwell /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-claude/bench/bench_gb10.py --suite sa3
cd /tmp && PYTHONPATH=/home/chai/worktrees/sa-claude/sageattention3_blackwell /home/chai/ComfyUI/.venv/bin/python /home/chai/worktrees/sa-claude/bench/sa3_extra.py
```
