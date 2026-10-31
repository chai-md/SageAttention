# SageAttention 2.2.0 / SageAttention3 — DGX Spark (GB10 / sm_121) 対応メモ

対象: `/home/chai/SageAttention` / ブランチ `gb10-sm121`
環境: DGX Spark GB10 (sm_121, 48 SM, L2 24 MB), torch 2.15.0+cu132, CUDA toolkit 13.0 (nvcc),
Python 3.13.14, venv `/home/chai/ComfyUI/.venv`

---

## 0. 結論（TL;DR）

| 項目 | 状態 |
|---|---|
| SA 2.2.0 ビルド (C++20 + `sm_121a`) | 成功。`_qattn_sm80` / `_qattn_sm89` / `_fused` 生成 |
| SA3 ビルド (C++20 + `sm_121a`, CUTLASS head) | 成功。`fp4attn_cuda` / `fp4quant_cuda` 生成 |
| SA3 の 170 SM ハードコード | デバイス問い合わせに修正（48 SM で最適、170 比 **-9.5%**） |
| GB10 帯域対策 (第 1 弾) | SA3 前処理の全テンソル往復を削除、SA2 の V 量子化を 3 パス → 2 パス（**1.12–1.46x**） |
| GB10 帯域対策 (第 2 弾, Claude Code レビュー) | SA3 `delta_s` を 16-bit のままカーネルへ + `qm` を量子化カーネル内で計算 → **L=16k 21.9 → 19.9 ms** |
| 数値検証 | SA2 の V 量子化は旧経路と **ビット一致**、SA3 は全サイズで cos 不変（0.9818） |

実測（bf16, H=24, D=128, causal=False、SDPA(FP16/BF16) 比）:

| L | SA2 SDPA | SA2 最終 | SA2 速度比 | SA3 初回後 | SA3 最終 | SA3 速度比 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 096 | 0.99 ms | 0.74 ms | 1.34x | 0.67 ms | 0.64 ms | 1.54x |
| 8 192 | 10.5 ms | 6.39 ms | 1.65x | 6.85 ms | 5.89 ms | 1.78x |
| 16 384 | 42.1 ms | 23.2 ms | 1.80x | 21.94 ms | 19.88 ms | 2.05x |
| 32 768 | 167 ms | 87.4 ms | 1.91x | — | — | — |

---

## 1. ビルド

```bash
source /home/chai/ComfyUI/.venv/bin/activate   # or use absolute paths
cd /home/chai/SageAttention
TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=8 \
  uv pip install --python /home/chai/ComfyUI/.venv/bin/python --no-build-isolation .

cd /home/chai/SageAttention/sageattention3_blackwell
MAX_JOBS=8 uv pip install --python /home/chai/ComfyUI/.venv/bin/python --no-build-isolation .
```

* `TORCH_CUDA_ARCH_LIST=12.1` を渡すと setup.py が `-gencode arch=compute_121a,code=sm_121a` を吐く
  （`12.1a` でも可。setup.py 側で `a` を落として数値化する実装）。
* SA2: 約 1.5 分 / SA3: 約 6 分（CUTLASS のテンプレートが重い）。`sageattention3_blackwell/csrc/cutlass`
  は setup.py が自動 clone（約 230 MB）。
* venv に `pip` モジュールが無いので `python -m pip` は不可。`uv pip` を使う。

### C++17 → C++20

`setup.py` / `sageattention3_blackwell/setup.py` の `-std=c++17` をすべて `-std=c++20` に変更済み
（`CXX_FLAGS` と `NVCC_FLAGS`、SA3 は `extra_compile_args["cxx"]` も含む）。
GB10 の CUDA 13.0 では C++20 の方が通りやすい（CUTLASS head も C++17 前提の記述が緩い）。

---

## 2. 変更点

### 2.1 SA3: SM 数ハードコード（170）→ デバイス問い合わせ

`sageattention3_blackwell/sageattn3/blackwell/launch.h`

```cpp
// 旧: dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, 170);
int num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
```

`StaticPersistentTileScheduler` は「SM あたり 1 CTA」の永続カーネルなので、RTX 5090 (170 SM) 前提の
値だと GB10 (48 SM) では 3.5 ウェーブになり、末尾の端数ウェーブで大きく損をする。
`SAGE3_NUM_SM` 環境変数で上書き可能（チューニング用）。

実測（L=16 384、単位 ms）: **48 SM 22.57 / 96 22.67 / 144 23.30 / 170 24.93**。
48 が最良。48 の倍数でも 48 を超えると悪化する（永続 CTA が余るため）。

### 2.2 SA3: mean 減算を FP4 量子化カーネルへ融合

`sageattn3/quantization/fp4_quantization_4d.cu` / `sageattn3/api.py`

旧 `preprocess_qkv` は
`k -= k.mean(...)`（全テンソル read+write）→ `triton_group_mean(q)`（`q - qm` をもう 1 枚生成）
→ `delta_s` の matmul と、**Q/K に対して計 3 パス余分**に走っていた。GB10 は DRAM 実測 ~220 GB/s
しか出ないので、この手の往復がそのまま時間になる。

変更:

* `scaled_fp4_quant` / `scaled_fp4_quant_permute` に任意の `mean` 引数を追加
  （`[B, H, G, D]` 連続テンソル。`G == 1` なら全ブロック共通の mean として使う）。
  カーネルは 16 要素ロード後に mean を引いてから amax/スケールを計算する（数値は等価）。
* Python 側は `q - qm` / `k - km` を実体化しない。`qm` は `q.view(B,H,L//128,128,D).mean(-2)`、
  `km` は `k.mean(-2)` の 1 パスのみ。
* 入力 `k` を in-place で壊さなくなった（旧実装は呼び出し側の `k` を書き換えていた）。

効果（L=16 384）: preprocess 7.14 ms → 3.23 ms。

### 2.3 SA3: `delta_s` の冗長項を削除

`qm @ (k - km)^T = qm @ k^T - (qm·km)` の第 2 項は **キー方向に定数**（全 j で同じ値）なので
softmax で完全にキャンセルする。このカーネルは logsumexp を返さないため、項を落としても出力は
不変（実測: 出力差 max 4.9e-4 / mean 2.4e-9 ≒ 丸め誤差）。`[B,H,G,L]` の read+write 1 往復
（L=16k で約 0.9 ms）を削減。

> 注意: 将来 `return_lse` 相当を実装する場合はこの項を戻す必要がある（lse は行定数シフトを受ける）。

### 2.4 SA2: V 量子化を 2 パス化（FP16 中間バッファ廃止）

`csrc/fused/fused.cu`, `csrc/fused/fused.h`, `csrc/fused/pybind.cpp`, `sageattention/quant.py`

旧 `per_channel_fp8(smooth_v=False)` は 3 パスだった:

1. `transpose_pad_permute`: V(HND) を読み、FP16 の転置・パディング・permute 版を書く（read+write）
2. `scale_fuse_quant`: 転置版を読み amax を計算（read）
3. 同じ転置版を再度読み、FP8 へ量子化して書く（read+write）

→ V 1 要素あたり 4.5 回の DRAM アクセス。GB10 ではこれがそのまま律速（実測 77–196 GB/s 相当）。

追加したカーネル:

* `ChannelAmaxKernel` — 1 スレッド 1 列、列方向に連続アクセス（warp のロードがそのまま coalesced）。
  ブロックごとの最大値を `atomicMax`（非負 float のビット順）で集約。**元の V を 1 回読むだけ**。
* `TransposePadPermuteQuantKernel` — 既存の smem 転置＋permute ロジックはそのままに、
  スケールを掛けて **直接 FP8 で書き出す**（FP16 中間を書かない）。

新 API: `_fused.channel_amax_cuda(v, amax, layout)` / `_fused.transpose_pad_permute_quant_cuda(v, v_fp8, amax, scale_max, layout)`。
`smooth_v=True`（`pv_accum_dtype="fp32"` かつ smooth_v 指定時のみ）は旧経路にフォールバック。

旧経路との比較（20 iter、L は H=24/D=128、単位 µs）:

| L | 旧 (3 パス) | 新 (2 パス) | 倍率 |
|---:|---:|---:|---:|
| 4 096 | 358.6 | 319.9 | 1.12x |
| 8 192 | 786.6 | 642.6 | 1.22x |
| 16 384 | 1720.2 | 1329.6 | 1.29x |
| 32 768 | 3745.2 | 2562.9 | 1.46x |

出力は **ビット一致**（`bench/verify_v_quant.py`: HND/NHD × fp16/bf16 × 端数長 3001 で
スケール・FP8 バイト列とも完全一致）。

### 2.5 変更しなかったもの（判断理由つき）

* **SA2 の sm_121 ディスパッチ**（`core.py`: `qk_quant_gran="per_warp"`, `pv_accum_dtype="fp32+fp16"`）は
  上流のまま。GB10 で全組み合わせを実測したが差は 1–4%（`per_warp+fp32` が僅差で最速、
  `per_thread` 系は 3–8% 遅い）。上流既定を尊重し、勝手に変えない。
* **SA2 の int8 量子化（per_warp_int8 / k.mean）** は実測 224–234 GB/s で DRAM ピーク（~220 GB/s）
  に張り付いており、削れるのは「パス数」だけ。V 以外は既に最小パス。
* **SA3 の FP4 アテンションカーネル**（`kernel_ws.h` 系）は CUTLASS の warp-specialized 実装で、
  stages/cluster のチューニングは別途（下記 5 章）。
* **SA2 アテンションカーネルのタイル形状**は動かさない。`CTA_K=64 → 128` を試したところ、
  コンパイルは通るが **出力が壊れる**（cos 0.31 / NaN）。smem の swizzle と permutation が
  `CTA_K=64, V_SMEM_STRIDE=CTA_K` 前提で組まれているため、タイルだけ変えるのは不可（下記 5 章 P3）。

### 2.6 第 2 弾: 外部レビュー（Claude Code）で入った SA3 の改善

`GB10_PORTING.md` の初版を書いた後に、SA3 を独立レビューさせて 2 点を追加した
（ブランチ `review-claude`、レポートは同ブランチの `REPORT.md`）。

1. **`delta_s` を 16-bit のままカーネルに渡す**（`-1.31 ms @ L=16k`）
   * 旧: bf16 GEMM → `.to(torch.float32)`（**1.31 ms**）→ カーネルが fp32 タイルとして read。
   * 新: `kernel_traits.h` の `ElementDS = ElementOut`、`smem_ds` を 16-bit に、
     `mainloop_tma_ws.h` の `add_delta_s` は 16 B の smem ロード 1 回を
     `cutlass::NumericArrayConverter<float, ElementDS, 8>` で float4 × 2 に変換。
     TMA の `TmaTransactionBytesK` と `api.cu` の dtype チェックも更新。
   * 数値は**ビット単位で同一**（bf16 → fp32 の拡大は情報を増やさない）。cos も全サイズで不変。
2. **`qm`（128 トークン平均）を Q 量子化カーネル内で計算**（`-0.43 ms @ L=16k`）
   * 量子化カーネルは 1 CTA = 128 トークン = ちょうど 1 グループなので、warp shuffle + smem 縮約で
     列平均を出し、`mean_out` として書き戻しつつ `x - mean` を量子化する。
   * 端数ブロックは 0 埋めされているため、平均は旧経路（パディング込みの `q.view().mean()`）と一致。

合計 **L=16k で 21.9 → 19.9 ms（-8.4%）**、SDPA 比 1.87x → **2.05x**。

### 2.7 第 3 弾: 外部レビュー（Claude Code, SA2 担当）で入った改善

`csrc/fused/fused.cu` に **L2 常駐型の「縮約 → 量子化」融合カーネル** `L2ReduceQuantKernel` を追加。

* K: `k.mean()` と K の int8 量子化を 1 launch に統合。V: amax と transpose+fp8 を 1 launch に統合。
  1 ヘッド分（32k×128 で 8.4 MB）は L2（24 MB）に収まるので、量子化パスは DRAM ではなく L2 から再読する。
* CTA は起動順に atomic なチケットを取得し、`reduce` と `quant` の役割を自己選択。quant CTA は
  自ヘッドの reduce 完了フラグを spin-wait する。チケット順序から「待つ側は必ず待たれる側より
  後」が保証されるのでデッドロックしない。`lag=1` で reduce(h+1) と quant(h) を重ねる。
* 出力は旧経路と**ビット一致**（`bench/verify_v_quant.py`、3 レイアウト）。
  48 形状（fp16/bf16 × GQA × 奇数長 1000/3001/4097 × D=64/128）＋ causal ＋ NHD の
  ストレステストで cos ≥ 0.99923、60 回連続呼び出しでも安定（`bench/stress_l2_fused.py`）。
* 効果: 前処理が L=16k で -0.68 ms、L=32k で -0.66 ms。エンドツーエンドでは
  **L=8k で -7.6%（6.74→6.39 ms）**、L=16k で -2〜3%、L=32k では実行ごとのばらつきと同程度。
* `SAGEATTN_L2_FUSED=0` で旧経路（別カーネル 2 発）に戻せる。`SAGEATTN_L2_LAG` /
  `SAGEATTN_L2_RED_TOKENS` でチューニング可。

レビュー本文は `reviews/claude-sa2-l2-fused.md`（SA3 側は `reviews/claude-sa3-delta-s-bf16.md`）。

---

## 3. 数値検証

```bash
cd /tmp   # ← リポジトリ直下で実行すると source tree が site-packages を隠す
python /home/chai/SageAttention/bench/verify_v_quant.py   # SA2 V量子化のビット一致
python /home/chai/SageAttention/bench/smoke_test.py       # fp16/bf16, GQA, NHD, causal, D=64/128
python /home/chai/SageAttention/bench/bench_gb10.py --suite sa2 --stages
python /home/chai/SageAttention/bench/bench_gb10.py --suite sa3
```

smoke test の結果（SDPA fp32 リファレンスとの比較）:

| ケース | max err | cos |
|---|---:|---:|
| SA2 fp16 H24/KV8 L2048 D128 | 0.0144 | 0.99928 |
| SA2 bf16 H12/KV12 L2048 D64 | 0.0100 | 0.99928 |
| SA2 NHD L4096 | 0.0076 | 0.99927 |
| SA2 causal L4096 | 0.1537 | 0.99938 |
| SA3 bf16 L2048 D128 | 0.0639 | 0.98213 |
| SA3 bf16 L2048 D64 | 0.0688 | 0.98098 |

SA3 の cos 0.98 は FP4 量子化由来（E2M1 は 3 有効ビット）。上流実装と同じ水準。

---

## 4. なぜこれ以上速くならないか（ローフライン）

`bench/roofline_gb10.py` の実測:

| 指標 | GB10 実測 | 備考 |
|---|---:|---|
| D2D copy | **218–222 GB/s** | 公称 273 GB/s の約 80% |
| bf16 GEMM (cuBLAS, 8192³) | 77.4 TFLOPS | |
| int8 GEMM (`torch._int_mm`) | 53.3 TOPS | torch 側の実装律速。テンソルコア上限の参考にはならない |
| SM clock | 2 184 MHz (max 3 003) | 電力上限で頭打ち（50 W 時点） |

SA2 のアテンションカーネル単体（L=32 768, H=24, D=128）は 80 ms で、
論理 FLOPs 1.3e13 → **約 161 TOPS**（int8 QK + fp8 PV の混合）。BF16 の 2 倍級のテンソルコアを
使っていることを踏まえると **ピークの 6–7 割**まで来ており、残りは L2 帯域（K/V の再読込）と
占有率（レジスタ 200 本超／CTA 128 スレッドで 2 CTA/SM）で決まっている。
ncu は `ERR_NVGPUCTRPERM` で使用不可（このホストではパフォーマンスカウンタ権限なし）。

→ **帯域側で削れるのは「無駄なパス」だけ**。今回の 2 つの変更（SA3 の mean 融合、SA2 の V 2 パス化）
はまさにそこを突いたもので、効果もそこそこに留まる。これ以上の大物はカーネル構造の変更になる。

---

## 5. 残タスク（優先度順・見積りつき）

### ~~P1: SA3 `delta_s` を 16-bit でカーネルに渡す~~ → 完了（-1.31 ms @ L=16k）

`kernel_traits.h` / `mainloop_tma_ws.h` / `launch.h` / `api.cu` / `api.py` を変更。
詳細は 2.6 節。数値はビット単位で同一。

### ~~P2: SA3 の `qm` を量子化カーネル内で計算~~ → 完了（-0.43 ms @ L=16k）

`scaled_fp4_quant_kernel` に `mean_out` を追加（warp shuffle + smem 縮約）。

### P3: SA2 アテンションカーネルの占有率（要注意・難物）

`CTA_Q=128, CTA_K=64, WARP_Q=32, WARP_K=64`、226 レジスタ/スレッド、2 CTA/SM（8 warps/SM）。
RO アキュムレータが 128 float/thread で上限を作っている。

* **試行済み・失敗**: `CTA_K=128` にすると**出力が壊れる**（cos 0.31 / NaN）。smem の swizzle
  （`V_SMEM_STRIDE = CTA_K`）と permutation が `CTA_K=64` 前提で組まれているため、タイル寸法だけの
  変更は不可。やるなら swizzle/permutation ごと再設計する必要がある（`permuted_smem.cuh`,
  `attn_utils.cuh` の `load_global_to_share` 系）。
* 現実的な次の一手: `pv_accum_dtype="fp32"` 系（RO を 64 bit に落とす）で 3 CTA/SM を狙う、
  あるいは fp8 アキュムレータ版カーネルの追加。いずれも数値検証が必須。

### P4: SA3 アテンションカーネルの stages/cluster チューニング

`launch.h` の `Flash_fwd_kernel_traits<Headdim, 128, 128, 3, 1, per_block, T, O>`（kBlockM, kBlockN,
num_stages, cluster）。GB10 は L2 24 MB なので kBlockN=256 で K/V 再読込を半減できる可能性。
`delta_s` が 16-bit になったぶん smem に余裕ができている（ステージ数を増やす余地）。

### P5: ComfyUI 連携

`/home/chai/ComfyUI/.venv` に入っているので `import sageattention` / `from sageattn3 import
sageattn3_blackwell` はそのまま動く。動画系ノードで SA3 を使う場合は
`num_qo_heads == num_kv_heads`（GQA 非対応）と `head_dim ∈ {64,128}` の制約に注意。

---

## 6. 落とし穴

* **リポジトリ直下で Python を走らせない**。`/home/chai/SageAttention/sageattention` が
  site-packages を隠し、`_fused` 等の .so が見つからず `ImportError` になる。`cd /tmp` などから実行する。
* venv に `pip` が無い → `uv pip install --no-build-isolation .`（build isolation を切らないと
  torch を解決しにいって失敗する）。
* `ncu` は権限エラーで使えない（`ERR_NVGPUCTRPERM`）。計測は CUDA event と A/B 比較で行う。
* toolkit は CUDA 13.0、torch は cu132。混在だが問題なくリンクする（CUDA_HOME=/usr/local/cuda）。
* SA3 の再ビルドは 6 分かかる。C++ だけ触ったときは `.cu` 2 本の再コンパイルで済むが、
  `launch.h` / `kernel_*` を触ると `api.cu` の再コンパイルが走る。
* `sageattention3_blackwell/csrc/cutlass`（230 MB）と `build/`、`*.egg-info` はコミットしない。
* SA2 の `per_channel_fp8` は `smooth_v=False` のときだけ新経路。`smooth_v=True` は旧経路（未検証のため）。
