# 引き継ぎ書 — SageAttention GB10 (sm_121) 対応の続き

最終更新: 2026-10-31 / 作業者: Hermes (dgxspark1)
対象: `/home/chai/SageAttention` ブランチ **`gb10-sm121`**（`main` から分岐、未 push）

> このファイルだけ読めば作業を再開できるように書いてある。技術的な詳細・実測値・設計判断は
> 同ディレクトリの **`GB10_PORTING.md`** を参照。まず `git log -1` と `git diff main --stat` を見ること。

---

## 1. いま何ができているか（触る前に知るべきこと）

* SA 2.2.0 と SA3 の両方が **GB10/sm_121 でビルド・実行できる**（C++20、`-gencode arch=compute_121a,code=sm_121a`）。
* 両方 `/home/chai/ComfyUI/.venv` に**インストール済み**（site-packages に .so がある）。
* 実装済みの最適化:
  1. SA3 の 170 SM ハードコード → `at::cuda::getCurrentDeviceProperties()->multiProcessorCount`（GB10=48）
  2. SA3 の mean 減算を FP4 量子化カーネルに融合（Q は 128 トークン毎、K は全系列の平均）
  3. SA3 の `delta_s` から softmax 不変な定数項を削除
  4. SA2 の V 量子化を 3 パス → 2 パス（`channel_amax_cuda` + `transpose_pad_permute_quant_cuda`）
  5. **SA3 `delta_s` を 16-bit のままカーネルへ**（fp32 変換 1.31 ms を削除、TMA/smem も 16-bit 化）
  6. **SA3 `qm` を Q 量子化カーネル内で計算**（`mean_out`、Q の余分な 1 パスを削除）
  7. **SA2 の K/V 前処理を L2 常駐の融合カーネルに**（`L2ReduceQuantKernel`。K: mean+int8、
     V: amax+transpose+fp8 をそれぞれ 1 launch に。旧経路とビット一致、
     `SAGEATTN_L2_FUSED=0` で無効化可）
* 検証済み: SA2 V 量子化は旧経路とビット一致、smoke test 全通過、
  cos 0.9993 (SA2) / 0.9818 (SA3、全サイズで不変)、48 形状のストレスでも cos ≥ 0.99923。
* 現状の速度（bf16, H=24, D=128, vs SDPA）: SA2 1.65x @8k / 1.80x @16k / 1.91x @32k、
  **SA3 2.05x @16k**（19.88 ms）。

---

## 2. 環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GB10 (sm_121, 48 SM, L2 24 MB, DRAM 実測 ~220 GB/s) |
| OS / kernel | Linux 7.0.0-1019-nvidia (aarch64) |
| CUDA toolkit | 13.0 (`/usr/local/cuda/bin/nvcc`) |
| torch | 2.15.0+cu132 |
| Python | 3.13.14 (`/home/chai/ComfyUI/.venv/bin/python`) |
| パッケージ管理 | venv に `pip` モジュールが無い。**`uv pip` を使う** |

---

## 3. コピペで使うコマンド

```bash
# --- 再ビルド (SA2: ~1.5 min) ---
cd /home/chai/SageAttention
TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=8 \
  /home/chai/ComfyUI/.venv/bin/uv pip install --python /home/chai/ComfyUI/.venv/bin/python \
  --no-build-isolation .

# --- 再ビルド (SA3: ~6 min, .cu を触った場合) ---
cd /home/chai/SageAttention/sageattention3_blackwell
MAX_JOBS=8 /home/chai/ComfyUI/.venv/bin/uv pip install --python /home/chai/ComfyUI/.venv/bin/python \
  --no-build-isolation .

# --- 検証・計測（必ずリポジトリ外から実行; 直下だと source tree が .so を隠す） ---
cd /tmp
python /home/chai/SageAttention/bench/verify_v_quant.py     # SA2: V量子化のビット一致
python /home/chai/SageAttention/bench/smoke_test.py        # 全 dtype/layout/causal/GQA
python /home/chai/SageAttention/bench/bench_gb10.py --suite sa2 --stages
python /home/chai/SageAttention/bench/bench_gb10.py --suite sa3
python /home/chai/SageAttention/bench/bench_v_quant_ab.py  # V量子化の A/B
python /home/chai/SageAttention/bench/roofline_gb10.py     # 帯域・GEMM の上限確認
SAGE3_NUM_SM=96 python /home/chai/SageAttention/bench/bench_gb10.py --suite sa3   # SM 数スイープ
```

---

## 4. タスクボード（この順でやる。受け入れ条件つき）

### T1. SA3 `delta_s` を bf16 化する — 見込み -1.7 ms @ L=16 384（約 8%）
* 触る: `sageattention3_blackwell/sageattn3/blackwell/{kernel_traits.h,mainloop_tma_ws.h,api.cu}`、
  `sageattention3_blackwell/sageattn3/api.py`（`.float()` を外す）。
* 現状: `delta_s` は `[B,H,G,L]` の fp32。`.float()` だけで 1.3 ms、カーネルの read で ~0.9 ms。
* 受け入れ: `bench_gb10.py --suite sa3` の `cos` が 0.9818±0.0005 を維持、L=16k で 21.9 ms → 20.2 ms 以下。
* ヒント: `SmemLayoutAtomDS` は `Layout<Shape<kBlockM,kBlockN>, Stride<_0,_1>>`（M 方向ストライド 0 =
  q-block 単位で同一値）。TMA ディスクリプタは `api.cu` の `delta_s` 用のもの。

### T2. SA3 `qm` 計算のカーネル内融合 — 見込み -0.4 ms @ L=16 384
* `scaled_fp4_quant_kernel` は 1 CTA = 128 トークン × head_dim なので、列方向平均を smem 縮約で出せる。
* 注意: `qm` は `delta_s` の matmul にも必要。順序（平均 → delta_s → 量子化）を崩さないこと。
* 受け入れ: 出力 cos 不変、preprocess 3.2 ms → 2.8 ms 以下。

### T3. SA2 アテンションカーネルのタイル形状 A/B — 見込み ±5%
* `csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu` の
  `CTA_Q=128, CTA_K=64, WARP_Q=32, WARP_K=64` を (a) `CTA_K=128`、(b) `CTA_Q=64` で試す。
  1 ファイル再ビルド（~1 分）で `bench_gb10.py --suite sa2` の L=16k/32k を比較。
* 受け入れ: 数値（cos）不変、L=32 768 で 87.7 ms を下回る場合は採用。

### T4. SA3 アテンションカーネルの stages/cluster — 見込み不明（要実測）
* `launch.h` の `Flash_fwd_kernel_traits<Headdim, 128, 128, 3, 1, ...>`（kBlockM, kBlockN, stages, cluster）。
* L2 24 MB なので kBlockN=256 で K/V 再読込が半減する可能性。smem 上限に注意。

### T5. ComfyUI からの利用経路の確認
* 動画系（Wan/Hunyuan）ノードで `sageattn3_blackwell` を差し込む場合、
  **GQA 非対応**（`num_qo_heads == num_kv_heads` 必須）と **head_dim 64/128 のみ**に注意。
* SA2 は GQA 可（smoke test に H24/KV8 のケースあり）。

---

## 5. 触ってはいけない / 注意

* `main` ブランチに戻さない。`git checkout main` すると未コミット変更が散らばる。
* **入力テンソルを in-place で書き換えない**。旧 SA3 は `k -= k.mean(...)` をしていたが、
  現在は非破壊（呼び出し側のテンソルを保つ）。この性質を壊さない。
* `smooth_v=True` の SA2 経路は**旧実装のまま**（新カーネル未対応）。触るなら
  `bench/verify_v_quant.py` にケースを足してビット一致を確認してから。
* `delta_s` の定数項削除は「lse を返さない」前提。`return_lse` 相当を実装するなら項を戻す。
* ベンチは**リポジトリ外**（`/tmp` など）から。直下だと `import sageattention` がソースを掴んで落ちる。
* `ncu` は使えない（`ERR_NVGPUCTRPERM`）。CUDA event の A/B と `--stages` で判断する。
* 顧客データ・非公開情報をこのリポジトリに置かない（公開 OSS のフォーク）。

---

## 6. ファイルマップ（変更点）

| ファイル | 内容 |
|---|---|
| `setup.py` | `-std=c++17` → `-std=c++20`（2 箇所） |
| `sageattention/quant.py` | `per_channel_fp8` に fused 経路（`smooth_v=False` 時） |
| `csrc/fused/fused.cu` | `ChannelAmaxKernel` / `TransposePadPermuteQuantKernel` + host 関数 |
| `csrc/fused/fused.h`, `csrc/fused/pybind.cpp` | 上記の宣言・バインディング |
| `sageattention3_blackwell/setup.py` | `-std=c++17` → `-std=c++20`（3 箇所） |
| `sageattention3_blackwell/sageattn3/blackwell/launch.h` | 170 → デバイス SM 数（`SAGE3_NUM_SM` で上書き可） |
| `sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu` | `mean` 引数の追加（quant/permute 両方）、`resolve_mean` ヘルパ |
| `sageattention3_blackwell/sageattn3/api.py` | `preprocess_qkv` の書き換え、`scale_and_quant_fp4*` に mean を渡す |
| `bench/*.py` | 計測・検証スクリプト（下記） |
| `GB10_PORTING.md` | 技術レポート（実測値・設計判断・残タスク） |

### bench/ の中身

| スクリプト | 用途 |
|---|---|
| `bench_gb10.py` | 総合ベンチ（`--suite sa2/sa3`, `--stages`, `--shape`, `--dtype`, `--causal`） |
| `verify_v_quant.py` | V 量子化の新旧ビット一致検証 |
| `smoke_test.py` | fp16/bf16・GQA・NHD・causal・D=64/128 の一括チェック |
| `bench_v_quant_ab.py` | V 量子化だけの A/B（20 iter） |
| `bench_sa3_stages.py` | SA3 のステージ別内訳 |
| `bench_variants.py` / `bench_backends.py` | SA2 の量子化粒度・PV 累積 dtype の比較 |
| `roofline_gb10.py` | DRAM 帯域 / bf16 GEMM / int8 GEMM の上限測定 |

---

## 7. 現時点の実測値（ベースラインとして使う）

bf16, H=24, D=128, non-causal, SDPA(fp16/bf16) 比:

| L | SA2 | SA3 |
|---:|---:|---:|
| 4 096 | 0.80 ms (1.24x) | 0.67 ms (1.47x) |
| 8 192 | 6.67 ms (1.56x) | 6.85 ms (1.55x) |
| 16 384 | 23.41 ms (1.80x) | 21.94 ms (1.87x) |
| 32 768 | 87.74 ms (1.89x) | — |

SA3 内訳（L=16 384, 合計 21.9 ms）: attn kernel 16.1 / delta_s matmul 0.92 / delta_s→fp32 1.31 /
k.mean 0.46 / qm 0.43 / quant Q+K+V 1.69。
→ 次の 1 手は T1（delta_s bf16）が最も効く。

---

## 8. エージェントに渡すときのプロンプト例

```
/home/chai/SageAttention のブランチ gb10-sm121 で作業を続けてください。
まず HANDOFF.md と GB10_PORTING.md を読んでください。
タスクは HANDOFF.md の T1（SA3 の delta_s を bf16 化）です。
変更後は必ず:
  cd /tmp && python /home/chai/SageAttention/bench/smoke_test.py
  cd /tmp && python /home/chai/SageAttention/bench/bench_gb10.py --suite sa3
を実行し、cos が 0.9818±0.0005 を保ち、L=16384 が 20.2 ms 以下になったことを報告してください。
リポジトリ直下で python を実行しないこと（source tree が .so を隠します）。
ビルドは sageattention3_blackwell で uv pip install --no-build-isolation .（約 6 分）。
```
