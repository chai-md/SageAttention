# REPORT — SageAttention 2.2.0 (SA2) GB10 帯域最適化（review-claude-sa2 / 2 回目）

対象: `csrc/` と `sageattention/` のみ（1 回目の SA3 レポートは `REPORT_SA3.md` に移動）。
ビルドは `setup.py build_ext --inplace` の 3 回だけ。(1) ベースライン、(2) 新カーネル、(3) lag と chunk を
実行時パラメータ化。(2) の 1 回目の試行は `if/else` の波括弧抜けで、数秒でコンパイルエラーになった。
2 回目以降の本ビルドの前には、`fused.cu` だけを `nvcc -c` で通して構文を確認した。

## 結果サマリ

**K の平均と V の amax を、それぞれ量子化カーネルと 1 launch に統合した**。各ヘッドの量子化パスは、
直前の縮約パスが読んだデータを L2 から再読する。
出力（`km`, `k_int8`, `k_scale`, `v_fp8`, `v_scale`）は旧経路と**ビット一致**。

`bench_gb10.py --suite sa2` の結果（bf16, non-causal）。A/B は同じビルドで `SAGEATTN_L2_FUSED=0/1` を
交互に切り替え、2 ラウンドずつ測った。

| shape | before ms (r1 / r2) | after ms (r1 / r2) | Δ (平均) | cos before = after |
|---|---:|---:|---:|---:|
| (1, 8, 4096, 128)   | 0.800 / 0.799 | 0.757 / 0.735 | −0.054 (−6.7 %) | 0.99927 |
| (1, 24, 8192, 128)  | 6.740 / 7.270 | **6.383 / 6.560** | −0.53 (−7.6 %) | 0.99925 |
| (1, 24, 16384, 128) | 23.308 / 23.299 | 22.724 / 22.874 | −0.50 (−2.2 %) | 0.99925 |
| (1, 24, 32768, 128) | 87.939 / 87.582 | 87.636 / 87.045 | −0.42 (−0.5 %) | – |

前処理だけを切り出した計測（`bench/sa2_extra.py`、20 iter、H=24, D=128, bf16）:

| L | Q+K 旧 (mean + per_warp_int8) | Q+K 新 | V 旧 (amax + transpose-quant) | V 新 | 前処理の削減 |
|---:|---:|---:|---:|---:|---:|
| 4096  | 0.424 | 0.383 | 0.307 | 0.239 | −0.11 ms |
| 8192  | 0.880 | 0.714 | 0.650 | 0.466 | −0.35 ms |
| 16384 | 1.715 | 1.475 | 1.369 | 0.926 | −0.68 ms |
| 32768 | 3.446 | 3.084 | 2.596 | 2.294 | −0.66 ms |

受け入れ条件:
- `smoke_test.py` 全通過: ✅（SA2 cos 0.99923–0.99938。値は変更前と同じ）
- `verify_v_quant.py` ビット一致: ✅（3 ケースすべて byte-mismatch 0、scale diff 0）
- **L=8192 が 6.67 ms 未満: ✅**（6.38 / 6.56 ms。before は同じセッションで 6.74 / 7.27 ms）
- L=32768: 87.64 / 87.05 ms で 87.7 ms を下回った。ただし差（−0.4 ms）は実行ごとのばらつき
  （before 側でも 87.58〜87.94 ms）と同程度なので、32k での改善は「前処理で −0.66 ms、全体では
  ノイズに埋もれる」と報告する。

## 1. 帯域監査（H=24, L=32768, D=128, bf16、`sageattn()` 1 回、sm_121 ディスパッチ）

1 テンソル: bf16 = 201 MB、int8 / fp8 = 101 MB。220 GB/s では 201 MB の 1 パスが ≈ 0.91 ms。

| # | ステージ（before） | read | write | 回数 | 回避案 | 見込み | 状態 |
|---|---|---|---|---:|---|---:|---|
| 1 | `k.mean(dim=2)` | K 201 MB | km 6 KB | 1 | K 量子化と同じ launch で縮約し、量子化パスは L2 から再読する | −0.9 ms（理想） | **実装**（#3 と統合） |
| 2 | `quant_per_warp_int8` (Q) | Q 201 MB | 101 MB + scale | 1 | なし（1 回読み・1 回書きで、すでに最小） | 0 | – |
| 3 | `quant_per_block_int8_fuse_sub_mean` (K) | K 201 MB | 101 MB | 1 | #1 と統合 | 上に含む | **実装** |
| 4 | `ChannelAmaxKernel` (V) | V 201 MB | 12 KB | 1 | V 量子化と同じ launch で縮約し、量子化パスは L2 から再読する | −0.9 ms（理想） | **実装**（#5 と統合） |
| 5 | `TransposePadPermuteQuantKernel` (V) | V 201 MB | 101 MB | 1 | #4 と統合 | 上に含む | **実装** |
| 6 | アテンション本体 | q_int8 101 MB + K/V（ヘッドごとに L2 常駐） | O 201 MB | 1 | 演算律速（81 ms ≈ 161 TOPS）。DRAM 量は全体で ~0.5 GB = 2.3 ms 相当で、計算と重なる | 0 | 対象外 |
| 7 | `torch.empty(o)` / `torch.empty` 類 | – | – | – | アロケータのキャッシュから取るだけで、トラフィックはない | 0 | 問題なし |
| 8 | `o[..., :head_dim_og]` | – | – | – | D=64/128 では全体を指す view でコピーなし | 0 | 問題なし |
| 9 | `km_broadcast` + `lse_correction` matmul | Q 201 MB | 小 | `return_lse` 時のみ | GQA の `repeat_interleave` は km（6 KB）だけが対象で無視できる。matmul は Q を 1 パス読む（0.9 ms）。Q 量子化カーネルで Q·km を同時に出せば消せる | −0.9 ms（`return_lse=True` 時のみ） | 却下（後述） |
| 10 | `.contiguous()` | – | – | 0 | この経路には存在しない（入力は last-dim contiguous の assert だけ） | 0 | 問題なし |
| 11 | `F.pad`（D<64 または 64<D<128。例: D=80/96） | Q, K, V 各 1 回 | Q, K, V padded 各 1 回 | 3 | 量子化カーネルが未パディングの入力をマスク付きで読み、出力側だけゼロ埋めする | D=96 @32k で −4〜5 ms | 却下（後述） |
| 12 | 返り値 `o[..., :D]`（パディング経路） | – | – | – | 非連続の view が返る。下流で `.contiguous()` すると 1 パス増える | 下流依存 | 記録のみ |
| 13 | `amax / scale_max`, `torch.zeros(amax)` | 12 KB | 12 KB | 2 | 小さすぎて launch コストだけ | ~5 µs | 新経路で `zeros` は不要になった |

**ブリーフの他の観点**:
- **`per_warp_int8` で Q と K を別カーネルにしている点**: 統合しても減るのは launch 1 回分（数 µs）だけで、
  読み書きのバイト数は変わらない。端数ブロックについては、Q が 512 スレッド CTA × 24576 個、
  K が 1024 スレッド CTA × 12288 個で、48 SM なら 100 波以上になり、テール損失は 1 % 未満。
  `BLKQ=128 / WARPQ=32 / BLKK=64` は L=4k–32k をすべて割り切る。
  アテンション本体（CTA_Q=128、2 CTA/SM で 96 スロット）は、8k/16k/32k × 24 ヘッドで
  1536 / 3072 / 6144 CTA となり、ちょうど 16 / 32 / 64 波で端数が出ない。端数が出るのは
  4k × 8 ヘッド（256 CTA = 2.67 波）だけ。
- **V amax（189 GB/s）の改善**: 単体で速くする代わりに、**DRAM から読むこと自体をやめた**
  （量子化パスと 1 launch にまとめ、量子化側は L2 から再読）。新しい縮約側は 16 B ベクトルロード
  （`__ldg`）、1024 スレッド、smem 縮約、決定的な CTA 間縮約で、atomicMax は使っていない。

## 2. 実装

### `L2ReduceQuantKernel<head_dim, IS_V, T>`（`csrc/fused/fused.cu`）

縮約 CTA と量子化 CTA を 1 つの grid に混ぜて起動する。

- **チケット順序**: 各 CTA は起動時に `atomicAdd` でチケットを取り、チケット番号から役割を決める。
  したがって役割の順序は「CTA が実際に動き出した順」になる。並びは
  `reduce(0) | reduce(1) quant(0) | reduce(2) quant(1) | … | quant(BH−1)`（lag 1）。
  ヘッド h の quant CTA は、ヘッド h の reduce CTA が終わるまで spin する。待たれる側はすべて
  自分より小さいチケットを持ち、しかも自分は誰も待たないので、**デッドロックしない**
  （decoupled look-back と同じ論法）。
- **決定的な縮約**: 各 reduce CTA は自分の partial（列和 / 列 absmax、fp32）を workspace に書き、
  カウンタを `atomicAdd` する。最後に到達した CTA が partial を index 順に畳み、`km`（入力 dtype。
  `torch.mean` と同じく fp32 で和を取り、`×1/L` してから丸める）または `amax`（fp32）を書く。
  そのあと `ready[bh]` を立てる。
- **量子化側**: 数式は `QuantInt8Kernel<…, sub_mean=true>` と
  `TransposePadPermuteQuantKernel<…, pad_zero=true>` のそれぞれと同一。`km` / `amax` は `__ldcg` で
  L2 から読み、他 SM が書いた値を L1 の古いキャッシュから読まないようにしている。
- 46 レジスタ、スピルなし、smem 32 KB。head_dim 64/128、fp16/bf16、HND/NHD、batch > 1、
  L が 64 の倍数でない場合のいずれも動く。

### ホスト / Python

- `_fused.mean_quant_per_block_int8_cuda(k, km_out, k_int8, k_scale, layout, lag, red_tokens)`
- `_fused.amax_transpose_pad_permute_quant_cuda(v, v_fp8, amax, scale_max, layout, lag, red_tokens)`
- `quant.per_warp_int8_smooth_k(q, k, …)` は `(q_int8, q_scale, k_int8, k_scale, km)` を返す。
  `core.sageattn_qk_int8_pv_fp8_cuda` は、`smooth_k` かつ per_warp のときこれを使う。`km` は
  `return_lse` の補正にもそのまま使う。
- `quant.per_channel_fp8(smooth_v=False)` は新カーネルを使う。`verify_v_quant.py` はこの経路を
  そのまま検証している。
- 既存の関数とカーネルは消していない（`bench_gb10 --stages` は旧関数を直接呼ぶ）。
  `SAGEATTN_L2_FUSED=0` を設定すると旧経路に戻る（A/B 用）。

### チューニング（ビルド 3 で実行時パラメータ化し、同じバイナリで掃引）

前処理の削減量（ms、Q+K と V の合計）:

| lag / tokens per reduce CTA | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|
| lag 0 / 512  | −0.21（悪化） | −0.02 | −0.05 | +0.04 |
| lag 0 / 1024 | −0.29 | −0.04 | −0.14 | −0.00 |
| lag 1 / 512  | **+0.12** | +0.33 | +0.32 | +0.08 |
| lag 1 / 1024 | +0.04 | **+0.35** | +0.62 | +0.29 |
| lag 1 / 2048 | −0.10 | +0.28 | **+0.73** | +0.91 / +0.74 |
| lag 1 / 4096 | – | – | +0.49 | +0.81 |

- **lag 0**（ヘッドごとに完全に直列）は、すべてのサイズで旧経路より遅かった。reduce が終わるまで
  quant CTA が SM を占有したまま spin するため。lag 1 では reduce(h+1) と quant(h) が重なるので、
  DRAM が遊ばない。
- **既定値**: lag 1、`red_tokens = clamp(round64(L/8), 512, 2048)`（ヘッドあたり約 8 個の reduce CTA）。
  Python 側（`quant._l2_red_tokens`）で決める。C++ 側の自動選択（`lag<0` なら L2 サイズから選ぶ）は、
  計測で外れと分かったので使っていない。
- **32k で効果が小さい理由**: 1 ヘッドが 8.4 MB あり、lag 1 では約 2.5 ヘッド分（~21 MB）が同時に
  L2 に乗っている必要がある。L2 は 24 MB なので一部がミスする。理想は −1.83 ms
  （201 MB × 2 パス分）だが、実測は −0.66〜−0.74 ms にとどまった。

## 3. 却下・保留した案

| 案 | 理由 |
|---|---|
| lag 0（厳密にヘッド単位） | 上表のとおり全サイズで悪化した |
| reduce CTA 内に tile を保持し、再読を完全になくす | ヘッドの全 CTA（32k で 512 個）が同時に常駐しないと spin がデッドロックする。オンチップ容量（~17 MB）も 1 ヘッド 8.4 MB に対して余裕がない |
| thread block cluster（≤16 CTA）でヘッド単位に同期 | 同時に動くクラスタが 6 ヘッド分（50 MB）になり、L2 に収まらない。効果がチケット方式を上回る見込みがない |
| Q 量子化を同じ launch に入れる | 読み書きのバイト数は変わらず、減るのは launch 1 回だけ |
| `return_lse` 時の `lse_correction` matmul を Q 量子化に統合（#9） | 既定の `sageattn()` は `return_lse=False` で、通常の経路には効かない。カーネル改修に対して恩恵が限定的 |
| パディング経路（D=80/96 など）の `F.pad` 3 回を消す（#11） | 効果は大きい（D=96 @32k で −4〜5 ms 見込み）が、Q / K / V の 3 つの量子化カーネルすべてに「論理 D と物理 D」の 2 つの値を持たせる改修が必要。今回の 3 ビルド予算と計測対象（D=128）の外にある。**次の候補として最優先** |
| K 平均を間引き（`k[:, :, ::16]`）で近似 | 出力が変わる（ビット一致しない）うえ、品質が実データの分布に依存する。今回の方式は厳密に同じ値を出す |
| アテンションの CTA_K=128 | ブリーフで禁止（出力が壊れる） |

## 4. 再現

```bash
cd /tmp
PYTHONPATH=/home/chai/worktrees/sa-claude-sa2 python …/bench/sa2_extra.py          # ビット一致チェック + 前処理 A/B
SAGEATTN_L2_FUSED=0 PYTHONPATH=… python …/bench/bench_gb10.py --suite sa2          # 旧経路
PYTHONPATH=… python …/bench/bench_gb10.py --suite sa2 --stages                     # 新経路
SAGEATTN_L2_LAG=0|1 SAGEATTN_L2_RED_TOKENS=<64 の倍数> …                            # チューニング用
```

注: `bench_gb10 --stages` の `k.mean` と `per_warp_int8(q,k)` は旧関数を直接呼んでいるので、
新経路の効果は `per_channel_fp8(v)` の列（32k: 3.06 → 2.62 ms）と全体時間にしか表れない。
K 側の効果は `sa2_extra.py` で確認する。

## 5. `git diff gb10-sm121 --stat`

（コード commit 後に貼り付け）
