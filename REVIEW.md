# REVIEW — GB10 (sm_121) 対応の第三者監査（Claude Opus 5.5 / 監査モード）

対象: `review-opus55`（`gb10-sm121` = `f40986f` から分岐）。ビルドは `setup.py build_ext --inplace` のみで、
共有 venv への install はしていない。計測はすべて `/tmp` から `PYTHONPATH` 経由で行った。

## 0. 要約

- **L2ReduceQuantKernel（A-1）は健全**。デッドロックしないという主張は、起動順にも occupancy にもよらず成立する。
  実機でも、47/48 SM を常駐 attention grid で塞ぎ、**残り 1 SM・常駐 1 CTA だけ**で走らせて完走した
  （出力はビット一致）。旧経路とのビット一致は 1536 構成すべてで確認した。multi-stream、CUDA graph、決定性（50 回）
  も問題なかった。
- **SA3 に upstream 由来の重大な正しさのバグが 2 つあった**（GB10 パッチ起因ではない。どちらも既存テストの範囲外だった）。
  1. **キーマスクが K の 32 トークン置換を無視していた**。K 長が 32 の倍数でないと、実キーが落ちてパディングキーが残る。
     causal では対角マスクも誤る。
  2. **causal の masking step 0 の PV ブロックを捨てていた**。
  修正後、causal の cos は **0.71–0.79 → 0.983–0.985**、L=1000 は 0.979 → 0.982、キー 8 個では 0.47 → 0.989 になった。
  なお、前回レビューの「L=1000 で cos が 0.979 なのはパディング由来の既存挙動」という説明は、(1) を指していた。
- **性能**: GB10 の FP4 `mma.sync` の実測上限は **430 TFLOPS**（公称 500 ではない）。SA3 は修正前 206 TFLOPS で、
  この上限の **48 %**。アブレーションの結果、カーネルは**演算律速でも帯域律速でもなく、ワープ内の直列チェーンの
  レイテンシで律速**されていた（各部分を外したときの短縮量の合計が全体時間を大きく超える）。
- **実施した修正（3 件、いずれも小さい）**: V の待ちと smem→reg を softmax の前へ移した（D=128 のみ）。
  これで attention カーネルは **16.0 → 13.4 ms（−16 %、16k）**、end-to-end は **19.91 → 17.62 ms**
  （SDPA 比 2.06x → 2.31x）。既存経路（D=128、L%128==0、non-causal）の出力はビット一致。
  ほかに上記バグ 2 件の修正と、未初期化アキュムレータの `clear()`。

---

## 1. 指摘一覧（重大度順）

行番号は特記がない限り修正**後**のもの。修正前の位置は (旧 L…) で示す。

### Critical

**C-1. SA3 キーマスクが K のトークン置換を無視している（upstream バグ。修正済み）**
`sageattention3_blackwell/sageattn3/blackwell/mainloop_tma_ws.h:746-760, 882`（旧 L747, L749, L847）

- K 量子化器 `scale_and_quant_fp4_permute`（`fp4_quantization_4d.cu:196-202`）は、32 トークンごとに
  `[0,1,8,9,16,17,24,25,2,3,…]` の順でトークンを並べ替えて格納する。そのため S タイルの列 c に対応するのはトークン
  `perm(c)` で、c そのものではない。`add_delta_s`（`:692`）は正しくトークン順で読んでいるが、マスク側は列番号 c を
  そのまま閾値と比較していた。
- 実害:
  - non-causal: `unpadded_seqlen_k % 32 != 0` の最後の 32 グループで、**実キーがマスクされ、パディングキーが残る**。
    パディングキーのスコアは `(q−qm)·(−km)`、つまり「キー = 0 ベクトル」相当になる。乱数入力では軽微
    （L=1000: cos 0.979）だが、K に大きなチャネル平均がある実モデル（SageAttention が smoothing する理由そのもの）
    では、このダミーキーが重みを奪いうる。
  - causal: 対角タイルで、行 r が見えるトークンが `{perm(0..r)}` になる。例えば行 3 は {0,1,8,9} を見る。
  - one-hot V で P を直接観測した結果: キー 8 個のとき SA3 の P の行和は **0.12**（6/8 の実キーが落ちている）。
- 修正: `k_token(c)` に写してから比較する。実行時に要素ごと計算すると spill が 24 → 120 B に増え、0.4 ms 遅くなった。
  そこで、thread-0 のコンパイル時列番号とスレッドごとのオフセット（bit 1-2 = `2·(lane%4)`）に分解して定数畳み込み
  させ、spill を元の量に戻した。両実装の出力はビット一致で確認した。
- 検証: `bench/sa3_mask_test.py`（新規、44 ケース）は、修正前のビルドで worst 0.46（FAIL）、修正後で worst 0.981（OK）。

**C-2. causal の masking step 0 の PV 寄与を破棄している（upstream バグ。修正済み）**
`mainloop_tma_ws.h:912`（旧 L867 `if (masking_step > 0) { rescale_o(...) }`）

- masking ループの PV は、新しい `tOrO` に累積してから `rescale_o` で `tOrO_store` に足し込む。ところが kBlockM=kBlockN
  では masking step は 0 番の 1 回しかなく、その寄与が**常に捨てられていた**。softmax の `row_sum` にはこのブロックが
  入っているので、出力は全体に縮小し、かつ誤った重みになる。FA3 版（O を先に rescale してから同じアキュムレータに
  gemm する）を 2 段累積に書き換えた際の取り残しと見られる（`git blame`: e08749b, upstream）。
- C-1 と C-2 を合わせた修正後、causal の cos は全 24 構成で 0.983–0.985（修正前 0.71–0.79）。
  causal の速度は 7.85 vs 7.87 ms で変わらない。

### High

**H-1. テストの穴: SA3 の causal と、K 長が 32 の倍数でないケースが一度もテストされていなかった**
`bench/smoke_test.py:57-64`（SA3 は non-causal、L=2048 だけ）、`bench/sa3_extra.py`

C-1・C-2 が残っていたのはこのため。`bench/sa3_mask_test.py` を追加した。smoke_test への組み込みを推奨する。

### Medium

**M-1. 未定義値（undef）への累積に依存していた（`tOrO` は修正済み、`AbsMaxP` は記録のみ）**
- `kernel_ws.h:180` の `partition_fragment_C` と、`mainloop_tma_ws.h` の `make_fragment_like(tOrO_store)` は、どちらも
  clear せずに PV MMA の C オペランドとして使われていた。SASS を見ると ptxas は undef を `RZ`（ゼロ）として
  `OMMA … RZ` を出しており、たまたま正しく動いているだけだった。`clear()` を追加した（`kernel_ws.h:181`、
  `mainloop_tma_ws.h:896, 943`）。生成される結果はビット一致（48/48 中、該当 24 構成）で、性能も変わらない。
- `AbsMaxP`（`mainloop_tma_ws.h:719`）は first tile で `fmaxf(AbsMaxP, …)` として**未初期化のまま読まれる**
  （`softmax_fused.h:60,63`）。0 初期化と結果がビット一致することを確認した（ptxas は undef = 0 として扱っている）。
  したがって明示的に `fill(AbsMaxP, 0.f)` を入れても挙動は変わらない。今回はビルド予算の都合で入れていないが、
  次の機会に入れることを推奨する。`-inf` で初期化すると、全列がマスクされた 16 列グループで 0/0 になるため、
  ガードが必要になる（ablation で試作し、精度差がほぼないことを確認済み）。

**M-2. `delta_s` の stride・shape を検証していない**
`api.cu:239, 267`（`CHECK_SHAPE` がコメントアウトされている）、`mainloop_tma_ws.h:224-226`

TMA レイアウトは shape からコンパクトなストライドを作るだけで、`ds_*_stride` は使われない。そのため、
非連続な `delta_s`（例えば view）を直接 `fp4attn_cuda.fwd` に渡すと、黙って誤った結果になる。Python の 2 経路は
どちらも連続なテンソルを渡すので、到達するのは直接呼び出した場合だけ。`CHECK_CONTIGUOUS(delta_s)` と shape の
検証を入れることを推奨する。

**M-3. masking ループの `col_limit_causal(…, n_block - 1)`（修正済み）** `mainloop_tma_ws.h:882`

ソフトウェアパイプライン化された FA3 の名残で、このループでは `n_block` が正しい。違いが出るのは
`seqlen_k − seqlen_q` が 128 の倍数でないときだけで、Python 経路は両方を 128 にパディングするため到達しない
（到達する構成では修正前後でビット一致）。

### Low

- **L-1. L2 カーネルのオフセットが uint32**（`csrc/fused/fused.cu:415` ほか）。`b*stride_bz_input` などは、要素数が
  2^32（bf16 で 8.6 GB）以上のテンソルで溢れる。SA2 の他のカーネルと同じ慣習だが、GB10 はユニファイドメモリが
  128 GB あるので、到達は理論上可能。
- **L-2. `num_tokens == 0`** のとき grid 0 で launch エラーになる（`fused.cu:1621-1631` に TORCH_CHECK がない）。
- **L-3. V の amax が `fmaxf` で NaN を無視する**（`fused.cu:439`）。旧 `ChannelAmaxKernel` も同じ挙動で、回帰ではない。
- **L-4. ベンチが worktree ではなくメインチェックアウトの `bench_gb10` を import している**
  （`bench/stress_l2_fused.py:13`、`bench_sa3_stages.py:12`、`bench_backends.py:7`）。
- **L-5. `setup.py build_ext` はヘッダ依存を追跡しない**（distutils、ninja なし）。`.h` だけを編集すると再ビルドされず、
  古い `.so` で計測してしまう。今回も 1 回それで空振りした。`touch api.cu` などで回避できる。HANDOFF の落とし穴に
  追記する価値がある。
- **L-6. D=64 は今回の修正で +2.3 %（10.01 → 10.24 ms @16k）**。V の前倒しは D=64 では逆効果（+5 %）だったので、
  `kEarlyV = kHeadDim == 128` で除外した。残る +2 % はマスク修正と clear によるもので、正しさの修正の対価として
  許容した。

---

## 2. A. 正しさ監査 — 確認して問題がなかった項目（no finding）

### A-1. `L2ReduceQuantKernel`（`csrc/fused/fused.cu:343-591, 1600-1717`）

| 観点 | 結論 | 根拠 |
|---|---|---|
| デッドロックしないか | **no finding** | チケットは、すでに常駐している CTA が `atomicAdd`（`:410`）で取得する。quant(h) が待つ reduce(h) の R 個の CTA は、すべてより小さいチケットを持つので、**すでに常駐済みか終了済み**であり、しかも自分は何も待たない。そのため、ハードウェアの起動順、occupancy（1024 thr・32 KB smem・46 reg で **1 CTA/SM**）、MPS の SM 制限、他カーネルとの共存のいずれにも依存しない。チケットは grid と全単射（lag 1: `R + (BH−1)(R+Qn) + Qn = BH(R+Qn)`）。**実機確認**: `SAGE3_NUM_SM=47` で常駐 attention を走らせ、空き 1 SM の状態で L2 カーネルを起動した。23.3 ms で完走し（attention は 124 ms まで継続）、ビット一致した（`bench/opus55/occ_test.py`）。 |
| メモリ順序 | no finding | partial は書いてから `__threadfence` → `atomicAdd`。last CTA は `__threadfence` → `__ldcg`。km/amax の書き込み → `__threadfence` → `__syncthreads` → `atomicExch(ready)`（`:486`）。待つ側は volatile spin（`:495`）→ `__threadfence` → `__syncthreads` → `__ldcg`。標準的な threadFenceReduction のパターンで、L1 の古いデータを読むことはない。 |
| 決定性 | no finding | 各 CTA 内の行和は固定順。CTA 間は「最後の CTA が index 順に fold する」ので、どの CTA が最後になっても同じ結果になる。km/amax は 50 回ともビット一致。 |
| `torch::zeros` の workspace | no finding（コスト ~2–3 µs） | memset 1 回と、キャッシングアロケータからの取得だけ。16k の 1.45 ms に対して誤差の範囲。ストリームはアロケータのストリーム意味論に従うので安全。 |
| multi-stream | no finding | 4 ストリーム × 20 回の同時起動ですべてビット一致（workspace は呼び出しごと）。 |
| CUDA graph capture | no finding | zeros はグラフ内の memset ノードとして capture されるので、replay のたびに再ゼロ化される。replay 10 回でビット一致。 |
| lag 境界（num_bh=1, R=1, L<red_tokens, L∈{1,63,64,65,511,1000,2048,4097}, red_tokens∈{64,512,2048}, lag 0/1, D 64/128, fp16/bf16, HND/NHD, B 1/2） | no finding | 1536 構成すべてで旧 2 カーネル経路（`k.mean` + `quant_per_block_int8_fuse_sub_mean`、`channel_amax` + `transpose_pad_permute_quant`）と **km・k_int8・k_scale・amax・v_fp8 がすべてビット一致**（`bench/opus55/l2_edge.py`）。 |

### A-2. SA3 の `delta_s` 16-bit 化

| 観点 | 結論 |
|---|---|
| `TmaTransactionBytesK`（`mainloop_tma_ws.h:147-150`） | no finding。`cosize(take<0,2>(SmemLayoutDS)) × 16 bit` = kBlockN×2 B で、TMA box（stride-0 の M モード、N=128）と一致する。 |
| smem アライン | no finding。`smem_ds` は `alignas(1024)`（`kernel_traits.h:57`）、ステージ間隔 256 B（TMA の 128 B 要件を満たす）、inner box 256 B（16 の倍数）。 |
| fp16 / bf16 | no finding。`ElementDS = ElementOut`。`api.cu:240` の dtype チェックがあり、両 dtype でビット一致を確認した。 |
| L%128 != 0 のパディング領域 | no finding（delta_s 側）。パディングキーの delta_s は `qm·0 = 0` で、マスクされる。ただしマスク自体が C-1 で壊れていた（修正済み）。パディング Q 行の出力は捨てられる。 |

### A-3. `mean_out` のカーネル内平均（`fp4_quantization_4d.cu:233-279`）

no finding。`acc / 128` はゼロ埋めトークンも分母に含めるが（`:266`）、旧経路も「pad してから `view().mean()`」なので
同じ定義になる。いずれにせよ delta_s が同じ qm を使うので、数学的には相殺される。D=64 は
`NUM_THREADS_PER_TOKEN=4` → shuffle の xor 4,8,16 → 16 warp、D=128 は 8 → xor 8,16 → 32 warp で、どちらの縮約も正しい。
`sa3_extra.py` で `torch.mean` との差 0、fp4 ビットと sf が一致することを確認した。

---

## 3. B. 性能分解（実測）

### 3.1 GB10 の実際の演算上限（`bench/opus55/mb.cu`, `mb2.cu`、raw PTX `mma.sync`）

| 命令 | 実測ピーク | 備考 |
|---|---:|---|
| FP4 `mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X m16n8k64` (f32 acc) | **430 TFLOPS** | 8 warps/SM 以上で飽和する。公称 500 は ~2.5 GHz 想定で、持続クロックは ~2.17 GHz。 |
| int8 `m16n8k32` s32 acc | 215 TOPS | |
| fp8 e4m3 `m16n8k32` f32 / f16 acc | 215 / 215 TFLOPS | GB10 では f16 acc にしても速くならない。 |
| `ex2.approx`（MUFU） | 14.5 /clk/SM（報告クロック基準。実クロックでは 16/clk） | |
| `cvt.rn.satfinite.e2m1x2.f32` / FFMA | 113 /clk/SM | P→FP4 の変換はフルレートで安い。 |

→ 修正前の SA3 は 206/430 = **48 %**、修正後は 245/430 = **57 %**。SA2 は 161/215 = **75 %**。

### 3.2 SA3 アテンションカーネルのアブレーション（bf16, H=24, L=16384, D=128、カーネル単体、各 2 回）

`bench/opus55/make_variant.py` が `/tmp` にカーネルのコピーを作り、`SAGE3_ABL` ビットで各段を外してビルドする
（bf16/D128/non-causal だけをインスタンス化し、45 s/variant）。外した結果が使われない MMA は ptxas が DCE で消すため、
その点を「実際に消えたもの」の列に明記している。

| variant | 外したもの（実際に消えたもの） | ms | 差 |
|---|---|---:|---:|
| baseline | – | 16.08 / 16.24 | – |
| no QK MMA | QK OMMA 32 個 + K ldmatrix | 12.67 | −3.4 |
| no softmax | max/exp2/rowsum/除算 | 10.00 / 10.17 | **−6.0** |
| no P→FP4 quantize | cvt + SF 交換 | 12.14 | −3.9 |
| no PV MMA | PV OMMA + V ldmatrix + quantize（DCE） | 9.40 / 9.23 | −6.8 |
| no `add_delta_s` | アキュムレータ初期化を undef に | 13.22 / 13.54 | **−2.9** |
| delta_s → zero-init | 同上（`clear`） | 13.15 / 13.60 | −2.9 |
| delta_s → 定数 init | LDS + 変換だけを外す | 14.33 / 13.85 | −2.0 |
| delta_s を MMA 後に加算 | init → post-add | 16.83 / 16.61 | +0.6（悪化） |
| no K smem→reg | K ldmatrix + SFK | 16.26 / 16.20 | **±0** |
| no V smem→reg | V ldmatrix + SFV | 12.74 / 12.73 | **−3.4** |
| MMA only（PV だけ残る） | softmax・quant・ds・ldsm・rescale | 4.16 / 4.22 | PV 半分の FLOPs で 396 TFLOPS ≈ 上限 |
| kStages 3 → 2 | – | 16.21 / 16.46 | ±0 → **producer/TMA は律速ではない** |
| WG1 を 1k/2k/4k cycle 遅延（障壁なし） | – | 16.0–16.3 | ±0（位相はすぐ再同期する） |
| **V の wait + ldmatrix を softmax の前へ** | 順序変更のみ | **13.48 / 13.57** | **−2.5（採用）** |

**読み方**:
- MMA だけなら 16k は 3.30 TFLOP / 430 = **7.7 ms** で済む。実際は 16.0 ms だった。
- 各段を外したときの短縮量の合計（3.4+6.0+3.9+2.9+3.4+…）が 16 ms を大きく超えるので、各段は**同じ直列チェーン上に
  並んでいる**。スループット律速なら、短縮量は加算的になるはずである。
- 外したときの短縮が、命令数に比べて不釣り合いに大きいものがある。delta_s（LDS.128 4 本と変換 32 命令/タイルで −2.9 ms）、
  V ldmatrix（16 LDSM/タイルで −3.4 ms、一方で同じ量の K ldmatrix は ±0）。これらは**レイテンシがクリティカルパスに
  露出している**ことを示す。K の ldmatrix は QK の OMMA と交互に発行されて隠れるが、V の 1 本目は softmax と PV の
  間に挟まっていて隠れていなかった。採用した修正はこれを softmax の前に移すもので、−2.5 ms になった。

### 3.3 メインループの命令構成（SASS、1 タイル・1 warp あたり 850 命令）

OMMA 64 / FFMA 140 / MUFU.EX2 82 / FMNMX 82 / FADD 72 / FMUL 66 / F2FP 36 / LDSM 32 / LDS 20 / SHFL 12 …

1 SMSP（2 consumer warp）・1 タイルあたりで見ると:
- **テンソルパイプ**: 2048 cycle（OMMA 1 本 = 16 cycle × 64 × 2 warp）
- **MUFU**: 82 × 8 × 2 = **1312 cycle**
- **FP32**: ~720 cycle
- **発行スロット**: 1700

実測は 16.0 ms ÷ 8192 タイル/SM × 2.17 GHz ≈ **4250 cycle/タイル**。これは「MMA 2048 + MUFU 1312 + その他 ~900」を
**ほぼ直列に足した値**に一致する。つまり、各 SMSP 上の 2 warp（WG1 と WG2 から 1 つずつ）が**同位相で**
「MMA → softmax → MMA」を進めていて、テンソルコアと MUFU が重なっていない。

### 3.4 SA2（64 %）との差（Q6）

ブリーフの「4bit 化でオーバーヘッドが相対的に増える」という説明は、**定量的に正しい**。タイルの形が同じなら、
softmax（MUFU ~1312 cycle）とその他（~900 cycle）の絶対量は SA2 でも SA3 でも同じである。一方、MMA は
int8/fp8 の上限 215 に対して FP4 は 430 なので、FP4 では半分の 2048 cycle（SA2 は 4096）で終わる。
直列モデルで予測すると:
- SA2: 4096 / (4096+1312+~200) ≈ **73 %**（実測 75 %）
- SA3: 2048 / (2048+1312+~900) ≈ **48 %**（実測 48 %）

なお、ブリーフの「SA2 64 %」は公称値を分母にしたもので、実測上限 215 に対しては 75 %。

### 3.5 「演算律速」という主張（Q5）について

DRAM 律速ではない、という点は正しい（`delta_s` の読みを半減しても変わらず、ステージを 2 に減らしても変わらない）。
ただし、**テンソルコアのスループットで律速されてもいない**（上限の 48 %）。正確には、**ワープ内の依存チェーンの
レイテンシで律速されている**（MMA → softmax の shuffle と MUFU → 量子化 → MMA が、2 warp/SMSP では隠しきれない）。

---

## 4. 次に効く手（見積りつき、16k・D=128・bf16 の attention カーネル 13.4 ms 基準）

1. **consumer warpgroup の ping-pong（名前付き障壁で WG1 の MMA と WG2 の softmax を交互にする）、または FA3 式に
   QK(n+1) を softmax(n) の前に発行するワープ内ソフトウェアパイプライン** — 見積り **−3 〜 −4 ms（→ 9.5–10.5 ms）**。
   - 根拠: §3.3 の直列モデルでは、理想的に重なれば 1 タイル ≈ max(MMA 2048, warp チェーン ~2300) ≈ 2400–2900 cycle
     （現状 ~3550 cycle）。
   - 障壁なしでずらすだけでは位相がすぐ戻る（stagger 実験で ±0）ので、障壁で**強制**する必要がある。
   - 注意: ソフトウェアパイプライン版は S アキュムレータの 2 重化で +64 reg が必要になる。ptxas は 168 reg 上限で
     コンパイルしており、すでに 24 B spill しているので、ping-pong の方がレジスタ面で安全。
   - kBlockN の拡大と段数の増加は**効かない**。段数 2→3 で ±0（§3.2）。kBlockN=256 は smem（現状 ~96 KB / 99 KB）と
     レジスタの両方で収まらない。
2. **`delta_s` の初期化をクリティカルパスから外す** — 見積り **−1 〜 −2 ms**。
   - init を消すと −2.9 ms、定数 init でも −2.0 ms なので、LDS から変換に至る経路だけで ~2 ms かかっている。
   - 前タイルの PV フェーズ中に次ステージの delta_s（1 thread あたり 32 値 = bf16x2 で 16 reg）を先読みしておき、
     QK の開始時は MOV だけで済むようにする。
   - MMA の後に足す方式（post-add）は +0.6 ms で逆効果だった。
3. **前処理の削減（end-to-end 側）** — 見積り **−0.5 〜 −0.9 ms**。16k の前処理は k.mean 0.46 + delta_s GEMM 0.90 +
   量子化 3 本 1.63 = 3.0 ms（全体 16.6 ms の 18 %）。
   - delta_s GEMM の K 読みを K 量子化器に融合すれば −0.45 ms。
   - km をトークンの間引きで推定すれば −0.4 ms（前回レビューの候補。実データでの検証が前提）。
   - 1・2 を実施すると、この割合は 25 % 近くまで上がる。

---

## 5. 実施した修正と before / after

コミット: `review-opus55`。変更ファイルは `sageattention3_blackwell/sageattn3/blackwell/{mainloop_tma_ws.h, kernel_ws.h}`、
新規テスト `bench/sa3_mask_test.py`、調査用スクリプト `bench/opus55/`、そしてこの `REVIEW.md`。

| # | 変更 | 数値への影響 |
|---|---|---|
| 1 | D=128 では V の `consumer_wait` と 1 本目の smem→reg を softmax の前へ（`kEarlyV`, `:828`） | **出力ビット一致**（パイプライン操作の順序「K wait → K release → V wait → V release」は不変で、独立なロードの位置が変わるだけ） |
| 2 | マスクをトークン順に（C-1）、masking ループの `n_block`（M-3） | causal、L%32≠0、少数キーで出力が変わる（正しくなる）。L%128==0 の non-causal はビット一致 |
| 3 | masking step 0 の PV 寄与を足す（C-2） | causal のみ変わる（正しくなる） |
| 4 | `clear(tOrO)` 3 箇所（M-1） | ビット一致（ptxas は元から RZ を使っていた） |

**ビット一致・精度**（`bench/opus55/sa3_ref.py`。修正前のビルドの出力を保存して比較。bf16/fp16 × D64/128 ×
L{1000,4096,8192} × causal × per_block_mean = 48 構成）:
- L%128==0 の non-causal 16 構成はすべてビット一致。
- causal 24 構成: cos 0.71–0.79 → **0.983–0.985**。
- L=1000 の non-causal 8 構成: cos 0.978–0.980 → **0.982**。
- 入力テンソルは非破壊のまま（`sa3_extra.py`、`sa3_ref.py` の assert）。
- `smoke_test.py` OK、`stress_l2_fused.py` OK（SA2 側は無変更）。

**速度**（idle を確認したうえで 7 回の最小値を 3 ラウンド。`bench/opus55/time_cfg.py`。`orig` は HEAD を /tmp で
フルビルドしたもの。カーネル単体、H=24）:

| 構成 | before ms | after ms | Δ |
|---|---:|---:|---:|
| bf16 D128 L8192 | 3.99 | **3.36** | −16 % |
| bf16 D128 L16384 | 15.96 | **13.43** | −16 % |
| bf16 D128 L32768 | 63.2 | **53.7** | −15 % |
| fp16 D128 L16384 | 14.2 | **13.3** | −6 % |
| bf16 D128 L16384 causal | 7.87 | 7.85 | ±0（正しくなった） |
| bf16 D64 L16384 | 10.01 | 10.24 | +2.3 %（L-6） |

**end-to-end**（`bench_gb10.py --suite sa3`。before は同じ worktree の修正前ビルドで、ComfyUI が GPU を使い始める前に
計測）:

| shape | before SA3 ms | after SA3 ms | SDPA 比 before → after | cos |
|---|---:|---:|---:|---:|
| (1, 8, 4096, 128) | 0.661 | 0.574 | 1.91x → 1.72x※ | 0.98186 = |
| (1, 24, 8192, 128) | 5.909 | **5.135** | 1.78x → 2.03x | 0.98180 = |
| (1, 24, 16384, 128) | 19.911 | **17.617** | 2.06x → 2.31x | 0.98185 = |
| (1, 24, 32768, 128) | 73.39（ブリーフ値） | **63.59** | 2.26x → 2.62x | – |

※ 4k の SDPA は計測ごとのばらつきが大きい（1.27 → 0.99 ms）。SA3 側は −13 %。

ステージ内訳（16k、after）: k.mean 0.46 | quantQ+qm 0.55 | delta_s mm 0.90 | quantK 0.52 | quantV 0.56 |
attn **13.62**（before 16.04）| 合計 16.61 ms。

SA2（無変更。参考として after を計測）: 8k 6.37 / 16k 22.42 / 32k 86.88 ms（cos 0.99925）。ブリーフの値と同等。

### 計測上の注意
- 途中から ComfyUI（`main.py --use-sage-attention`）が GPU を 96 % 使い始めた。その間の数値はすべて破棄した。
  採用した数値は、GPU 使用率 <10 % が 12 s 続いたことを確認してから計測したものと、7 回の最小値を取ったもの。
  アブレーション（§3.2）とマイクロベンチ（§3.1）は ComfyUI の起動前に計測した。
- `ncu` は使えないため、アブレーション、SASS 静的解析、raw PTX マイクロベンチを組み合わせて切り分けた。

## 6. 再現

```bash
# SA3 ビルド（ヘッダだけ変えたときは api.cu を touch すること: L-5）
cd sageattention3_blackwell && touch sageattn3/blackwell/api.cu && MAX_JOBS=6 python setup.py build_ext --inplace
cd /tmp
PYTHONPATH=<wt>/sageattention3_blackwell python <wt>/bench/sa3_mask_test.py         # C-1/C-2 の回帰テスト
PYTHONPATH=<wt> python <wt>/bench/opus55/l2_edge.py                                # L2 ビット一致 1536 構成 + stream/graph/決定性
SAGE3_NUM_SM=47 python <wt>/bench/opus55/occ_test.py                               # 1 SM での前進性
nvcc -O3 -gencode arch=compute_121a,code=sm_121a <wt>/bench/opus55/mb.cu -o mb && ./mb   # MMA/MUFU の上限
python <wt>/bench/opus55/make_variant.py NAME MASK [STAGES]                        # アブレーションのビルド（/tmp に出力）
python <wt>/bench/opus55/time_variant.py NAME 16384
```
