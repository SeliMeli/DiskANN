# RobustPrune × HashPrune — experiment results

WSL: 31 GB RAM, 16 cores (AVX2, no AVX-512), NVMe. max_degree=64, beam_width=4.
All runs contention-free (no concurrent compilation), page cache pre-warmed.
Runner `scripts/run_exp.sh` (RSS watchdog). Raw logs in `experiments/logs/`.

## TL;DR
- **HashPrune's purpose is bounded memory**, not better prune quality: `npoints × l_max`
  regardless of fanout/replication. RobustPrune-merge must accumulate candidates → more RAM.
  **BigANN 10M: HashPrune 10.67 GB vs accumulate(merge_l_max=128) 14.9 GB** (+10 GB at cap 256).
- **The paper's claim reproduces (measured correctly): RobustPrune-in-leaf is worse than k-NN.**
  In the controlled, iso-memory comparison (same leaf→HashPrune→final-prune pipeline, only the
  leaf candidate generator differs), k-NN beats RobustPrune on **both recall and QPS**.
- My earlier "RobustPrune-merge wins" was an **artifact**: it was measured at recall@1000
  (IO-bound — density buys recall for free) and used an *accumulator that keeps 4× more
  candidates* (more recall at higher memory). At recall@10 (CPU-bound), with an iso-budget
  HashPrune, the result flips.
- **Best config found: k-NN → HashPrune(l_max=128) → final RobustPrune** — beats the baseline
  and every RobustPrune-in-leaf variant on recall AND QPS, at HashPrune's bounded memory.

---
## A. Enron 1M (384d, cosine) — controlled comparison, recall@10 (CPU-bound)

Same pipeline (leaf candidates → **HashPrune(l_max)** → final RobustPrune), iso-memory; only
the leaf candidate generator / reservoir size changes:

| config | leaf candidates | l_max | avg_deg | R@10 L=10 | QPS L=10 | R@10 L=40 | QPS L=40 | peak RSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mA | k-NN (k=2) | 64 | **47.2** | 70.73 | 2392 | 86.51 | 1265 | 2.28 GB |
| **mB** | **RobustPrune→64** | 64 | **64.0** | 66.10 | 2310 | 83.46 | 1203 | 2.27 GB |
| mC | k-NN (k=3) | **128** | 58.9 | **74.82** | **2573** | **88.93** | **1333** | 2.33 GB |

- **mA (k-NN) vs mB (RobustPrune-in-leaf), iso-everything:** mB is denser (64.0 vs 47.2) and
  **worse on both recall (83.5 vs 86.5) and QPS (1203 vs 1265)**. ⇒ confirms the paper:
  RobustPrune-in-leaf over-densifies and degrades the graph.
- **mC (HashPrune l_max=128) is the winner** — bigger reservoir + final prune beats both.

## B. Enron 1M — iso-candidate-budget (128): HashPrune vs RobustPrune-accumulate

| config | approach | budget | avg_deg | R@10 L=40 | QPS L=40 | peak RSS |
|---|---|---:|---:|---:|---:|---:|
| mC | k-NN → **HashPrune** → fp | 128 | 58.9 | **88.93** | **1333** | 2.33 GB |
| mD | RobustPrune→128 → **accumulate** → fp | 128 | 59.5 | 85.95 | 1263 | 2.24 GB |

At equal candidate budget, **HashPrune's LSH-bucketed selection beats RobustPrune+accumulate on
recall AND QPS** (and is bounded-memory by construction). Answers "l_max=128 vs RobustPrune-128".

## C. Enron 1M — the earlier (accumulate) variants, for reference, recall@10

| config | leaf→merge | avg_deg | R@10 L=40 | QPS L=40 | peak RSS |
|---|---|---:|---:|---:|---:|
| base_pipnn | k-NN→HashPrune(64), **no fp** | 58.9 | 86.63 | 1257 | 2.33 GB |
| exp1 | RobustPrune all-pairs→accumulate(256)→fp | 63.6 | 88.70 | 1280 | 3.29 GB |
| exp2 | GEMM-128 RobustPrune→accumulate(256)→fp | 63.6 | 88.77 | 1275 | 3.30 GB |

exp1/exp2 only looked good vs the *no-final-prune* base; against **mC** they lose on recall, QPS
*and* memory (3.3 GB vs 2.33 GB). The "win" was accumulate keeping more candidates, not RobustPrune.

## D. Enron 1M — recall@1000 (IO-bound) — why it misled

| L | base_pipnn | exp1 | exp2 |
|---|---:|---:|---:|
| 1000 | 89.53 | 90.68 | 90.69 |
| 2000 | 96.62 | 97.68 | 97.68 |
| 3000 | 98.11 | 98.80 | 98.80 |

QPS iso (~74→25). At recall@1000 each query does ~1000 IOs (~200 ms) so QPS is IO-capped and the
denser RobustPrune graphs get higher recall "for free". This regime hides the density/QPS penalty.

---
## E. BigANN 10M (128d, L2) — recall@10 (the paper's dataset)

| variant | leaf→merge | build | avg_deg | peak RSS | R@10 L=50 | QPS L=50 |
|---|---|---:|---:|---:|---:|---:|
| **base** | k-NN→HashPrune(64), no fp | 206 s | 62.2 | **10.67 GB** | **96.36** | **1021** |
| exp2 | GEMM-128 RobustPrune→accumulate(128)→fp | 484 s | 57.2 | 14.92 GB | 94.83 | 867 |
| control | k-NN top-128, no leaf prune→accumulate→fp | 259 s | 50.6 | 14.63 GB | 62.60 | 696 |

- **HashPrune base wins recall AND QPS** over RobustPrune-merge (exp2), at far lower memory.
- The no-leaf-prune top-128 control is *terrible* (62.6%) — redundant near-duplicate candidates
  make a poorly-navigable graph. (exp1 all-to-all at 10M was stopped — redundant with exp2, ~8 min/build.)

---
## Conclusions
1. **Keep HashPrune.** It is the memory-bounding mechanism and, measured correctly (CPU-bound,
   iso-budget), it produces a *better or equal* graph than RobustPrune-in-leaf at lower memory.
2. **RobustPrune-in-leaf hurts** (over-dense, more comps/hop) — confirms the paper, on both datasets.
3. **The one genuinely useful knob: a bigger HashPrune reservoir + final prune** (mC, l_max=128) —
   the best Enron recall@10 config, still bounded memory.
4. Method lesson: measure at the **recall regime you care about**. recall@1000 (IO-bound) inverts
   the apparent ranking vs recall@10 (CPU-bound).

Caveat: Enron baseline leaf_k=3; tuned is 2 (minor). Two L=160 QPS points (mA, exp2) are transient
outliers (~99/89 vs ~430) — recall there is reliable; use L=10/40 for QPS.

### armH_hp — rc=0, Build time: 61.126s, peakRSS=2.29GB, 2026-06-01T02:40:12+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   1841.2      8440.8us       13697us       19733us   30.7   7034.1us   1319.4us               87.3us      1521.9       30.7         0.0%  67.780
15  10   1688.5      9368.8us       14809us       24727us   35.8   7804.2us   1471.1us               93.4us      1732.9       35.8         0.0%  74.910
20  10    122.0    131014.8us      667275us     1192685us   40.8 128908.6us   2003.6us              102.5us      1934.5       40.8         0.0%  78.420
30  10   1247.2     12718.5us       18437us       33035us   50.4  10558.4us   2060.7us               99.4us      2309.1       50.4         0.0%  82.410
40  10   1061.9     14910.4us       22217us       36540us   60.1  12444.6us   2356.3us              109.5us      2673.0       60.1         0.0%  84.900
60  10    827.8     19150.0us       27663us       45692us   79.5  16184.3us   2866.3us               99.4us      3369.6       79.5         0.0%  88.060
80  10    699.8     22658.8us       32418us       50232us   98.9  19338.5us   3208.1us              112.2us      4044.0       98.9         0.0%  90.290
120  10    505.2     31358.1us       42648us       56769us  137.7  27116.8us   4125.5us              115.8us      5343.9      137.7         0.0%  92.820
160  10    407.1     39083.7us       51563us       68003us  176.4  34111.9us   4858.7us              113.0us      6591.4      176.4         0.0%  93.990
```

### armHfp_hp_rp — rc=0, Build time: 65.162s, peakRSS=2.32GB, 2026-06-01T02:42:01+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2101.7      7502.2us       12358us       24063us   30.6   6301.6us   1127.3us               73.3us      1519.1       30.6         0.0%  67.680
15  10   1759.8      8985.1us       14138us       29631us   35.8   7518.9us   1386.8us               79.3us      1731.9       35.8         0.0%  74.820
20  10   1543.7     10241.2us       16482us       24508us   40.8   8547.4us   1608.3us               85.5us      1935.1       40.8         0.0%  78.490
30  10   1320.5     12015.4us       19666us       27068us   50.5  10178.1us   1751.2us               86.1us      2311.8       50.5         0.0%  82.560
40  10   1127.3     14018.2us       23878us       37248us   60.2  11938.0us   1996.7us               83.6us      2677.1       60.2         0.0%  85.060
60  10    829.9     19108.8us       27155us       39432us   79.4  15933.5us   3070.6us              104.6us      3365.5       79.4         0.0%  88.030
80  10    704.1     22560.3us       34232us       44985us   98.8  19187.8us   3270.8us              101.8us      4041.4       98.8         0.0%  90.310
120  10    522.8     30437.7us       42248us       59509us  137.7  26134.7us   4198.1us              105.0us      5344.3      137.7         0.0%  92.830
160  10    397.6     39939.1us       52338us       64932us  176.4  34125.3us   5695.5us              118.3us      6589.7      176.4         0.0%  93.930
```

### armR_acc_rp — rc=0, Build time: 68.515s, peakRSS=5.36GB, 2026-06-01T02:43:52+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2216.8      7103.5us       11024us       19377us   28.9   5954.5us   1075.3us               73.7us      1622.9       28.9         0.0%  74.660
15  10   1860.8      8430.9us       12477us       23154us   33.5   6952.2us   1395.8us               82.9us      1841.8       33.5         0.0%  80.710
20  10   1718.1      9181.4us       15138us       24761us   38.1   7688.6us   1404.5us               88.3us      2057.1       38.1         0.0%  83.550
30  10   1451.5     10883.5us       16900us       25189us   47.3   9185.8us   1618.5us               79.3us      2476.4       47.3         0.0%  87.430
40  10   1190.3     13272.8us       20452us       34322us   56.7  11197.9us   1988.3us               86.7us      2890.9       56.7         0.0%  89.590
60  10    937.9     16895.2us       25994us       37721us   75.8  14462.4us   2341.6us               91.2us      3715.7       75.8         0.0%  92.070
80  10    735.6     21565.5us       32572us       51880us   95.2  18502.1us   2971.4us               91.9us      4536.1       95.2         0.0%  93.770
120  10    528.8     30023.3us       42431us       60946us  134.0  25800.5us   4124.8us               98.0us      6128.3      134.0         0.0%  95.570
160  10    427.2     37139.2us       52062us       73776us  173.2  32280.1us   4757.2us              101.8us      7681.0      173.2         0.0%  96.710
```

### armRP4_hp — rc=0, Build time: 77.138s, peakRSS=2.32GB, 2026-06-01T07:41:33+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2211.4      7118.9us       11272us       18368us   30.5   5999.3us   1041.5us               78.1us      1642.1       30.5         0.0%  70.030
15  10   1943.7      8133.9us       13209us       25274us   35.5   6844.5us   1211.8us               77.6us      1882.7       35.5         0.0%  76.760
20  10   1713.4      9219.2us       15593us       34306us   40.3   7914.4us   1234.2us               70.6us      2104.1       40.3         0.0%  80.090
30  10   1398.2     11312.9us       19451us       28461us   49.8   9746.1us   1494.7us               72.1us      2527.6       49.8         0.0%  84.390
40  10   1218.0     12996.7us       21645us       43288us   59.4  11258.2us   1662.5us               76.0us      2943.6       59.4         0.0%  86.850
60  10    938.3     16895.4us       28428us       41347us   78.6  14662.7us   2151.7us               80.9us      3753.1       78.6         0.0%  89.920
80  10    755.8     20961.6us       33084us       46649us   97.8  18342.4us   2532.8us               86.4us      4531.7       97.8         0.0%  91.840
120  10    544.1     29136.7us       43263us       59420us  136.4  25558.6us   3480.9us               97.2us      6050.9      136.4         0.0%  94.060
160  10    423.3     37506.6us       51921us       61668us  175.2  33032.7us   4367.7us              106.1us      7517.7      175.2         0.0%  95.240
```

### armRP8_hp — rc=0, Build time: 79.126s, peakRSS=2.30GB, 2026-06-01T07:43:08+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2344.6      6728.6us       11473us       24048us   30.3   5839.9us    829.4us               59.3us      1605.9       30.3         0.0%  70.490
15  10   2058.7      7671.3us       12852us       26487us   35.2   6731.6us    885.0us               54.7us      1825.6       35.2         0.0%  77.030
20  10   1785.6      8860.0us       15594us       27748us   40.1   7674.5us   1116.3us               69.1us      2033.2       40.1         0.0%  79.830
30  10   1459.7     10746.6us       20674us       37932us   49.7   9422.4us   1258.6us               65.6us      2431.9       49.7         0.0%  83.820
40  10   1195.0     13254.4us       23338us       38186us   59.4  11637.7us   1545.2us               71.5us      2817.9       59.4         0.0%  86.030
60  10    932.9     17006.3us       29540us       44897us   78.8  15008.4us   1914.9us               83.0us      3559.3       78.8         0.0%  89.280
80  10    754.6     20990.7us       34088us       52535us   97.9  18155.7us   2743.7us               91.3us      4256.7       97.9         0.0%  91.030
120  10    544.1     28905.0us       42791us       61625us  136.7  25584.2us   3229.8us               91.0us      5627.9      136.7         0.0%  93.290
160  10    423.5     37475.7us       53640us       75227us  175.7  33335.1us   4045.0us               95.5us      6951.4      175.7         0.0%  94.640
```

### ppK_knn3 — rc=0, Build time: 68.430s, peakRSS=2.30GB, 2026-06-01T08:10:22+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   1788.4      7899.7us       12368us      261069us   31.6   7106.3us    729.7us               63.7us      1104.8       31.6         0.0%  71.080
15  10   1883.5      8371.0us       13420us       26477us   36.4   7482.3us    824.8us               64.0us      1242.5       36.4         0.0%  77.230
20  10   1683.8      9383.2us       15418us       28091us   41.4   8384.0us    928.8us               70.4us      1384.5       41.4         0.0%  81.100
30  10   1374.0     11476.4us       21828us       31075us   50.8  10253.4us   1146.4us               76.6us      1641.9       50.8         0.0%  85.090
40  10   1223.7     12943.0us       20989us       35179us   60.4  11599.9us   1271.3us               71.9us      1897.0       60.4         0.0%  87.670
60  10    874.9     18127.5us       29711us       44990us   79.4  16132.8us   1906.0us               88.7us      2388.2       79.4         0.0%  90.580
80  10    720.0     22053.3us       32953us       48971us   98.5  19704.6us   2260.8us               87.8us      2872.0       98.5         0.0%  92.340
120  10    121.5    131287.7us      745214us     2964019us  137.4 128093.2us   3094.4us              100.1us      3836.7      137.4         0.0%  94.500
160  10    399.5     39790.2us       52310us       66081us  176.6  35522.9us   4161.9us              105.4us      4772.1      176.6         0.0%  95.580
```

### ppR_m64 — rc=0, Build time: 239.208s, peakRSS=2.26GB, 2026-06-01T08:14:39+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2095.0      7507.5us       12338us       22568us   32.6   6540.3us    888.1us               79.0us      1008.6       32.6         0.0%  64.220
15  10   1835.3      8592.0us       14357us       26572us   38.1   7496.6us   1022.2us               73.2us      1148.5       38.1         0.0%  71.470
20  10   1586.7      9962.2us       17283us       29905us   43.3   8741.8us   1141.4us               78.9us      1277.8       43.3         0.0%  75.250
30  10   1394.3     11371.6us       19346us       30788us   53.1  10144.5us   1151.0us               76.1us      1508.3       53.1         0.0%  79.660
40  10   1131.1     14003.7us       24547us       51031us   62.8  12512.8us   1409.1us               81.8us      1731.4       62.8         0.0%  82.490
60  10    913.9     17375.2us       29355us       46757us   82.2  15595.6us   1691.1us               88.5us      2158.6       82.2         0.0%  86.100
80  10    718.8     22107.7us       36126us       57104us  101.6  19750.3us   2255.8us              101.6us      2573.3      101.6         0.0%  88.390
120  10    530.9     29920.6us       45297us       73215us  140.7  27033.8us   2786.1us              100.7us      3380.1      140.7         0.0%  91.170
160  10    413.0     38479.0us       56291us       83186us  179.7  34831.2us   3543.8us              104.0us      4164.0      179.7         0.0%  92.780
```

### ppR_k3 — rc=0, Build time: 109.538s, peakRSS=2.32GB, 2026-06-01T08:16:46+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2070.0      7624.9us       12317us       20208us   33.1   6777.7us    780.5us               66.7us       930.9       33.1         0.0%  62.410
15  10   1857.0      8498.5us       14364us       25476us   38.6   7674.5us    761.0us               62.9us      1064.7       38.6         0.0%  70.040
20  10   1624.3      9737.0us       17161us       36994us   43.8   8785.5us    887.5us               64.1us      1192.0       43.8         0.0%  73.710
30  10   1351.2     11712.8us       22224us       36577us   53.4  10545.9us   1094.3us               72.6us      1420.3       53.4         0.0%  78.480
40  10   1152.6     13734.4us       26882us       41859us   63.4  12459.5us   1202.3us               72.6us      1650.8       63.4         0.0%  82.230
60  10    889.0     17869.3us       29920us       45627us   82.4  16117.3us   1666.2us               85.8us      2075.5       82.4         0.0%  86.260
80  10    728.8     21769.0us       35573us       52503us  101.4  19751.6us   1934.2us               83.2us      2494.0      101.4         0.0%  88.490
120  10    529.9     29941.3us       45251us       61729us  140.0  27267.7us   2584.9us               88.7us      3319.0      140.0         0.0%  91.570
160  10    420.6     37748.5us       55021us       73546us  178.9  34389.6us   3264.6us               94.4us      4137.7      178.9         0.0%  93.150
```

### ppK_raw — rc=0, Build time: 66.666s, peakRSS=2.29GB, 2026-06-01T08:19:58+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2240.3      7042.4us       11134us       20308us   29.9   6002.5us    967.4us               72.5us      1481.7       29.9         0.0%  74.060
15  10   1994.7      7929.4us       12180us       24297us   34.6   6761.0us   1092.9us               75.5us      1670.2       34.6         0.0%  79.820
20  10   1687.5      9344.4us       14439us       21303us   39.3   7869.7us   1382.2us               92.5us      1857.5       39.3         0.0%  82.860
30  10   1396.0     11301.8us       18877us       30028us   49.0   9641.4us   1566.9us               93.5us      2223.5       49.0         0.0%  86.820
40  10   1184.7     13328.7us       21975us       32002us   58.5  11420.4us   1818.3us               90.1us      2572.3       58.5         0.0%  89.140
60  10    932.5     17026.7us       25656us       41158us   77.7  14771.6us   2158.3us               96.8us      3255.7       77.7         0.0%  91.770
80  10    730.5     21678.4us       32384us       40878us   96.9  18798.8us   2777.9us              101.7us      3923.1       96.9         0.0%  93.300
120  10    542.3     29294.4us       43566us       54589us  135.8  25819.9us   3380.2us               94.4us      5230.4      135.8         0.0%  95.140
160  10    428.5     37059.3us       50369us       66146us  174.9  32834.2us   4122.1us              103.1us      6501.9      174.9         0.0%  96.160
```

### ppR_raw — rc=0, Build time: 226.385s, peakRSS=2.31GB, 2026-06-01T08:24:04+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2041.2      7730.2us       13128us       23596us   31.0   6350.9us   1308.0us               71.3us      1474.3       31.0         0.0%  66.040
15  10   1782.7      8796.7us       14415us       23496us   36.4   7252.2us   1470.7us               73.8us      1681.2       36.4         0.0%  73.700
20  10   1658.8      9503.5us       15494us       38947us   41.7   7871.1us   1568.8us               63.5us      1881.9       41.7         0.0%  77.810
30  10   1333.5     11836.8us       20122us       30464us   51.5   9812.4us   1952.4us               72.0us      2226.6       51.5         0.0%  81.580
40  10   1147.7     13807.2us       23021us       40997us   61.2  11542.4us   2193.1us               71.7us      2555.0       61.2         0.0%  84.140
60  10    883.3     17969.3us       28041us       39139us   80.4  15053.0us   2835.1us               81.2us      3181.4       80.4         0.0%  87.230
80  10    700.0     22694.4us       35389us       56909us   99.9  19106.9us   3498.1us               89.3us      3790.2       99.9         0.0%  89.470
120  10    517.7     30586.1us       44198us       69277us  139.1  25918.3us   4581.4us               86.4us      4969.6      139.1         0.0%  92.040
160  10    400.4     39688.3us       54136us       69782us  177.8  33552.1us   6034.9us              101.4us      6086.9      177.8         0.0%  93.300
```

### vamana_robust — rc=0, Build time: 227.165s, peakRSS=1.21GB, 2026-06-01T10:00:10+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2489.4      6346.2us        9928us       21664us   28.2   5433.9us    846.3us               66.0us      1457.2       28.2         0.0%  76.180
15  10   2089.3      7511.0us       12516us       24948us   32.9   6467.7us    974.0us               69.2us      1647.4       32.9         0.0%  82.690
20  10   1844.4      8576.0us       14186us       27034us   37.8   7330.0us   1167.3us               78.6us      1832.4       37.8         0.0%  85.390
30  10   1415.9     11169.9us       20040us       30886us   47.2   8790.3us   2299.8us               79.8us      2187.1       47.2         0.0%  88.320
40  10   1298.3     12206.5us       19985us       40584us   56.8  10543.4us   1586.5us               76.6us      2534.5       56.8         0.0%  90.560
60  10    970.4     16322.8us       30235us       45573us   75.8  14362.3us   1876.6us               83.9us      3201.9       75.8         0.0%  92.500
80  10    770.9     20556.3us       33933us       44515us   95.2  18069.2us   2402.4us               84.7us      3866.9       95.2         0.0%  93.670
120  10    561.2     28315.6us       43434us       56059us  134.1  25158.1us   3071.3us               86.2us      5162.3      134.1         0.0%  95.390
160  10    438.7     36269.7us       52687us       74669us  173.3  32716.5us   3472.9us               80.4us      6424.3      173.3         0.0%  96.110
```

### vamana_hp — rc=0, Build time: 189.764s, peakRSS=1.29GB, 2026-06-01T10:03:34+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2279.8      6915.1us       11258us       29189us   31.8   6018.7us    846.9us               49.5us      1376.7       31.8         0.0%  64.270
15  10   1958.2      8063.7us       14317us       28154us   37.3   6983.0us   1021.7us               58.9us      1558.8       37.3         0.0%  71.160
20  10   1749.2      9044.3us       15181us       31811us   42.6   7863.7us   1128.1us               52.6us      1730.8       42.6         0.0%  74.420
30  10   1426.5     11099.1us       18495us       32538us   52.5   9647.5us   1391.9us               59.8us      2038.0       52.5         0.0%  78.460
40  10   1209.4     13112.5us       24469us       38016us   62.4  11492.6us   1552.7us               67.2us      2332.4       62.4         0.0%  81.210
60  10    913.0     17308.6us       31752us       48943us   82.1  15309.2us   1933.8us               65.5us      2900.3       82.1         0.0%  84.920
80  10    756.2     21009.9us       35810us       52340us  101.5  18655.9us   2285.8us               68.3us      3439.3      101.5         0.0%  86.950
120  10    541.4     29352.2us       46954us       79325us  140.2  26156.1us   3122.6us               73.5us      4471.2      140.2         0.0%  89.650
160  10    414.9     37778.4us       54860us       90876us  179.3  33671.0us   4027.9us               79.5us      5483.5      179.3         0.0%  90.910
```

### vamana_robust_r128 — rc=137, build:?, peakRSS=1.34GB, 2026-06-01T10:31:57+00:00
```
```

### vamana_hp_r128 — rc=137, build:?, peakRSS=0.03GB, 2026-06-01T10:32:05+00:00
```
```

### vamana_hp_lmax128 — rc=1, build:?, peakRSS=0.80GB, 2026-06-01T10:39:20+00:00
```
```

### vamana_hp_lmax128 — rc=0, Build time: 494.470s, peakRSS=1.50GB, 2026-06-01T10:58:52+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2298.6      6847.1us       10704us       19431us   28.6   5856.2us    940.8us               50.1us      1429.1       28.6         0.0%  74.120
15  10   2117.9      7448.3us       11715us       27581us   33.5   6414.4us    984.1us               49.7us      1614.8       33.5         0.0%  80.360
20  10   1701.7      9305.7us       14531us       28369us   38.4   7970.4us   1275.7us               59.5us      1795.7       38.4         0.0%  83.110
30  10   1476.8     10715.1us       17638us       31399us   48.0   9336.6us   1318.5us               60.0us      2134.5       48.0         0.0%  86.450
40  10   1290.0     12297.5us       20026us       39090us   57.6  10831.2us   1406.3us               60.1us      2460.3       57.6         0.0%  88.800
60  10    982.7     16136.7us       28834us       44781us   76.7  14244.9us   1826.8us               65.0us      3087.2       76.7         0.0%  91.110
80  10    780.2     20258.8us       34835us       44837us   96.2  18097.8us   2095.7us               65.2us      3719.7       96.2         0.0%  92.540
120  10    107.4    148715.9us      737996us     2399878us  135.4 145026.7us   3596.9us               92.2us      4945.8      135.4         0.0%  94.360
160  10    432.7     36739.9us       51553us       62911us  174.2  32835.7us   3809.9us               94.3us      6111.7      174.2         0.0%  95.230
```

### ppR_topk64_k3 — rc=0, Build time: 130.346s, peakRSS=2.31GB, 2026-06-03T11:23:04+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2061.4      7284.2us       11498us      288193us   33.6   6731.0us    521.9us               31.3us       919.1       33.6         0.0%  62.170
15  10   1541.6      8395.5us       12507us      293095us   39.0   7889.7us    470.7us               35.1us      1050.3       39.0         0.0%  69.620
20  10   1229.8      9451.9us       14721us      284804us   44.2   8822.1us    595.5us               34.3us      1175.5       44.2         0.0%  73.710
30  10   1426.6     11121.9us       19449us      208312us   54.0  10392.4us    693.3us               36.2us      1404.0       54.0         0.0%  78.850
40  10   1197.8     13256.5us       26642us       44854us   64.0  12389.1us    831.7us               35.8us      1629.0       64.0         0.0%  82.580
60  10    924.8     17190.7us       35270us       50472us   82.8  16106.0us   1050.4us               34.3us      2040.0       82.8         0.0%  86.080
80  10    747.5     21268.2us       40019us       62640us  101.8  19984.9us   1248.8us               34.5us      2449.7      101.8         0.0%  88.200
120  10    189.6     69772.7us      416581us     1561831us  140.5  67934.1us   1798.6us               40.0us      3260.9      140.5         0.0%  91.320
160  10    187.6     85038.5us      552525us     1546309us  179.5  82696.1us   2292.3us               50.0us      4067.6      179.5         0.0%  92.900
```

### hp512_k2 — rc=0, Build time: 117.671s, peakRSS=2.33GB, 2026-06-08T12:12:28+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2289.4      6902.9us       11161us       29543us   29.7   6239.7us    633.0us               30.2us      1381.5       29.7         0.0%  72.140
15  10   2098.5      7539.7us       11920us       29229us   34.6   6840.1us    669.9us               29.7us      1557.5       34.6         0.0%  78.420
20  10   1774.7      8904.0us       17647us      243346us   39.5   8155.8us    718.4us               29.8us      1726.5       39.5         0.0%  81.570
30  10   1371.2     11552.1us       20810us      298312us   48.8  10591.4us    928.5us               32.2us      2043.0       48.8         0.0%  85.360
40  10   1267.3     12506.6us       19486us       38727us   58.3  11404.9us   1069.6us               32.1us      2353.2       58.3         0.0%  87.460
60  10    984.4     16115.7us       26273us       40624us   77.5  14780.0us   1307.3us               28.4us      2970.9       77.5         0.0%  90.560
80  10    784.4     20269.5us       33839us       53964us   96.6  18655.8us   1581.4us               32.2us      3563.9       96.6         0.0%  92.270
120  10    107.0    149395.1us      848159us     1597885us  135.5 146808.5us   2541.9us               44.7us      4747.1      135.5         0.0%  94.430
160  10    414.5     38380.3us       51642us       67723us  174.7  35614.0us   2728.9us               37.3us      5900.1      174.7         0.0%  95.610
```

### acc512_k2 — rc=0, Build time: 64.321s, peakRSS=2.26GB, 2026-06-08T12:14:15+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2475.4      6374.0us        9497us       26761us   30.2   5767.7us    578.4us               28.0us      1248.6       30.2         0.0%  70.260
15  10   2180.0      7242.1us       11293us       27391us   35.1   6587.6us    625.4us               29.1us      1399.1       35.1         0.0%  77.000
20  10   1830.3      8625.3us       14009us       31946us   39.8   7756.3us    835.3us               33.7us      1540.6       39.8         0.0%  79.870
30  10   1516.9     10429.7us       16322us       36456us   49.2   9548.5us    850.6us               30.6us      1819.6       49.2         0.0%  84.190
40  10   1268.1     12523.2us       21169us       37265us   58.5  11380.7us   1111.6us               30.9us      2088.4       58.5         0.0%  86.280
60  10    942.2     16843.6us       33546us       60741us   77.7  15550.6us   1260.8us               32.2us      2634.1       77.7         0.0%  89.740
80  10    752.8     21095.3us       37518us       69177us   97.1  19486.1us   1575.7us               33.6us      3168.6       97.1         0.0%  91.960
120  10    557.7     28533.5us       43295us       58116us  135.9  26446.8us   2049.8us               36.9us      4209.4      135.9         0.0%  94.320
160  10    435.2     36570.7us       52590us       66135us  175.0  33996.6us   2536.1us               38.0us      5235.1      175.0         0.0%  95.360
```

### hp512_k2_warm — rc=0, Build time: 48.725s, peakRSS=2.35GB, 2026-06-08T12:17:33+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2474.3      6370.1us       10062us       24432us   29.7   5694.0us    645.5us               30.6us      1383.5       29.7         0.0%  72.050
15  10   1921.7      8208.0us       12406us       68663us   34.6   7455.8us    720.0us               32.2us      1558.3       34.6         0.0%  78.260
20  10   1839.1      8578.3us       14056us       29862us   39.5   7678.7us    871.5us               28.1us      1727.3       39.5         0.0%  81.490
30  10   1531.6     10332.5us       15766us       32677us   48.9   9307.6us    989.5us               35.5us      2044.3       48.9         0.0%  85.370
40  10   1277.1     12387.5us       19104us       39746us   58.3  11108.4us   1242.7us               36.4us      2355.4       58.3         0.0%  87.420
60  10    960.6     16538.5us       29316us       41880us   77.5  15015.8us   1487.4us               35.3us      2970.8       77.5         0.0%  90.480
80  10    784.1     20286.8us       34672us       46346us   96.6  18493.9us   1757.9us               35.0us      3564.2       96.6         0.0%  92.160
120  10    558.3     28459.2us       41903us       54075us  135.6  26016.3us   2402.3us               40.6us      4750.9      135.6         0.0%  94.440
160  10    423.1     37420.7us       54888us       84109us  174.8  34542.5us   2840.3us               37.9us      5903.9      174.8         0.0%  95.620
```

### acc512_k2_warm — rc=0, Build time: 56.912s, peakRSS=2.25GB, 2026-06-08T12:19:40+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2325.7      6773.4us       10527us       25579us   30.2   6083.5us    651.0us               38.9us      1248.6       30.2         0.0%  70.260
15  10   2095.3      7512.6us       11973us       24561us   35.1   6719.4us    756.2us               37.0us      1399.1       35.1         0.0%  77.000
20  10   1742.3      9092.2us       14649us       31198us   39.8   8144.6us    903.6us               44.0us      1540.6       39.8         0.0%  79.870
30  10   1486.1     10654.4us       20516us       32165us   49.2   9665.1us    952.1us               37.2us      1819.6       49.2         0.0%  84.190
40  10   1241.7     12766.6us       21235us       38458us   58.5  11553.0us   1169.0us               44.6us      2088.4       58.5         0.0%  86.280
60  10    963.0     16486.5us       28337us       38425us   77.7  14982.6us   1461.2us               42.7us      2634.1       77.7         0.0%  89.740
80  10    769.6     20628.5us       32957us       44834us   97.1  18823.9us   1756.0us               48.6us      3168.6       97.1         0.0%  91.960
120  10    534.9     29683.1us       42433us       56257us  135.9  26898.5us   2723.0us               61.6us      4209.4      135.9         0.0%  94.320
160  10    429.9     36881.3us       50927us       70477us  175.0  33884.6us   2937.7us               58.9us      5235.1      175.0         0.0%  95.360
```

### hp512_k2_lmax8 — rc=0, Build time: 52.518s, peakRSS=2.34GB, 2026-06-08T12:30:07+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
40  10    808.8     19537.1us       37728us       71433us   91.6  18976.5us    522.7us               37.9us       342.3       91.6         0.0%  11.180
```

### hp512_k2_lmax32 — rc=0, Build time: 53.500s, peakRSS=2.36GB, 2026-06-08T12:31:11+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
40  10   1142.7     13867.5us       26636us       39494us   66.3  13003.7us    834.3us               29.5us      1493.8       66.3         0.0%  78.390
```

### hp512_k2_lmax64 — rc=0, Build time: 55.536s, peakRSS=2.35GB, 2026-06-08T12:32:21+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
40  10   1278.0     12371.9us       18713us       34454us   58.3  11202.7us   1132.4us               36.8us      2354.7       58.3         0.0%  87.320
```

### vamana_rp_base2 — rc=0, Build time: 227.875s, peakRSS=1.20GB, 2026-06-09T07:02:16+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2321.0      6782.3us       11468us       29754us   28.3   5877.5us    865.6us               39.3us      1466.7       28.3         0.0%  76.980
15  10   2116.6      7476.4us       12026us       28624us   33.1   6618.6us    825.4us               32.3us      1655.6       33.1         0.0%  83.270
20  10   1899.4      8326.2us       12544us       35541us   37.9   7343.3us    945.6us               37.2us      1837.7       37.9         0.0%  85.740
30  10   1546.3     10257.2us       17699us       31105us   47.2   9171.0us   1050.0us               36.2us      2187.4       47.2         0.0%  88.660
40  10   1336.7     11888.0us       17286us       32202us   56.6  10643.1us   1206.7us               38.2us      2527.1       56.6         0.0%  90.720
60  10    997.5     15910.0us       26091us       39093us   75.7  14330.3us   1539.4us               40.3us      3197.7       75.7         0.0%  92.630
80  10    788.2     20139.6us       30900us       48192us   95.0  18101.4us   1993.8us               44.5us      3864.4       95.0         0.0%  93.690
120  10    537.7     29589.0us       45150us      105269us  134.3  26871.4us   2668.2us               49.4us      5171.6      134.3         0.0%  95.500
160  10    382.2     41554.1us       62685us      121114us  173.3  38181.8us   3315.3us               57.0us      6430.5      173.3         0.0%  96.220
```

### vamana_hp_res_l64 — rc=0, Build time: 197.010s, peakRSS=1.30GB, 2026-06-09T07:06:08+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2211.3      7122.7us       11501us       36478us   32.0   6370.7us    712.9us               39.2us      1386.6       32.0         0.0%  64.780
15  10   1837.1      8555.2us       15817us       41068us   37.6   7841.2us    683.2us               30.8us      1574.6       37.6         0.0%  72.380
20  10   1713.8      9200.1us       15598us       27889us   42.6   8362.4us    806.7us               31.0us      1730.8       42.6         0.0%  74.940
30  10     98.1    162954.9us      765751us     1403577us   52.5 161230.7us   1677.4us               46.8us      2039.2       52.5         0.0%  78.950
40  10   1181.1     13416.0us       24300us       61837us   62.1  12308.6us   1073.1us               34.3us      2323.5       62.1         0.0%  81.340
60  10    923.3     17213.4us       30881us       47791us   81.8  15615.6us   1561.6us               36.2us      2886.5       81.8         0.0%  84.850
80  10    759.7     20922.3us       36767us       54076us  101.4  19130.9us   1757.3us               34.1us      3436.2      101.4         0.0%  87.400
120  10    539.4     29498.7us       45447us       66206us  139.9  27198.2us   2259.9us               40.6us      4466.0      139.9         0.0%  89.700
160  10    424.7     37417.5us       55969us       75419us  178.8  34317.0us   3054.7us               45.9us      5469.9      178.8         0.0%  90.920
```

### vamana_hp_res_l128 — rc=0, Build time: 498.870s, peakRSS=1.51GB, 2026-06-09T07:15:09+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   1359.3     11678.8us       10691us      471466us   28.5  11019.7us    630.5us               28.6us      1428.1       28.5         0.0%  74.260
15  10    122.7    130372.1us      458038us      721078us   33.5 128844.6us   1482.7us               44.7us      1614.0       33.5         0.0%  80.490
20  10   1782.7      8828.2us       16214us       25289us   38.4   7873.6us    919.6us               35.0us      1794.4       38.4         0.0%  83.370
30  10   1482.9     10665.4us       16541us       31795us   48.0   9508.3us   1122.9us               34.2us      2132.2       48.0         0.0%  86.700
40  10   1262.6     12572.0us       19730us       32900us   57.6  11185.8us   1349.1us               37.1us      2458.2       57.6         0.0%  88.940
60  10    951.6     16655.1us       27443us       43227us   76.5  14979.9us   1637.1us               38.0us      3082.4       76.5         0.0%  91.090
80  10    775.5     20463.8us       31338us       45806us   96.1  18350.8us   2065.9us               47.1us      3713.1       96.1         0.0%  92.540
120  10    533.8     29775.4us       43766us       57183us  135.3  26884.5us   2845.5us               45.4us      4943.8      135.3         0.0%  94.360
160  10    428.7     37057.7us       51429us       65993us  174.1  33580.6us   3424.5us               52.6us      6109.4      174.1         0.0%  95.210
```

### vamana_hp_diag_l64 — rc=0, Build time: 200.186s, peakRSS=1.29GB, 2026-06-09T07:30:24+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
40  10   1128.6     14023.3us       24583us       43478us   62.9  12676.3us   1303.9us               43.2us      2339.8       62.9         0.0%  81.210
```

### vamana_hp_diag_l128 — rc=0, Build time: 498.553s, peakRSS=1.49GB, 2026-06-09T07:38:49+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
40  10   1302.2     12178.4us       21436us       35575us   57.6  10932.6us   1206.5us               39.3us      2460.9       57.6         0.0%  89.010
```

### vamana_hp_slack_l128 — rc=0, Build time: 242.936s, peakRSS=1.61GB, 2026-06-09T07:47:07+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
40  10   1316.3     12026.2us       18884us       36063us   57.5  10771.6us   1215.7us               39.0us      2495.2       57.5         0.0%  89.490
```

### vamana_hp_slack_l128_full — rc=0, Build time: 243.500s, peakRSS=1.61GB, 2026-06-09T07:52:28+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2571.5      6150.9us       10514us       24797us   28.5   5507.3us    609.6us               34.0us      1443.4       28.5         0.0%  75.040
15  10   2170.1      7261.5us       12239us       26294us   33.4   6525.9us    700.7us               34.9us      1631.9       33.4         0.0%  81.110
20  10   1933.8      8171.1us       14529us       31315us   38.3   7370.8us    765.4us               34.9us      1813.8       38.3         0.0%  83.760
30  10   1585.8      9990.6us       16506us       33742us   47.8   9012.5us    942.7us               35.4us      2157.1       47.8         0.0%  86.820
40  10   1320.1     12014.6us       22994us       39875us   57.5  10927.3us   1056.0us               31.3us      2497.1       57.5         0.0%  89.410
60  10   1012.2     15697.8us       30549us       43076us   76.6  14186.8us   1475.8us               35.1us      3142.2       76.6         0.0%  91.740
80  10    783.2     20226.0us       33549us       47623us   96.0  18182.1us   2001.4us               42.6us      3783.3       96.0         0.0%  92.960
120  10    562.2     28257.3us       43449us       60346us  134.9  25825.9us   2391.0us               40.4us      5032.4      134.9         0.0%  94.910
160  10    447.3     35619.6us       50532us       72994us  174.0  32504.4us   3068.6us               46.6us      6235.2      174.0         0.0%  95.970
```

### vamana_hp_online_l128 — rc=1, build:?, peakRSS=1.09GB, 2026-06-09T08:16:12+00:00
```
```

### vamana_hp_online_l128b — rc=0, Build time: 244.478s, peakRSS=2.84GB, 2026-06-09T08:26:47+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2238.3      7049.9us       12778us       26830us   29.9   6240.6us    770.3us               39.0us      1473.9       29.9         0.0%  73.970
15  10   1894.5      8341.8us       14203us       26664us   35.0   7362.2us    936.1us               43.5us      1664.7       35.0         0.0%  80.630
20  10   1814.2      8697.4us       13568us       32412us   39.6   7753.7us    906.7us               37.0us      1835.2       39.6         0.0%  82.770
30  10   1464.8     10793.0us       17147us       30683us   49.2   9586.1us   1168.9us               38.0us      2174.0       49.2         0.0%  85.880
40  10   1234.7     12833.7us       20526us       35161us   58.7  11492.1us   1302.3us               39.4us      2497.5       58.7         0.0%  88.050
60  10    872.7     18163.4us       26581us       37527us   78.0  15906.6us   2202.3us               54.4us      3137.9       78.0         0.0%  90.670
80  10    758.0     20969.0us       33970us       46002us   97.6  18939.2us   1986.8us               43.0us      3770.4       97.6         0.0%  92.580
120  10    545.8     28949.5us       43109us       67512us  136.6  26199.4us   2697.5us               52.7us      4984.3      136.6         0.0%  94.430
160  10    416.7     38169.3us       54033us       84386us  175.7  34531.0us   3580.6us               57.8us      6161.4      175.7         0.0%  95.380
```

### vamana_hp_carry_l128 — rc=0, Build time: 232.771s, peakRSS=2.84GB, 2026-06-09T08:58:11+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2538.8      6219.7us        9582us       27709us   29.7   5538.9us    645.3us               35.5us      1469.7       29.7         0.0%  74.080
15  10   2158.7      7317.4us       12305us       31855us   34.5   6590.5us    695.8us               31.1us      1653.7       34.5         0.0%  80.410
20  10   1924.7      8209.8us       13238us       30414us   39.4   7341.3us    833.9us               34.6us      1829.2       39.4         0.0%  83.100
30  10   1510.3     10477.0us       20275us       32996us   49.0   9226.3us   1212.6us               38.1us      2172.8       49.0         0.0%  86.150
40  10   1313.4     12072.2us       19202us       32253us   58.5  10767.1us   1263.9us               41.2us      2497.4       58.5         0.0%  88.320
60  10    971.2     16340.9us       28724us       45508us   77.7  14510.0us   1787.3us               43.7us      3134.4       77.7         0.0%  90.990
80  10    775.7     20479.8us       32316us       50969us   97.3  18210.7us   2225.8us               43.3us      3763.7       97.3         0.0%  92.600
120  10    553.4     28709.0us       45537us       59858us  136.3  26182.3us   2483.0us               43.7us      4984.2      136.3         0.0%  94.290
160  10    426.7     37199.0us       54247us       66651us  175.2  34109.2us   3046.9us               42.9us      6152.4      175.2         0.0%  95.200
```

### vamana_hp_online_l64 — rc=0, Build time: 174.020s, peakRSS=2.00GB, 2026-06-09T10:49:40+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2219.3      7116.5us       12715us       32348us   33.8   6462.6us    620.7us               33.3us      1360.4       33.8         0.0%  56.740
20  10   1673.0      9483.9us       16483us       32465us   45.5   8646.5us    801.4us               35.9us      1723.5       45.5         0.0%  67.410
40  10   1117.9     14212.1us       29892us       49749us   66.3  12898.0us   1276.7us               37.3us      2319.7       66.3         0.0%  75.490
80  10    721.4     22037.1us       37673us       50846us  105.9  20331.4us   1667.2us               38.5us      3353.5      105.9         0.0%  81.780
160  10    405.9     39074.4us       55281us       75711us  183.8  35730.6us   3295.2us               48.6us      5244.3      183.8         0.0%  86.250
```

### vamana_hp_l83 — rc=0, Build time: 205.515s, peakRSS=2.33GB, 2026-06-09T11:21:31+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2234.7      7050.4us       12604us       31024us   31.8   6374.2us    642.6us               33.5us      1390.6       31.8         0.0%  65.650
15  10   1996.7      7903.9us       12981us       31407us   37.6   7136.7us    733.4us               33.7us      1585.7       37.6         0.0%  73.190
20  10   1662.2      9490.4us       19514us       32450us   42.9   8606.4us    847.7us               36.4us      1762.8       42.9         0.0%  76.760
30  10   1187.8     13378.1us       24833us       40851us   52.7  12369.9us    972.1us               36.0us      2072.2       52.7         0.0%  80.470
40  10   1204.6     13075.8us       21654us       36814us   62.4  11881.5us   1155.2us               39.1us      2363.9       62.4         0.0%  82.820
60  10    807.5     19667.7us       34305us      184080us   81.7  18113.9us   1517.5us               36.3us      2925.3       81.7         0.0%  85.840
80  10    722.1     21967.9us       38581us       67883us  101.1  19971.2us   1952.0us               44.7us      3468.5      101.1         0.0%  87.830
120  10    532.8     29782.0us       46711us       76229us  140.3  27447.3us   2290.8us               43.8us      4536.0      140.3         0.0%  90.260
160  10    421.9     37662.4us       52844us       67207us  179.7  34634.7us   2977.1us               50.6us      5561.3      179.7         0.0%  91.810
```

### vamana_hp_p10_l83 — rc=0, Build time: 185.035s, peakRSS=2.32GB, 2026-06-09T11:25:38+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2488.2      6345.8us       10188us       27725us   30.1   5670.2us    639.4us               36.2us      1377.2       30.1         0.0%  67.660
15  10   2147.1      7353.7us       12142us       27772us   35.2   6643.7us    677.5us               32.5us      1553.2       35.2         0.0%  74.160
20  10   1863.0      8466.4us       14846us       28185us   40.4   7474.5us    954.2us               37.7us      1728.5       40.4         0.0%  77.920
30  10   1499.6     10555.6us       18598us       32186us   50.1   9530.6us    986.4us               38.6us      2043.0       50.1         0.0%  82.000
40  10   1252.8     12654.7us       25772us       38428us   60.0  11539.0us   1078.4us               37.3us      2347.5       60.0         0.0%  84.730
60  10    916.8     17257.7us       33246us       69687us   79.3  15745.6us   1469.8us               42.3us      2931.1       79.3         0.0%  87.770
80  10    759.1     20914.1us       34361us       47931us   98.6  18902.5us   1959.2us               52.4us      3493.5       98.6         0.0%  89.800
120  10    562.3     28273.5us       43565us       64367us  137.5  25970.3us   2258.4us               44.8us      4586.8      137.5         0.0%  91.880
160  10    431.8     36819.8us       53787us       75322us  176.4  34095.7us   2679.7us               44.4us      5639.0      176.4         0.0%  93.180
```

### vamana_hp_p8_l83 — rc=0, Build time: 170.574s, peakRSS=2.32GB, 2026-06-09T11:28:44+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2517.4      6259.0us        9756us       20831us   29.1   5578.9us    643.5us               36.6us      1256.9       29.1         0.0%  67.020
15  10   2029.8      7607.7us       12514us       19222us   34.4   6738.3us    826.8us               42.5us      1420.1       34.4         0.0%  73.580
20  10   1862.4      8483.5us       13827us       22728us   39.2   7579.6us    860.3us               43.6us      1567.9       39.2         0.0%  76.590
30  10   1532.4     10347.5us       18902us       29752us   49.0   9227.5us   1078.5us               41.6us      1852.5       49.0         0.0%  80.670
40  10   1317.4     12016.2us       18797us       37104us   58.5  10943.6us   1037.5us               35.2us      2116.2       58.5         0.0%  82.910
60  10    971.3     16340.6us       29692us       50471us   77.7  14857.4us   1442.8us               40.4us      2636.9       77.7         0.0%  85.780
80  10    787.4     20165.2us       35706us       44775us   97.1  18502.3us   1616.9us               46.0us      3147.7       97.1         0.0%  88.050
120  10    568.2     28006.6us       44204us       61464us  136.1  25851.2us   2107.9us               47.5us      4130.8      136.1         0.0%  90.230
160  10    425.4     37356.6us       53355us       94969us  175.5  34491.0us   2806.7us               58.9us      5093.7      175.5         0.0%  91.880
```

### vamana_hp_p7_l83 — rc=0, Build time: 157.539s, peakRSS=2.32GB, 2026-06-09T11:31:38+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2504.5      6289.2us        9710us       15756us   29.4   5627.6us    624.7us               36.8us      1095.1       29.4         0.0%  60.970
15  10   2071.9      7601.6us       12461us       24294us   34.6   6824.4us    736.3us               40.9us      1227.9       34.6         0.0%  66.690
20  10   1893.5      8333.0us       12886us       29781us   39.7   7616.9us    679.6us               36.5us      1355.3       39.7         0.0%  70.220
30  10   1547.1     10228.9us       17414us       39334us   49.4   9403.2us    790.9us               34.8us      1587.3       49.4         0.0%  74.380
40  10   1257.5     12584.7us       21221us       43005us   59.1  11560.2us    983.0us               41.4us      1810.0       59.1         0.0%  77.030
60  10    926.5     17053.9us       27600us       65046us   78.3  15750.2us   1259.7us               44.0us      2243.2       78.3         0.0%  80.570
80  10    740.9     21405.3us       38151us       57885us   97.9  19514.1us   1840.1us               51.1us      2659.7       97.9         0.0%  83.030
120  10    531.8     29748.4us       45282us       66056us  137.6  27501.9us   2191.5us               55.0us      3483.9      137.6         0.0%  86.780
160  10    420.8     37768.2us       54481us       77920us  176.7  35000.4us   2712.3us               55.5us      4274.1      176.7         0.0%  88.450
```

### vamana_hp_p6_l83 — rc=0, Build time: 137.670s, peakRSS=2.32GB, 2026-06-09T11:34:13+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2351.6      6695.3us       11368us       24359us   31.0   6168.6us    485.3us               41.4us       821.8       31.0         0.0%  51.340
15  10   2047.7      7735.4us       14229us       28789us   36.6   7052.6us    646.7us               36.1us       927.6       36.6         0.0%  56.780
20  10   1810.0      8743.0us       14607us       32113us   41.7   8149.0us    557.8us               36.3us      1021.8       41.7         0.0%  60.290
30  10   1484.6     10634.3us       18897us       32895us   51.5   9764.1us    832.0us               38.2us      1191.7       51.5         0.0%  64.230
40  10   1227.0     12896.9us       28397us       39100us   62.1  12079.2us    782.1us               35.6us      1369.5       62.1         0.0%  68.030
60  10    932.4     16997.3us       30622us       49128us   82.3  15859.3us   1101.1us               36.8us      1693.5       82.3         0.0%  72.690
80  10    724.4     21888.3us       39466us       77998us  101.7  20617.3us   1224.2us               46.8us      1992.4      101.7         0.0%  75.170
120  10    538.8     29503.7us       47289us       75048us  141.5  27764.0us   1696.1us               43.6us      2589.9      141.5         0.0%  78.610
160  10    427.5     37196.5us       53886us       71580us  180.4  35034.1us   2111.3us               51.0us      3152.5      180.4         0.0%  80.730
```

### vamana_hp_p10_l128 — rc=0, Build time: 216.621s, peakRSS=2.84GB, 2026-06-09T11:38:34+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2587.9      6096.3us       10140us       24926us   28.7   5292.1us    767.9us               36.3us      1395.8       28.7         0.0%  73.040
15  10   2184.5      7228.4us       13374us       27155us   33.7   6498.4us    696.0us               34.0us      1583.3       33.7         0.0%  79.420
20  10   2036.7      7757.1us       12471us       24482us   38.5   6779.1us    941.1us               36.9us      1751.4       38.5         0.0%  82.140
30  10   1520.8     10392.2us       21626us       38727us   47.9   9405.6us    951.3us               35.3us      2081.9       47.9         0.0%  85.170
40  10   1315.7     12043.0us       27298us       38137us   57.4  10905.7us   1103.4us               33.9us      2399.0       57.4         0.0%  87.470
60  10    981.2     16168.7us       30244us       41363us   77.0  14343.0us   1784.0us               41.8us      3038.0       77.0         0.0%  90.300
80  10    783.6     20236.8us       34498us       49191us   96.2  18146.0us   2046.1us               44.7us      3641.1       96.2         0.0%  91.910
120  10    564.9     28133.6us       43817us       60985us  135.2  25716.0us   2378.7us               38.9us      4829.8      135.2         0.0%  94.050
160  10    427.5     37242.5us       54847us       72140us  174.4  34027.0us   3177.9us               37.6us      5983.9      174.4         0.0%  95.120
```

### vamana_hp_p8_l128 — rc=0, Build time: 182.963s, peakRSS=2.84GB, 2026-06-09T11:41:53+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2606.4      6059.1us        9715us       31257us   28.2   5505.3us    521.6us               32.2us      1227.3       28.2         0.0%  67.360
15  10   2225.1      7086.4us       11285us       30634us   33.2   6401.2us    648.7us               36.5us      1382.5       33.2         0.0%  73.340
20  10   1996.9      7922.7us       13086us       36839us   38.1   7196.7us    692.0us               34.0us      1534.8       38.1         0.0%  76.740
30  10   1538.1     10307.0us       21254us       48794us   47.8   9431.6us    840.4us               35.0us      1820.9       47.8         0.0%  81.030
40  10   1336.3     11838.3us       19143us       45444us   57.4  10790.6us   1014.0us               33.8us      2092.9       57.4         0.0%  83.460
60  10    943.7     16755.2us       30147us       53449us   76.6  15176.0us   1529.8us               49.3us      2625.2       76.6         0.0%  86.220
80  10    742.1     21433.0us       37878us       70145us   96.3  19594.4us   1791.9us               46.7us      3153.6       96.3         0.0%  88.300
120  10    551.0     28829.1us       44885us       57243us  135.0  26598.2us   2186.7us               44.1us      4150.4      135.0         0.0%  90.600
160  10    425.9     37311.4us       55132us       85804us  174.6  34403.3us   2857.5us               50.6us      5139.6      174.6         0.0%  92.100
```

### vamana_hp_p6_l128 — rc=0, Build time: 145.604s, peakRSS=2.84GB, 2026-06-09T11:44:31+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2337.7      6763.4us       10793us       42059us   31.0   6282.3us    447.2us               33.8us       821.8       31.0         0.0%  51.340
15  10   2119.8      7438.5us       12092us       23481us   36.6   6900.4us    501.4us               36.7us       927.6       36.6         0.0%  56.780
20  10   1730.5      8978.2us       17825us       34927us   41.8   8356.9us    586.1us               35.2us      1022.1       41.8         0.0%  60.370
30  10   1474.3     10599.6us       17384us       38922us   51.5   9738.2us    820.1us               41.3us      1191.4       51.5         0.0%  64.230
40  10   1224.4     12942.9us       23340us       40538us   62.0  12122.3us    784.0us               36.7us      1369.0       62.0         0.0%  68.020
60  10    940.1     16809.2us       30093us       52841us   82.3  15790.5us    985.3us               33.3us      1693.6       82.3         0.0%  72.690
80  10    742.6     21404.1us       37411us       51986us  101.7  19922.6us   1435.5us               46.0us      1992.5      101.7         0.0%  75.180
120  10    530.8     29929.1us       46164us       73474us  141.6  28130.3us   1750.1us               48.6us      2590.2      141.6         0.0%  78.620
160  10    416.4     38176.0us       54304us       75370us  180.4  36098.9us   2027.9us               49.3us      3152.4      180.4         0.0%  80.740
```

### vamana_hp_p_l128 16 — rc=0, Build time: 211.240s, peakRSS=1.20GB, 2026-06-09T11:48:38+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2634.6      5977.5us        9540us       18344us   28.0   5271.2us    673.5us               32.7us      1449.2       28.0         0.0%  76.950
15  10   2273.8      6947.1us       11431us       29054us   32.7   6170.6us    744.3us               32.2us      1633.8       32.7         0.0%  83.160
20  10   1914.3      8235.4us       14798us       31621us   37.5   7342.0us    859.6us               33.7us      1816.9       37.5         0.0%  85.740
30  10   1597.9      9881.0us       16183us       35070us   46.9   8792.3us   1052.7us               36.0us      2168.4       46.9         0.0%  88.790
40  10   1340.6     11799.0us       20091us       36724us   56.2  10442.0us   1318.3us               38.6us      2502.2       56.2         0.0%  90.350
60  10   1000.3     15873.3us       30274us       44258us   75.3  14316.4us   1524.0us               32.8us      3177.7       75.3         0.0%  92.550
80  10    791.0     20072.5us       37713us       56647us   94.7  18234.1us   1805.0us               33.3us      3843.7       94.7         0.0%  93.760
120  10    555.6     28585.7us       46363us       75901us  133.8  26107.7us   2438.9us               39.0us      5151.1      133.8         0.0%  95.440
160  10    443.1     35838.0us       51665us       80477us  173.1  32580.4us   3214.6us               43.0us      6419.0      173.1         0.0%  96.200
```

### vamana_hp_p16_l128 — rc=0, Build time: 241.279s, peakRSS=2.85GB, 2026-06-09T11:56:30+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2460.9      6406.7us       10438us       33962us   30.0   5589.7us    778.5us               38.5us      1475.0       30.0         0.0%  75.580
15  10   2123.2      7464.4us       12427us       27486us   34.9   6706.6us    722.8us               35.0us      1657.7       34.9         0.0%  81.720
20  10   1871.6      8296.7us       13848us       32597us   39.6   7314.9us    946.5us               35.3us      1827.9       39.6         0.0%  84.100
30  10   1591.8      9962.8us       15289us       29843us   49.0   8969.2us    959.7us               33.9us      2157.4       49.0         0.0%  86.790
40  10   1238.7     12822.2us       23973us       41520us   58.5  11449.5us   1334.5us               38.3us      2481.9       58.5         0.0%  88.840
60  10    958.9     16577.8us       30026us       40660us   77.9  15140.0us   1402.4us               35.4us      3119.8       77.9         0.0%  91.290
80  10    786.0     20204.2us       32531us       57665us   97.5  18315.2us   1850.3us               38.7us      3749.8       97.5         0.0%  92.910
120  10    550.4     28892.0us       44286us       55067us  136.5  26148.3us   2694.8us               48.9us      4962.8      136.5         0.0%  94.370
160  10    431.3     36875.5us       52894us       72029us  175.5  33669.7us   3154.8us               51.1us      6129.3      175.5         0.0%  95.420
```

### vamana_hp_p12_l128 — rc=0, Build time: 252.862s, peakRSS=2.84GB, 2026-06-09T12:01:00+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2529.7      6229.3us       10311us       24649us   29.1   5545.4us    650.1us               33.8us      1448.4       29.1         0.0%  73.620
15  10   2039.6      7728.0us       16002us       27590us   34.2   6673.8us   1016.2us               38.1us      1641.9       34.2         0.0%  80.880
20  10   1878.6      8359.2us       13914us       28749us   39.0   7406.6us    916.5us               36.0us      1816.8       39.0         0.0%  83.310
30  10   1518.5     10432.4us       18720us       32343us   48.6   9176.2us   1221.9us               34.3us      2157.4       48.6         0.0%  86.790
40  10   1229.8     12896.6us       26674us       62176us   58.1  11536.0us   1323.4us               37.2us      2482.5       58.1         0.0%  88.630
60  10    997.3     15945.1us       26926us       39152us   77.5  14401.7us   1504.3us               39.0us      3127.1       77.5         0.0%  91.150
80  10    780.7     20363.0us       36493us       51378us   97.0  18374.5us   1949.9us               38.7us      3757.9       97.0         0.0%  92.620
120  10    554.2     28670.9us       44347us       64858us  135.9  25862.5us   2763.2us               45.2us      4982.6      135.9         0.0%  94.360
160  10    432.4     36786.5us       52287us       67195us  174.9  33355.0us   3381.1us               50.3us      6165.5      174.9         0.0%  95.490
```

### vamana_hp_p11_l83 — rc=0, Build time: 197.347s, peakRSS=2.33GB, 2026-06-09T12:04:33+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2183.1      7224.2us       14322us       34222us   30.8   6343.0us    845.7us               35.5us      1396.8       30.8         0.0%  68.410
15  10   1936.0      8149.8us       13933us       27803us   36.1   7233.8us    880.2us               35.7us      1581.1       36.1         0.0%  75.450
20  10   1805.5      8751.4us       15383us       34693us   41.1   7925.7us    789.7us               36.0us      1750.3       41.1         0.0%  78.230
30  10   1464.5     10830.9us       18700us       34657us   50.8   9687.4us   1110.2us               33.3us      2064.9       50.8         0.0%  82.150
40  10   1209.7     13073.1us       26379us       39594us   60.7  11754.6us   1283.1us               35.4us      2370.5       60.7         0.0%  84.810
60  10    948.3     16720.0us       29382us       43845us   79.7  15135.6us   1543.6us               40.8us      2941.1       79.7         0.0%  87.450
80  10    762.2     20854.4us       34539us       50334us   99.1  19006.0us   1809.9us               38.5us      3507.0       99.1         0.0%  89.330
120  10    549.1     28978.0us       45429us       59032us  138.1  26473.0us   2459.0us               45.9us      4603.7      138.1         0.0%  91.590
160  10    423.7     37475.4us       53854us       70923us  177.1  34455.8us   2975.0us               44.6us      5658.4      177.1         0.0%  92.840
```

### vamana_hp_p12_l83 — rc=0, Build time: 195.015s, peakRSS=2.33GB, 2026-06-09T12:08:05+00:00
```
           Recall@: 10
 L KNN      QPS  Mean Latency   95% Latency  99.9 Latency    IOs    IO (us)   CPU (us)   PQ Preprocess (us)  Mean Comps  Mean Hops  Cache Hit %  Recall
10  10   2432.0      6503.3us       11711us       20880us   31.2   5833.7us    634.5us               35.1us      1403.1       31.2         0.0%  67.570
15  10   1948.1      8114.9us       19546us       35918us   36.6   7224.4us    854.5us               35.9us      1592.5       36.6         0.0%  74.430
20  10   1796.0      8811.9us       14875us       32575us   41.7   7923.1us    849.4us               39.4us      1765.0       41.7         0.0%  77.910
30  10   1472.1     10746.5us       16840us       31149us   51.4   9538.6us   1169.1us               38.9us      2076.5       51.4         0.0%  81.380
40  10   1213.2     13080.3us       23903us       36806us   61.4  11738.0us   1301.8us               40.5us      2383.8       61.4         0.0%  84.200
60  10    949.2     16674.1us       27684us       40314us   80.6  15034.4us   1595.0us               44.6us      2951.5       80.6         0.0%  87.400
80  10    739.2     21479.3us       36019us       49184us  100.0  19462.3us   1969.5us               47.4us      3506.8      100.0         0.0%  89.240
120  10    545.6     29130.7us       44352us       75283us  139.0  26497.6us   2586.8us               46.3us      4592.5      139.0         0.0%  91.360
160  10    432.0     36821.9us       52724us       87786us  177.9  33576.8us   3196.8us               48.2us      5638.6      177.9         0.0%  92.660
```
