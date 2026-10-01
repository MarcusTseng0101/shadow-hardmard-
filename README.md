# From Garbage to Spectra: Shadow-Enhanced Hadamard Testing

以 Qiskit Aer 實作並驗證 Faehrmann, Eisert, Kueng, *PRL* **135**, 150603 (2025)
（arXiv:2505.15913）的 shadow-enhanced Hadamard test，並做出三量子位元 XXZ 模型的
**symmetry-resolved spectral analyzer**。

## 核心想法（一段推導）

系統 = qubit 0..n-1，ancilla = qubit n。ancilla 做 H、控制 `U = e^{-iHt}` 之後：

```
|Φ> = ( |0>|ψ> + |1> U|ψ> ) / √2
```

傳統 Hadamard test 只量 ancilla（X 或 Y 基底，結果 s = ±1）得到 `g(t) = <ψ|U|ψ>`，
系統暫存器直接丟掉（"garbage"）。但若系統也做隨機 Pauli 量測（classical shadow，
單次估計值 ô），把 ancilla 結果條件化後的系統態 ρ̃_± 代入：

| ancilla 基底 | ρ̃₊ − ρ̃₋ | ρ̃₊ + ρ̃₋ |
|---|---|---|
| X | ½(Uρ + ρU†) | ½(ρ + UρU†) |
| Y (先 S†) | (i/2)(ρU† − Uρ) | ½(ρ + UρU†) |

所以**同一批 shots** 同時給出：

| 估計量 | 期望值 | 用途 |
|---|---|---|
| E_X[s] + i E_Y[s] | g(t) = ⟨ψ\|U\|ψ⟩ | 原本的 Hadamard 訊號 |
| E_X[s·ô] + i E_Y[s·ô] | ⟨ψ\|O U\|ψ⟩ | **ancilla–系統聯合觀測量** |
| E[ô]（忽略 ancilla） | ½(⟨O⟩_ψ + ⟨O⟩_{Uψ}) | 若 [O,H]=0 → 就是 ⟨O⟩_ψ：**能量、守恆量免費拿到** |

取 O = 對稱 sector 投影 P_m（[P_m,H]=0）：

```
g_m(t) = <ψ|P_m U(t)|ψ> = Σ_{k∈sector m} |c_k|² e^{-iE_k t},     Σ_m g_m = g
```

每個 sector 訊號只含該 sector 的能階 → 對每個 g_m 做頻譜分析，就同時得到
**能階、權重、對稱標籤**。這就是 symmetry-resolved spectroscopy。

## 程式結構

```
shadow_hadamard/
  model.py       XXZ Hamiltonian、M_z = Σ Z_i、sector 投影、輸入態、所有精確參考值
  experiment.py  電路、(biased) Pauli shadow 估計器、Aer 取樣、誤差棒
  spectral.py    Hermitian 延拓 + matrix pencil + NNLS + 非線性精修 + 顯著性篩選；damped=True 做雜訊外推
  trotter.py     閘層級的受控二階 Trotter 演化（H/S/RZ/CX）、quasi-energy 參考值
  noise.py       去極化 + 讀出雜訊模型、讀出錯誤校正（tensored inverse）
scripts/
  run_fundamental.py   基礎題：驗證表 + shots 收斂
  run_advanced.py      進階題：symmetry-resolved spectrum + 量測策略比較
  run_noise.py         延伸：Trotter 誤差與成本、雜訊掃描、雜訊緩解
tests/                 估計器無偏性（無限 shots 精確到 1e-10）、誤差棒誠實度、Trotter 電路 = 精確受控 Trotter 么正、緩解還原
results/               圖、JSON、log
```

## 執行

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib pytest
python -m pytest -q                      # ~10 s
python -m scripts.run_fundamental        # ~1 min
python -m scripts.run_advanced           # 數分鐘；--J --delta --h --periodic 可換成助教給的模型
python -m scripts.run_noise              # ~3 min；Trotter + 雜訊 + 緩解
```

## 結果

### 基礎題（t = 1.0，2×10⁵ shots，uniform shadows）

| 量 | 估計 ± 1σ | 精確 | z |
|---|---|---|---|
| Re g(t) | −0.4098 ± 0.0029 | −0.4083 | −0.55 |
| Im g(t) | −0.5109 ± 0.0027 | −0.5085 | −0.88 |
| ⟨H⟩ | −0.118 ± 0.015 | −0.112 | −0.41 |
| ⟨M_z⟩ | 0.8634 ± 0.0063 | 0.8628 | +0.09 |
| Re ⟨ψ\|H U\|ψ⟩ | 0.644 ± 0.021 | 0.647 | −0.14 |
| Im ⟨ψ\|H U\|ψ⟩ | −0.491 ± 0.021 | −0.504 | +0.60 |
| Re ⟨ψ\|M_z U\|ψ⟩ | −0.063 ± 0.009 | −0.058 | −0.55 |
| Im ⟨ψ\|M_z U\|ψ⟩ | −0.622 ± 0.009 | −0.609 | −1.39 |

RMSE 對總 shots 的 log-log 斜率 −0.47 ~ −0.53（理想 −0.5），見 `results/fundamental_convergence.png`。

### 進階題（XXZ：J=1, Δ=0.5, 開放邊界；K=40 個時間點）

自旋翻轉對稱使 S^z=±m 能階兩兩簡併：8 個被佔據能階、只有 4 個相異能量。
**只看 ancilla 的傳統 Hadamard test 只看得到 4 個合併峰且沒有標籤；
回收系統暫存器後，8 個能階的能量、權重、S^z 標籤全部重建**（`results/advanced_spectrum_*.png`）。

量測策略比較（6 次重複平均，`results/advanced_strategies.png`）：

| 策略 | 預算 | 解析出的能階比例 | 平均 \|ΔE\| | 平均 \|Δw\| | 假峰 |
|---|---|---|---|---|---|
| uniform shadows | 5×10⁴ | 0.81 | 0.0082 | 0.0090 | 0 |
| | 2×10⁵ | 0.96 | 0.0067 | 0.0050 | 0 |
| | 10⁶ | 1.00 | 0.0033 | 0.0022 | 0 |
| Z-biased (0.15/0.15/0.70) | 5×10⁴ | 1.00 | 0.0049 | 0.0041 | 0 |
| | 2×10⁵ | 1.00 | 0.0023 | 0.0020 | 0 |
| | 10⁶ | 1.00 | 0.0009 | 0.0008 | 0 |
| Z-basis only | 5×10⁴ | 1.00 | 0.0033 | 0.0019 | 0 |
| | 2×10⁵ | 1.00 | 0.0016 | 0.0014 | 0 |
| | 10⁶ | 1.00 | 0.0007 | 0.0006 | 0 |

取捨：Z-only 頻譜最準，但無法從 shadow 直接估 ⟨H⟩（XX/YY 項）；Z-biased 在頻譜幾乎一樣好的同時仍保留完整 shadow 資訊（能量、任意 Pauli 觀測量）。
能量也可由頻譜自洽得到：Σ w_k E_k 與精確 ⟨H⟩ 差約 0.005–0.02。

### 延伸：閘層級電路、雜訊與雜訊緩解（`results/trotter_error.png`、`noise_mitigation.png`、`noise_signal.png`）

**Trotter 電路。** 受控演化拆成 H / S / RZ / CX：每個 Pauli 項做「基底轉換 → parity ladder →
CRZ(2θ) = RZ-CX-RZ-CX → 還原」。測試確認整個電路**逐元素等於**受控 Trotter 么正（含相對相位）。
- 所有取樣時間用**同一個步長 δ**，所以訊號是 Trotter 步 S₂(δ) 的 Floquet quasi-energy 的**精確**指數和；
  Trotter 誤差因此能跟 shot / 閘雜訊分開。
- 同一個 bond 的 XX、YY、ZZ 互相對易而且排在一起，所以**每一步都精確守恆 M_z**：
  對稱標籤在 Trotter 化之後仍然是精確的（有測試）。
- quasi-energy 誤差 ∝ δ²（實測斜率 2.02）。δ = dt/2 時誤差 0.045，代價是 t = 13 時已經 1844 個 CX
  —— **長時間頻譜在真硬體上是被電路深度限制的**，這張成本圖就是硬體可行性的估計。

**雜訊模型。** 單量子位元閘去極化 p₁、CX 去極化 p₂、讀出翻轉 p_ro（density-matrix 模擬）。
**緩解方法**：(1) 讀出錯誤校正（REM，用全 0 / 全 1 校正電路求每個 qubit 的 2×2 混淆矩陣再取逆）；
(2) **damped fit**：閘雜訊讓訊號隨深度（∝ t）指數衰減，所以每個 sector 擬合
`g_m(t) = Σ w e^{-iEt} e^{-γ|t|}`，回報 t = 0 的權重 —— 等於把雜訊外推到零。能量不受純衰減影響。

Trotter δ = 0.283、K = 24、只量 Z、2×10⁵ shots、5 次平均（對照 = Trotter quasi-energy 頻譜）：

| 雜訊 (p₂) | 方法 | 解析比例 | \|ΔE\| | \|Δw\| | Σw（應為 1）| 假峰 |
|---|---|---|---|---|---|---|
| 無 | — | 1.00 | 0.0029 | 0.0014 | 0.998 | 0 |
| 1e-4 | raw | 1.00 | 0.0047 | 0.0127 | 0.899 | 0 |
| | REM + damped | 1.00 | 0.0047 | **0.0021** | **1.001** | 0 |
| 1e-3 | raw | 0.88 | 0.0236 | 0.0798 | 0.504 | 1.0 |
| | REM | 0.88 | 0.0236 | 0.0776 | 0.524 | 1.0 |
| | REM + damped | **1.00** | 0.0137 | **0.0044** | **1.006** | **0** |
| 3e-3 | raw | 0.68 | 0.0508 | 0.1302 | 0.245 | 2.4 |
| | REM + damped | 0.93 | 0.0476 | **0.0078** | **1.008** | 0.4 |

- **REM 單獨幾乎沒用**：讀出錯誤只占損失的一小部分，主因是閘雜訊造成的衰減。
- **damped fit 把權重誤差壓低 6–17 倍、Σw 拉回 1**，p₂ = 1e-3 時 8 個能階全部救回、假峰歸零。
- 單一時間點（基礎題的量，t = 1、164 個 CX、p₂ = 1e-3）只能做 REM：Re g 從 −0.340 → −0.346
  （參考 −0.417）—— **閘雜訊造成的偏差在單點上修不掉**；多時間點的頻譜分析才有資訊把衰減估出來。
- 未做：助教指定的模型參數（`--J --delta --h --periodic` 直接換）。

### 真硬體：IBM `ibm_kingston`（`scripts/run_hardware.py`、`results/hardware.json`）

job `daut44qhcrkc73e09m3g`，Trotter δ = 0.25，uniform shadows，每個時間點 X / Y 各 2×10⁴ shots，
共 110 個電路、9.6×10⁴ shots（含讀出校正）。實體 qubit [52, 53, 39, 54]。
body 每個 (t, 基底) 只 transpile 一次，再用原生 rz / sx 接上隨機量測基底 ——
所有隨機設定讀同一組實體 qubit，讀出校正才對得上。參考值 = 無雜訊 Trotter 的精確值。

| t | 量 | 參考 | 原始 | 讀出校正（REM） |
|---|---|---|---|---|
| 0.5（80 個雙量子位元閘）| Re g | 0.361 | 0.243 ± 0.007 | **0.339 ± 0.007** |
| | Im g | −0.082 | −0.055 ± 0.007 | −0.013 ± 0.007 |
| | ⟨H⟩（免費）| −0.112 | −0.055 ± 0.033 | −0.074 ± 0.033 |
| | ⟨M_z⟩（免費）| 0.863 | 0.906 ± 0.014 | 0.890 ± 0.014 |
| | Re ⟨ψ\|M_z U\|ψ⟩（聯合）| 0.479 | 0.360 ± 0.021 | **0.471 ± 0.021** |
| 1.0（156 個雙量子位元閘）| Re g | −0.417 | −0.143 ± 0.007 | −0.117 ± 0.007 |
| | Im g | −0.496 | −0.392 ± 0.007 | −0.410 ± 0.006 |
| | ⟨H⟩（免費）| −0.112 | −0.011 ± 0.032 | −0.013 ± 0.032 |
| | ⟨M_z⟩（免費）| 0.863 | 0.871 ± 0.014 | **0.859 ± 0.014** |
| | Re ⟨ψ\|M_z U\|ψ⟩（聯合）| −0.057 | −0.028 ± 0.021 | 0.011 ± 0.021 |

- **t = 0.5：REM 之後 Re g 與聯合量都回到參考值附近**（0.339 vs 0.361、0.471 vs 0.479）。
  這一點上主要的損失是讀出錯誤。
- **t = 1.0：Hadamard 訊號本身壞掉了**。|g| 從 0.648 掉到約 0.43，**而且相位也轉了**
  （Re g 偏得比 Im g 多）—— 那不是單純的去極化衰減，是**同調（coherent）誤差**，REM 修不掉。
- **⭐ 免費拿到的守恆量反而最穩**：⟨M_z⟩ 在 t = 1.0 是 0.859 ± 0.014（參考 0.863）。
  因為它只用系統暫存器的邊際分布、不需要 ancilla 的相干性，而 M_z 在每個 Trotter 步都守恆。
  **這正是論文「把 garbage 拿來用」的實際價值：即使 Hadamard 訊號被雜訊吃掉，回收的暫存器還是給出可靠的物理量。**
- ⟨H⟩ 的誤差棒大（uniform shadows 估 XX/YY 項變異數大），t = 1.0 偏 3σ —— 能量項含 X/Y 基底，
  比 M_z 對閘雜訊敏感。

## 設計重點

- **精確受控演化**：`|0⟩⟨0|⊗1 + |1⟩⟨1|⊗e^{-iHt}` 以 16×16 `UnitaryGate` 實作（保留相位，控制版本才正確）。
  換成 Trotter 電路只需改 `controlled_evolution`。
- **Biased classical shadows**：每個 qubit 以機率 (p_X, p_Y, p_Z) 選基底，Pauli 字串估計器
  `∏_{i∈supp} [b_i = P_i] s_i / p_{P_i}`，uniform (1/3) 即 Huang–Kueng–Preskill 原版。
  Sector 投影全是 Z 字串，偏向 Z 可大幅降低變異數；只量 Z 則最適合頻譜但無法估 XX/YY（程式會回報 NaN 而不是給錯的數）。
- **無偏性測試**：`mode="exact"` 用精確機率取代取樣（等於無限 shots），所有估計量與解析值差 < 1e-10。
- **誤差棒誠實度測試**：40 次獨立重複的 z² 平均 ≈ 1。
- **頻譜分析**：
  1. `g_m(-t) = conj g_m(t)` → 資料長度免費加倍；
  2. Nyquist：`|E| ≤ Σ|係數|`，取 `dt = 0.9π / Σ|係數|` 避免混疊；
  3. matrix pencil 的 model order = sector 維度（由對稱性得知的上界）；
  4. 權重必為非負實數 → NNLS，再對 (E, w) 聯合非線性最小平方；
  5. 權重小於 3σ（σ 由擬合殘差估）的峰視為雜訊丟棄。
