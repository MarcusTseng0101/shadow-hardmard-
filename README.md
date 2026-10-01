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
  spectral.py    Hermitian 延拓 + matrix pencil + NNLS + 非線性精修 + 顯著性篩選
scripts/
  run_fundamental.py   基礎題：驗證表 + shots 收斂
  run_advanced.py      進階題：symmetry-resolved spectrum + 量測策略比較
tests/                 估計器無偏性（無限 shots 精確到 1e-10）、誤差棒誠實度
results/               圖、JSON、log
```

## 執行

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib pytest
python -m pytest -q                      # ~10 s
python -m scripts.run_fundamental        # ~1 min
python -m scripts.run_advanced           # 數分鐘；--J --delta --h --periodic 可換成助教給的模型
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
