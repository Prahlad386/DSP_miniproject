# Automatic Musical Pitch Detection and Note Identification

A research-oriented Digital Signal Processing (DSP) framework for fundamental frequency ($f_0$) tracking and note identification using Short-Time Fourier Transform (STFT), autocorrelation, and adaptive time-frequency harmonic analysis.

---

## 1. Project Overview & Research Objective

Standard pitch detection methods often suffer from severe failure modes:
* **Time-domain methods (e.g., YIN):** Susceptible to sub-harmonic octave halving ($2T_0, 3T_0$) and signal noise.
* **Frequency-domain methods (e.g., HPS):** Limited by finite window spectral leakage, low-frequency mainlobe overlap, and inharmonicity ($f_k \neq k f_0$).

This project proposes an **Adaptive Time-Frequency Energy-Weighted Fusion (ATF-EWF)** pitch estimation architecture with dynamic phase/group delay refinement and octave clamping. The system is benchmarked against classic published baselines across controlled synthetic signal sweeps and real-world audio datasets.

---

## 2. Directory Structure

```text
dsp-pitch-detection/
├── README.md                 # Project documentation & execution guide
├── config.py                 # Centralized global DSP parameters
├── requirements.txt          # Environment dependencies
├── main.tex                  # IEEE journal technical report source
├── references.bib            # BibTeX citation library
├── data/
│   ├── raw/                  # Downloaded audio stems (NSynth / Bach10)
│   ├── interim/              # Unzipped metadata & audio chunks
│   └── processed/            # Subject-independent train/val/test split manifests
├── docs/
│   └── references/           # PDF copies of foundational literature papers
├── results/
│   ├── figures/              # Generated DSP plots (spectra, CMNDF, group delay)
│   └── metrics/              # Exported evaluation metrics (CSV files)
└── src/
    ├── utils/                # Frequency lookup, MIDI utilities & data splitters
    ├── models/               # YIN, HPS, and proposed ATF-EWF implementations
    └── experiments/          # Synthetic signal engine & evaluation pipelines
```
## 3. Completed Infrastructure & Setup (Person 4 Hand-Off)

The initial repository framework, configuration parameters, and core utilities have been fully built and verified.

### 3.1 Environment Specification

**File:** `requirements.txt`

Dependencies are specified for reproducible installation and include:

- `numpy`
- `scipy`
- `matplotlib`
- `librosa`
- `pandas`
- `soundfile`
- `tqdm`

### 3.2 IEEE Manuscript Template

**Files:** `main.tex`, `references.bib`

A pre-structured LaTeX manuscript template has been established in accordance with the course report guidelines.

### 3.3 Centralized Global DSP Configuration

**File:** `config.py`

All algorithms use the same centralized analysis parameters to ensure consistent experimental conditions.

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Sampling Rate | \(f_s\) | 44.1 kHz | Audio sampling frequency |
| Frame Length | \(N_w\) | 2048 samples | Approximately 46.4 ms per frame |
| Hop Length | \(N_h\) | 512 samples | 75% overlap between consecutive frames |
| FFT Size | \(N_{\text{FFT}}\) | 4096 points | Zero-padded FFT for spectral analysis |
| Lower Pitch Bound | — | A0 (27.5 Hz) | Lowest supported note |
| Upper Pitch Bound | — | C8 (4186.0 Hz) | Highest supported note |

### 3.4 Pitch and MIDI Conversion Utilities

**File:** `src/utils/pitch_utils.py`

The following utility functions have been implemented:

- **`hz_to_midi(f0)`** — Converts a continuous frequency in Hz into an integer MIDI note index and its sub-semitone cent deviation, \(\Delta c \in [-50, +50]\).
- **`midi_to_note_name(midi_note)`** — Converts an integer MIDI note number into Scientific Pitch Notation. For example, MIDI note `69` corresponds to `A4`.

### 3.5 Data Ingestion Utility

**File:** `src/utils/fetch_data.py`

Provides directory verification and automated audio file loading for the dataset ingestion pipeline.

### 3.6 Literature Collection

**Directory:** `docs/references/`

The primary literature references for the baseline algorithms have been archived:

- **YIN:** de Cheveigné and Kawahara (2002).
- **Harmonic Product Spectrum (HPS):** Schroeder (1968).

---

## 4. Environment Setup & Execution Instructions

### 4.1 Installation

Clone the repository and install the required Python dependencies.

```bash
git clone <repository-url>
cd dsp-pitch-detection
pip install -r requirements.txt
```

### 4.2 Running Utility Checks

Verify pitch conversion, MIDI/cent calculations, and raw dataset ingestion.

**Test pitch conversion utilities:**

```bash
python3 -m src.utils.pitch_utils
```

**Verify raw data ingestion:**

```bash
python3 -m src.utils.fetch_data
```

---

## 5. Next Steps & Development Roadmap (Persons 1–3)

With the initial infrastructure established, development responsibilities transition to Persons 1, 2, and 3.

### 5.1 Development Allocation Matrix

| Person | Responsibility | Primary Focus |
|---|---|---|
| Person 1 | Lead DSP Architect & Proposed Hybrid Method | Hybrid algorithm and DSP theory |
| Person 2 | Synthetic Audio Simulator & Baseline 1 | Synthetic signal generation and YIN |
| Person 3 | Dataset Partitioning, Baseline 2 & Evaluation | Dataset splits, HPS, and benchmarking |

---

### 5.2 Person 1 — Lead DSP Architect & Proposed Hybrid Method

**File to create:** `src/models/atf_ewf.py`

#### Task 1: Joint Salience Function Formulation

Implement the fusion function:

\[
S_{\text{fused}}(\tau)
=
\alpha(t)S_{\text{time}}(\tau)
+
\left(1-\alpha(t)\right)S_{\text{freq}}(\tau)
\]

where \(\alpha(t)\) is computed from the local frame's spectral flatness.

#### Task 2: Phase/Group Delay Refinement Module

Implement instantaneous phase derivative correction using the group delay expression:

\[
\tau_g(\omega)
=
-\frac{d}{d\omega}
\arg\left\{X(e^{j\omega})\right\}
\]

The objective is to improve sub-bin frequency estimation accuracy.

#### Task 3: Inter-Frame Continuity

Develop a temporal dynamic programming or octave-clamping algorithm to suppress frame-to-frame octave jumps and improve pitch trajectory continuity.

#### Task 4: Computational Complexity Analysis

Derive the theoretical floating-point operation (FLOP) count per frame, comparing relevant computational costs, including:

- FFT-based processing: \(O(N\log N)\).
- Direct time-domain processing: \(O(N^2)\), where applicable.

Document the derivation and write **Section IV — Proposed Algorithm** in `main.tex`.

---

### 5.3 Person 2 — Synthetic Audio Simulator & Baseline 1 (YIN) Lead

**Files to create:**

- `src/models/yin.py`
- `src/experiments/synthetic_generator.py`

#### Task 1: Synthetic Signal Engine

Develop a synthetic audio generator for complex harmonic stacks with controllable signal characteristics.

**Additive noise**

- Additive White Gaussian Noise (AWGN).
- Pink noise.
- Signal-to-noise ratio (SNR) ranging from \(-10\) dB to \(+30\) dB.

**Pitch dynamics**

- Vibrato: 5 Hz modulation rate and 50-cent depth.
- Amplitude tremolo.

**Physical string stiffness**

Model inharmonic partials using:

\[
f_k = kf_0\sqrt{1 + Bk^2}
\]

where \(B \in [0, 0.001]\) is the inharmonicity coefficient, \(f_0\) is the fundamental frequency, and \(k\) is the partial index.

#### Task 2: Baseline 1 — YIN Implementation

Reimplement the complete YIN fundamental-frequency estimation algorithm in `src/models/yin.py`, including:

1. Difference function.
2. Cumulative Mean Normalized Difference Function (CMNDF), \(d'_t(\tau)\).
3. Absolute thresholding, with a nominal threshold of \(\theta \approx 0.1\).
4. Parabolic interpolation for refined period estimation.

#### Task 3: Controlled Noise Sweeps & Diagnostic Figures

Run automated Monte Carlo experiments across the specified SNR range.

Generate diagnostic figures and save them in `results/figures/`, including:

- CMNDF curves.
- Group delay profiles.
- Pole-zero maps.

Write **Section V — Synthetic Experiments** in `main.tex`.

---

### 5.4 Person 3 — Dataset Partitioning, Baseline 2 (HPS) & Evaluation Lead

**Files to create:**

- `src/utils/dataset_splitter.py`
- `src/models/hps.py`
- `src/experiments/evaluate.py`

#### Task 1: Subject-Independent Dataset Partitioning

Implement `src/utils/dataset_splitter.py` to ingest audio files from `data/raw/` and generate the following manifests in `data/processed/`:

- `train_val_manifest.csv`
- `test_manifest.csv`

**Requirement:** Ensure zero instrument/subject overlap between the training/validation and test splits.

#### Task 2: Baseline 2 — Harmonic Product Spectrum (HPS)

Reimplement the HPS fundamental-frequency estimation baseline in `src/models/hps.py` using downsampled Short-Time Fourier Transform (STFT) spectral products.

#### Task 3: Automated Evaluation Pipeline

Implement `src/experiments/evaluate.py` to calculate the following metrics.

| Metric | Symbol | Definition |
|---|---|---|
| Gross Pitch Error | GPE | Percentage of evaluated frames with pitch error greater than 50 cents |
| Fine Cent Error | \(\mathrm{RMSE}_{\text{cents}}\) | Root mean square pitch error in cents on correctly voiced frames |
| Octave Error Rate | \(E_{\text{octave}}\) | Percentage of predictions at \(0.5f_0^*\) or \(2.0f_0^*\) |
| Voicing Decision Error | \(V_{\text{err}}\) | False-positive and false-negative errors in voiced/unvoiced frame detection |

Here, \(f_0^*\) denotes the reference fundamental frequency.

#### Task 4: Dataset Benchmarking & Manuscript Writing

Run all three models over the held-out test split:

- YIN.
- HPS.
- ATF-EWF (proposed hybrid method).

Log evaluation results in `results/metrics/`, perform failure-mode analysis, and write the following sections in `main.tex`:

- **Section III:** Dataset and Experimental Methodology.
- **Section VI:** Results and Discussion.
- **Section VII:** Conclusion and Future Work.

---

## 6. Expected Development Outputs

| Person | Expected Deliverables |
|---|---|
| Person 1 | `src/models/atf_ewf.py`, complexity analysis, Section IV |
| Person 2 | `src/models/yin.py`, `src/experiments/synthetic_generator.py`, diagnostic figures, Section V |
| Person 3 | `src/utils/dataset_splitter.py`, `src/models/hps.py`, `src/experiments/evaluate.py`, test metrics, Sections III, VI, and VII |
| Person 4 | Repository framework, centralized configuration, pitch utilities, data ingestion, manuscript shell, and literature archive |

### Final Integration Checklist

- [ ] All algorithms use the centralized parameters defined in `config.py`.
- [ ] YIN and HPS baselines are implemented and tested.
- [ ] The proposed ATF-EWF hybrid algorithm is implemented and tested.
- [ ] Synthetic signals cover the specified noise, pitch dynamics, and inharmonicity conditions.
- [ ] Dataset splits prevent instrument/subject leakage.
- [ ] All evaluation metrics are calculated consistently.
- [ ] Figures are saved in `results/figures/`.
- [ ] Evaluation outputs are saved in `results/metrics/`.
- [ ] All assigned manuscript sections are completed.
- [ ] The complete pipeline runs reproducibly from the repository root.

