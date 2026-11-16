# 🃏 Laya Solitaire AI & Fine-Tuning Pipeline

[English](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md)

---

A Pygame-based Klondike Solitaire card game environment integrated with the **Laya Decision Model**, featuring real-time decision monitor visualization, automated gameplay log collection, LoRA fine-tuning, and a before/after benchmark evaluation pipeline.

---

## 📋 Table of Contents
1. [Prerequisites](#-prerequisites)
2. [Installation & Setup](#-installation--setup)
3. [Quick Start](#-quick-start)
4. [End-to-End Pipeline Replication](#-end-to-end-pipeline-replication)
   - [Step 1: High-Quality Data Collection (Headless)](#step-1-high-quality-data-collection-headless)
   - [Step 2: Pipeline Pre-check (Dry Run)](#step-2-pipeline-pre-check-dry-run)
   - [Step 3: LoRA Fine-Tuning](#step-3-lora-fine-tuning)
   - [Step 4: Benchmark Evaluation (Before vs. After)](#step-4-benchmark-evaluation-before-vs-after)
   - [Step 5: Run GUI with Fine-Tuned Model](#step-5-run-gui-with-fine-tuned-model)
5. [Project Structure](#-project-structure)
6. [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 💻 Prerequisites

* **Python**: 3.10 or 3.11 recommended
* **Operating System**: Windows, macOS, or Linux
* **Hardware**:
  * **CPU**: Capable of running the Pygame GUI, heuristic rule solver, and basic inference.
  * **GPU (Recommended)**: NVIDIA GPU with CUDA support (VRAM >= 6GB) for LoRA fine-tuning and fast neural inference.

---

## ⚙️ Installation & Setup

Follow these steps to set up the project on a clean environment or machine:

### 1. Clone the Repository
```bash
git clone https://github.com/nomdle/laya-solitaire.git
cd laya-solitaire
```

### 2. Create and Activate a Virtual Environment
Using a virtual environment is strongly recommended to avoid package conflicts:

* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **macOS / Linux (Bash/Zsh):**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
Install all required libraries (`pygame`, `transformers`, `peft`, `trl`, `laya`, etc.):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for CUDA / GPU Users:**  
> If you have an NVIDIA GPU, make sure you have the CUDA-enabled PyTorch installed. Refer to the [Official PyTorch Getting Started Guide](https://pytorch.org/get-started/locally/) if needed:  
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

---

## 🚀 Quick Start

You can test the game environment immediately after installation.

### 1. Rule-Based (Heuristic) Autonomous Play (GUI)
Runs the game using the built-in expert rule engine:
```bash
python game_engine.py
```

* **Keyboard Controls:**
  * `SPACE`: Toggle Bot Play Pause / Resume
  * `O`: Toggle Laya Decision Monitor Overlay Panel (Show / Hide)
  * `N`: Step forward 1 turn manually
  * `R`: Reset and deal a new game
  * `L`: Toggle live decision logging on/off
  * `ESC`: Quit game
  * **Mouse**: Click on the Stock pile (top-left) to draw cards manually.

### 2. Vanilla Laya Model Play (GUI)
Runs the game using the pre-trained foundational decision model (`convaiinnovations/laya` from Hugging Face):
```bash
python game_engine.py --model convaiinnovations/laya
```

### 3. Human Play Mode
Play Klondike Solitaire manually:
```bash
python game_engine.py --human
```

---

## 🔁 End-to-End Pipeline Replication

Since heavy model checkpoints and large dataset files are excluded from Git version control, you can replicate the entire workflow from data collection to fine-tuning and benchmark evaluation using the following steps:

```text
[Step 1: Data Collection] ➔ [Step 2: Dry Run] ➔ [Step 3: LoRA Training] ➔ [Step 4: Benchmark] ➔ [Step 5: Run Fine-Tuned Bot]
```

### Step 1: High-Quality Data Collection (Headless)
Runs automated games headlessly in the background. It automatically detects and discards stalled or looping games, saving only verified progressive episodes into `dataset/laya_dataset.jsonl`:
```bash
# Collect 50 verified episodes (takes a few minutes)
python game_engine.py --headless --collect 50
```

### Step 2: Pipeline Pre-check (Dry Run)
Validates dataset loading, token length distributions, option shuffling (to eliminate position bias), and LoRA linear module discovery:
```bash
python train_laya.py --dry_run
```

### Step 3: LoRA Fine-Tuning
Fine-tunes the Laya decision model with Low-Rank Adaptation (LoRA) using the collected dataset:
```bash
python train_laya.py \
    --model_name convaiinnovations/laya \
    --dataset_path dataset/laya_dataset.jsonl \
    --output_dir ./checkpoint-laya-solitaire \
    --epochs 3 \
    --batch_size 2
```
* Once training finishes, adapter weights (`adapter_model.safetensors`) and metadata will be saved into `./checkpoint-laya-solitaire/`.

### Step 4: Benchmark Evaluation (Before vs. After)
Evaluates the baseline heuristic agent against the fine-tuned LoRA checkpoint across N episodes and prints a side-by-side performance comparison table:
```bash
# Compare across 30 episodes
python evaluate.py --episodes 30 --checkpoint ./checkpoint-laya-solitaire
```

* **Sample Output:**
  ```text
  =================================================================
   [BENCHMARK] LAYA SOLITAIRE PERFORMANCE COMPARISON
  =================================================================
  Metric                    | Before (Heuristic) | After (Fine-Tuned)
  -----------------------------------------------------------------
  Clear Rate (Win Rate)     | 0.0%               | 0.0%              
  Avg Clear Time            | N/A                | N/A               
  Avg Clear Moves           | N/A                | N/A               
  Avg Foundations Progress  | 3.7 / 52 cards     | 5.2 / 52 cards    
  =================================================================
  ```

### Step 5: Run GUI with Fine-Tuned Model
Launch the Pygame interface to watch your fine-tuned model play in real time:
```bash
python game_engine.py --model ./checkpoint-laya-solitaire
```

---

## 📁 Project Structure

```text
solitaire-bot/
├── .gitignore               # Ignores large checkpoints (*.safetensors, *.pt) and dataset files
├── requirements.txt         # Required Python packages
├── README.md                # English main documentation
├── README.ko.md             # Korean documentation
├── README.ja.md             # Japanese documentation
├── task.md                  # Project specification document
├── game_engine.py           # Pygame 60FPS UI, Klondike rules engine, and action generator
├── laya_agent.py            # Laya SDK wrapper, LoRA inference, and heuristic fallback
├── data_logger.py           # 2-stage quality-filtered episode JSONL data collector
├── train_laya.py            # LoRA fine-tuning script with position-bias option shuffling
├── evaluate.py              # Comparative benchmark evaluation script
└── dataset/
    └── .gitkeep             # Preserves empty dataset directory in Git
```

---

## ❓ Troubleshooting & FAQ

### Q1. `RuntimeWarning: laya: this checkpoint ships invalid temperatures...`
* **Explanation**: This is an informational warning emitted by the `laya` package indicating that certain temperature coefficients in the remote checkpoint were clipped to safe defaults (`0.5`). It does **not** affect inference or gameplay functionality.

### Q2. `UnicodeEncodeError` on Windows Terminal
* All scripts include UTF-8 stdout reconfiguration guards (`sys.stdout.reconfigure(encoding="utf-8")`). If your terminal still exhibits character issues, switch the code page to UTF-8:
  ```powershell
  chcp 65001
  ```

### Q3. Can I run without an NVIDIA GPU?
* Yes. The game engine, heuristic bot, data collection, and basic inference can run on CPU. For LoRA fine-tuning (`train_laya.py`), a GPU is recommended for reasonable training speeds, but it will automatically fall back to CPU if no CUDA device is detected.
