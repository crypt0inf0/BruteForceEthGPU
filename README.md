# BruteForceEthGPU

A CUDA-accelerated tool for recovering Ethereum private keys from partial BIP39 mnemonics.

## Install Rust

1. Install Rust using rustup:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Configure current shell:
```bash
source $HOME/.cargo/env
```

3. Verify installation:
```bash
rustc --version
cargo --version
```

4. Update Rust (optional):
```bash
rustup update
```

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0 or later
- Rust 1.54 or later
- Build essentials (gcc, g++, make)

## Installation

1. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cuda nvidia-cuda-toolkit

# Fedora/RHEL
sudo dnf install -y gcc gcc-c++ make cuda
```

2. Clone and build:
```bash
git clone https://github.com/KOWHAIKIWI/BruteForceEthGPU
cd BruteForceEthGPU
cargo build --release
```

## Configuration

The tool can be configured via environment variables or a .env file:

```bash
# .env file example
KNOWN_SEED_WORDS="word1,word2,word3"  # Known seed words (comma-separated)
TARGET_ADDRESS="1234..."              # Target ETH address (no 0x prefix)
TOTAL_WORKERS=4                       # Number of GPU workers
MATCH_MODE=0                          # 0=exact, 1=prefix, 2=zero address
MATCH_PREFIX_LEN=4                    # Length for prefix matching
DANGER_TEMP=85                        # GPU temperature limit (Celsius)
```

## Usage

1. Basic usage with CLI:
```bash
./target/release/seeds \
  --known "word1,word2,word3" \
  --address "1234..." \
  --workers 4
```

2. Using start script (recommended):
```bash
./start_everything.sh
```

The script will:
- Prompt for configuration if not in .env
- Launch GPU health monitor
- Start watchdog process
- Initialize workers
- Show real-time status

## Monitor & Control

- Status files:
  - `found_seeds.txt`: Created when match found
  - `worker*.offset`: Resume points for each worker
  - `overheated.flag`: Created if GPU overheats
  - `miner*.log`: Individual miner logs
  - `restart.log`: Watchdog restart events

- Health monitoring:
  - Automatic shutdown if GPU temperature exceeds limit
  - Crash recovery via watchdog
  - Progress saved automatically

## Performance Tips

1. Adjust workers to match GPU count
2. Monitor temperatures with `nvidia-smi`
3. Use resume files for long runs
4. Consider cooling for extended operation

## Recovery Files

The program creates offset files that allow resuming interrupted searches:
- `worker0.offset`
- `worker1.offset`
- etc.

To resume, simply restart the program - it will automatically continue from saved offsets.
