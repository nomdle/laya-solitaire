# 📄 [TASK.md](http://TASK.md) — Laya 기반 카드 게임 에이전트 &amp; 파인튜닝 로깅 시스템 구축

## 1. 개요 (Overview)

Python Pygame 기반의 카드 게임(솔리테어 변형 또는 간단한 카드 게임) 환경을 구축하고, Laya 결정 모델(Decision Model)과의 인터페이스, 실시간 시각화, 플레이 로그 자동 수집, 그리고 Laya 모델 파인튜닝 파이프라인을 통합 구현합니다.

---

## 2. 주요 구성 요소 (Architecture)

1. \*\*`game_[engine.py](http://engine.py)` (Pygame 게임 환경)\*\*

   \* 카드 게임 로직 및 UI 화면 렌더링 (30~60 FPS).

   \* 사람 플레이어 및 Laya 에이전트 제어 지원.

   \* 매 턴마다 현재 게임 상태를 512 토큰 이내의 텍스트(`context`)로 요약하는 함수 제공.

   \* 유효한 가능 행동 리스트(`options`) 생성 기능.

2. \*\*`laya_[agent.py](http://agent.py)` (Laya Decision Agent Interface)\*\*

   \* `convai/laya-base` (또는 지정된 체크포인트) 로드 및 추론.

   \* `context`와 `options`를 입력받아 최적 행동(`best_option`) 반환.

3. \*\*`data_[logger.py](http://logger.py)` (파인튜닝 데이터 수집기)\*\*

   \* 에이전트 또는 사람 플레이어의 매 턴 행동 데이터를 JSONL 규격으로 저장.

   \* \*\*데이터 저장 포맷 (JSONL):\*\*

     \`\`\`json

     {

       "context": "Stock Remaining: 10\\nWaste Top: Heart 7\\nFoundations: \[Spade A\]\\nTableau Tops: \[Diamond 8, Club 9\]",

       "options": \["Move Heart 7 to Club 9", "Draw Stock", "No Action"\],

       "target": "Move Heart 7 to Club 9"

     }

     \`\`\`

4. \*\*`train_[laya.py](http://laya.py)` (LoRA 파인튜닝 스크립트)\*\*

   \* 수집된 `laya_solitaire_dataset.jsonl`을 기반으로 Hugging Face TRL / LoRA 학습 수행.

   \* 학습 완료 후 파인튜닝된 체크포인트 내보내기.

---

## 3. 세부 구현 요구사항 (Requirements)

### A. Pygame 렌더링 &amp; 시각화

\* 실행 시 UI 창이 뜨며 카드가 배치되고 이동하는 모습을 눈으로 실시간 확인 가능해야 함.

\* 봇 플레이 모드 시, 매 턴 0.3~0.5초의 딜레이를 주어 사람이 행동 과정을 쉽게 관찰할 수 있도록 설정.

### B. State Representation (Context 요약)

\* Laya의 기본 입력 제한(512 토큰)을 초과하지 않도록 key-value 형태의 간결한 Text Format 정의.

### C. Logging System

\* 성공적인 유효 행동만 기록하거나, 게임 승리/이득 상황의 로그만 필터링하여 저장할 수 있는 플래그 제공.

---

## 4. 실행 디렉토리 구조 (Directory Structure)

\`\`\`text

.

├── [TASK.md](http://TASK.md)

├── requirements.txt         # pygame, torch, transformers, trl, peft 등

├── game\_[engine.py](http://engine.py)          # Pygame 게임 및 State Extractor

├── laya\_[agent.py](http://agent.py)           # Laya Model Wrapper

├── data\_[logger.py](http://logger.py)          # JSONL Logging Utility

├── train\_[laya.py](http://laya.py)           # Fine-tuning Script

└── dataset/

    └── laya\_dataset.jsonl  # 수집되는 학습 데이터