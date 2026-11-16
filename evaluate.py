# evaluate.py
import time
import os
import sys
import argparse
from game_engine import SolitaireEngine
from laya_agent import LayaAgent

# Windows 콘솔 인코딩 에러 방지
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def evaluate_agent(model_path: str, use_heuristic: bool, num_episodes: int = 30, max_steps: int = 150):
    # 로컬 경로 절대경로화
    target_path = os.path.abspath(model_path) if os.path.exists(model_path) else model_path
    
    agent = LayaAgent(model_name_or_path=target_path, use_heuristic=use_heuristic)
    engine = SolitaireEngine()
    
    wins = 0
    clear_times = []
    total_moves = []
    foundation_scores = []

    print(f"\n[Evaluate] Testing '{target_path}' (Heuristic={use_heuristic}) for {num_episodes} episodes...")

    for ep in range(1, num_episodes + 1):
        engine.reset()
        ep_start_time = time.time()

        for step in range(max_steps):
            if engine.is_won():
                break
            if engine.is_loop_detected() or engine.is_stalled(25):
                break

            context = engine.get_context_summary()
            valid_actions = engine.get_valid_actions()
            options = [act.to_option_string() for act in valid_actions]

            if not options or (len(options) == 1 and options[0] == "No Action"):
                break

            best_option, _ = agent.select_action(context, options)
            executed = engine.execute_option_string(best_option)
            if not executed:
                break

        # 점수 기록
        foundation_cards = engine.get_total_foundation_cards()
        foundation_scores.append(foundation_cards)

        if engine.is_won():
            wins += 1
            duration = round(time.time() - ep_start_time, 2)
            clear_times.append(duration)
            total_moves.append(engine.move_count)
            print(f"  - Game #{ep}: CLEAR [WIN] | Time: {duration}s | Moves: {engine.move_count}")
        else:
            print(f"  - Game #{ep}: END (Stall/Loop) | Foundations: {foundation_cards}/52 | Moves: {engine.move_count}")

    avg_time = f"{round(sum(clear_times) / len(clear_times), 2)}s" if clear_times else "N/A"
    avg_moves = f"{round(sum(total_moves) / len(total_moves), 1)}턴" if total_moves else "N/A"
    avg_foundations = round(sum(foundation_scores) / len(foundation_scores), 2)
    win_rate = round((wins / num_episodes) * 100, 1)

    return {
        "win_rate": f"{win_rate}%",
        "avg_clear_time": avg_time,
        "avg_clear_moves": avg_moves,
        "avg_foundations": f"{avg_foundations}장 / 52장"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--checkpoint", type=str, default="./checkpoint-laya-solitaire")
    args = parser.parse_args()

    # 1. 학습 전 (Heuristic Rule Engine)
    before_stats = evaluate_agent(model_path="convaiinnovations/laya", use_heuristic=True, num_episodes=args.episodes)

    # 2. 학습 후 (Fine-Tuned Checkpoint)
    after_stats = evaluate_agent(model_path=args.checkpoint, use_heuristic=False, num_episodes=args.episodes)

    print("\n" + "="*65)
    print(" [BENCHMARK] LAYA SOLITAIRE PERFORMANCE COMPARISON")
    print("="*65)
    print(f"{'Metric':<25} | {'Before (Heuristic)':<18} | {'After (Fine-Tuned)':<18}")
    print("-" * 65)
    print(f"{'Clear Rate (승률)':<23} | {before_stats['win_rate']:<18} | {after_stats['win_rate']:<18}")
    print(f"{'Avg Clear Time (클리어 시간)':<20} | {before_stats['avg_clear_time']:<18} | {after_stats['avg_clear_time']:<18}")
    print(f"{'Avg Clear Moves (클리어 턴수)':<19} | {before_stats['avg_clear_moves']:<18} | {after_stats['avg_clear_moves']:<18}")
    print(f"{'Avg Foundations (평균 카드진전)':<17} | {before_stats['avg_foundations']:<18} | {after_stats['avg_foundations']:<18}")
    print("="*65)
