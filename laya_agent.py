"""
laya_agent.py - Laya Decision Agent Interface

Wraps the Laya model (e.g., convaiinnovations/laya or fine-tuned LoRA checkpoint) to select
the optimal action from the available options given the game context.
Includes a heuristic rule-based expert fallback for offline execution,
testing, and bootstrap data generation.
"""

import os
import re
import json
from typing import List, Optional, Tuple, Dict, Any


class LayaAgent:
    """
    Agent interface for Laya Decision Model.
    Supports:
    1. Fine-tuned LoRA DecisionModel checkpoints (trained with train_laya.py)
    2. Native Laya SDK (`laya.Agent`)
    3. Hugging Face CausalLM checkpoints
    4. Smart heuristic decision engine fallback
    """

    def __init__(
        self,
        model_name_or_path: str = "convaiinnovations/laya",
        use_heuristic: bool = False,
        device: Optional[str] = None
    ):
        self.model_name_or_path = model_name_or_path
        self.use_heuristic = use_heuristic
        self.device = device
        self.model = None
        self.tokenizer = None
        self.laya_agent = None
        self.is_lora_decision_model = False
        self.qtype_val = 0
        self.max_length = 512
        self._init_success = False

        if not self.use_heuristic:
            self._load_model()

    def _load_model(self):
        """Attempts to load via native laya package, fine-tuned LoRA wrapper, or Hugging Face."""
        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = self.model_name_or_path
        if os.path.exists(model_path):
            model_path = os.path.abspath(model_path)

        # 1. Check if this is a fine-tuned LoRA adapter directory (from train_laya.py)
        is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        if is_adapter:
            try:
                import laya
                from peft import PeftModel
                from train_laya import LayaDecisionWrapper

                meta_path = os.path.join(model_path, "laya_train_meta.json")
                base_model_name = "convaiinnovations/laya"
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        base_model_name = meta.get("base_model", "convaiinnovations/laya")
                        self.qtype_val = meta.get("qtype", 0)
                        self.max_length = meta.get("max_length", 512)

                print(f"[LayaAgent] Detected fine-tuned LoRA adapter at '{model_path}'.")
                print(f"[LayaAgent] Loading base Laya Agent '{base_model_name}' on {self.device}...")

                self.laya_agent = laya.Agent(model_id_or_path=base_model_name, device=self.device)
                wrapped = LayaDecisionWrapper(self.laya_agent.model)
                self.model = PeftModel.from_pretrained(wrapped, model_path).to(self.device)
                self.model.eval()
                self.tokenizer = self.laya_agent.tok
                self.is_lora_decision_model = True
                self._init_success = True
                print("[LayaAgent] Fine-tuned LoRA DecisionModel loaded successfully.")
                return
            except Exception as e:
                print(f"[LayaAgent] Failed loading LoRA adapter with LayaDecisionWrapper: {e}")

        # 2. Try native laya package if applicable
        if "convai" in self.model_name_or_path.lower() or self.model_name_or_path in ["laya", "convaiinnovations/laya"]:
            try:
                import laya
                print(f"[LayaAgent] Attempting to initialize native laya.Agent...")
                model_id = "convaiinnovations/laya" if self.model_name_or_path in ["convai/laya-base", "laya"] else self.model_name_or_path
                self.laya_agent = laya.Agent(model_id_or_path=model_id, device=self.device)
                self._init_success = True
                print("[LayaAgent] Native laya.Agent initialized successfully.")
                return
            except Exception as e:
                print(f"[LayaAgent] Native laya load skipped/failed: {e}")

        # 3. Fallback: Try generic Hugging Face transformers CausalLM
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[LayaAgent] Attempting Hugging Face CausalLM load from '{model_path}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(self.device)

            self.model.eval()
            self._init_success = True
            print("[LayaAgent] Hugging Face CausalLM model loaded successfully.")
            return

        except Exception as e:
            print(f"[LayaAgent] Could not load model from '{model_path}': {e}")

        print("[LayaAgent] Running in built-in heuristic decision engine mode.")
        self.use_heuristic = True
        self._init_success = False

    def select_action(self, context: str, options: List[str]) -> Tuple[str, bool]:
        """
        Given the textual context and list of options, choose the best option.
        Returns (best_option, is_beneficial).
        """
        if not options:
            return "No Action", False

        if len(options) == 1:
            return options[0], False

        if not self.use_heuristic and self._init_success:
            try:
                if self.is_lora_decision_model and self.model is not None:
                    best_action = self._lora_decision_inference(context, options)
                    return best_action, self._evaluate_beneficial(best_action)
                elif self.laya_agent is not None:
                    best_action = self._laya_native_inference(context, options)
                    return best_action, self._evaluate_beneficial(best_action)
                elif self.model is not None:
                    best_action = self._hf_model_inference(context, options)
                    return best_action, self._evaluate_beneficial(best_action)
            except Exception as e:
                print(f"[LayaAgent] Inference error: {e}. Falling back to heuristic.")

        # Fallback to heuristic policy
        return self._heuristic_choice(context, options)

    def _lora_decision_inference(self, context: str, options: List[str]) -> str:
        """Runs fast decision scoring via fine-tuned LoRA DecisionModel."""
        import torch
        from train_laya import build_prompt

        prompt, char_pos = build_prompt(context, options)
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]

        marker_pos = []
        for c in char_pos:
            hit = next((t for t, (s, e) in enumerate(offsets) if e > s and s <= c < e), None)
            if hit is None:
                # Option truncated by max_length, fallback to heuristic or first option
                return options[0]
            marker_pos.append(hit)

        input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=self.device)
        m_pos = torch.tensor([marker_pos], dtype=torch.long, device=self.device)
        m_mask = torch.ones((1, len(marker_pos)), dtype=torch.bool, device=self.device)
        qtype = torch.tensor([self.qtype_val], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                marker_pos=m_pos,
                marker_mask=m_mask,
                qtype=qtype,
            )
            logits = out.logits
            best_idx = logits[0, :len(options)].argmax().item()

        if 0 <= best_idx < len(options):
            return options[best_idx]
        return options[0]

    def _laya_native_inference(self, context: str, options: List[str]) -> str:
        """Runs decision inference using native laya.Agent."""
        criteria = {f"opt_{i}": opt for i, opt in enumerate(options)}
        questions = {
            "best_action": {
                "type": "choice",
                "instructions": "Select the optimal Klondike Solitaire move to maximize progress.",
                "criteria": criteria
            }
        }
        res = self.laya_agent.system_one(state=context, questions=questions)
        ans = res.get("answers", {}).get("best_action")
        choice_key = ans.get("choice") if isinstance(ans, dict) else ans
        if choice_key and choice_key in criteria:
            return criteria[choice_key]
        return options[0]

    def _hf_model_inference(self, context: str, options: List[str]) -> str:
        """Calculates likelihood / scores each option with CausalLM."""
        import torch

        prompt = (
            f"### Context:\n{context}\n\n"
            f"### Options:\n" + "\n".join(f"- {opt}" for opt in options) + "\n\n"
            f"### Best Action:\n"
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        best_score = float("-inf")
        best_opt = options[0]

        with torch.no_grad():
            for opt in options:
                candidate_text = prompt + opt
                candidate_ids = self.tokenizer.encode(candidate_text, return_tensors="pt").to(self.device)
                
                prompt_len = input_ids.shape[1]
                target_ids = candidate_ids.clone()
                target_ids[:, :prompt_len] = -100

                outputs = self.model(candidate_ids, labels=target_ids)
                loss = outputs.loss.item()
                score = -loss

                if score > best_score:
                    best_score = score
                    best_opt = opt

        return best_opt

    def _heuristic_choice(self, context: str, options: List[str]) -> Tuple[str, bool]:
        """Smart heuristic priority solver for Solitaire."""
        scored_options = []

        for opt in options:
            score = 0
            is_beneficial = False

            if "Foundation" in opt:
                score += 100
                is_beneficial = True
            elif "reveal" in opt.lower():
                score += 85
                is_beneficial = True
            elif "Tableau" in opt and "King" in opt and "Empty" in opt:
                score += 45
                is_beneficial = True
            elif "Waste to Tableau" in opt:
                score += 40
                is_beneficial = True
            elif "Tableau" in opt and "to" in opt:
                score += 35
                is_beneficial = True
            elif "Draw Stock" in opt:
                score += 10
                is_beneficial = False
            elif "No Action" in opt:
                score = -10
                is_beneficial = False
            else:
                score += 5

            scored_options.append((score, opt, is_beneficial))

        scored_options.sort(key=lambda x: x[0], reverse=True)
        top = scored_options[0]
        return top[1], top[2]

    def _evaluate_beneficial(self, action: str) -> bool:
        """Determines if the action was a progress/beneficial action."""
        if "Foundation" in action or "reveal" in action.lower():
            return True
        if "Waste to Tableau" in action or "to Empty" in action:
            return True
        return False
