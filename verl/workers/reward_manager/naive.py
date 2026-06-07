# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
import time

import torch
import json
from verl import DataProto
from verl.utils.reward_score import _default_compute_score

# Directory for saving successful trajectories for future SFT
SFT_SAVE_DIR = os.environ.get("IGPO_SFT_SAVE_DIR", "/root/autodl-tmp/output/sft_trajectories")
os.makedirs(SFT_SAVE_DIR, exist_ok=True)

# Debug: log all trajectories for diagnosis
DEBUG_TRAJ_DIR = os.environ.get("IGPO_DEBUG_TRAJ_DIR", "/root/autodl-tmp/output/debug_trajectories")
os.makedirs(DEBUG_TRAJ_DIR, exist_ok=True)


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False, val_type='f1', info_gain_rewards=None, is_validation=False):
        """We will expand this function gradually based on the available datasets"""
        data_str = str(data)
        if is_validation:
            f1_scores = []
            em_scores = []
            noformatf1_scores = []
        # Track successful samples for SFT saving
        sft_samples = []
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # info_gain_reward - add null check
            info_gain_reward = info_gain_rewards[i] if info_gain_rewards is not None else []

            score = self.compute_score(
                data_source=data_source,
                prompt_str = prompt_str,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                val_type=val_type,
                info_gain_reward=info_gain_reward,
                tokenizer=self.tokenizer,
                is_validation=is_validation,
            )

            if is_validation:
                f1_scores.append(score['f1'])
                em_scores.append(score['em'])
                noformatf1_scores.append(score['noformatf1'])
                reward_tensor[i, :valid_response_length] = torch.tensor(score['scores'])
                # Save successful validation samples for SFT
                f1_val = score['f1']
            else:
                reward_tensor[i, :valid_response_length] = torch.tensor(score)
                # Determine f1 from token-level scores (last non-zero value is the f1 reward)
                f1_val = score[-1] if isinstance(score, list) and score else 0.0

            # Debug: save all trajectories for diagnosis
            if not is_validation and not hasattr(self, '_debug_step'):
                self._debug_step = 0
            if not is_validation:
                if not hasattr(self, '_debug_sample_idx'):
                    self._debug_sample_idx = 0
                    self._debug_step_file = os.path.join(
                        DEBUG_TRAJ_DIR, f"step_{self._debug_step}.jsonl")
                    self._debug_fh = open(self._debug_step_file, 'w')
                if self._debug_sample_idx >= len(data):
                    # New step: close old file, open new one
                    self._debug_fh.close()
                    self._debug_step += 1
                    self._debug_sample_idx = 0
                    self._debug_step_file = os.path.join(
                        DEBUG_TRAJ_DIR, f"step_{self._debug_step}.jsonl")
                    self._debug_fh = open(self._debug_step_file, 'w')
                debug_record = {
                    "sample_idx": i,
                    "f1": f1_val,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "info_gain_reward": info_gain_reward,
                    "response_preview": response_str[:500],
                    "response_len": len(response_str),
                    "has_answer_tag": "<answer>" in response_str,
                    "has_think_tag": "" in response_str,
                }
                self._debug_fh.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
                self._debug_fh.flush()
                self._debug_sample_idx += 1
                # On last sample of step, also save full trajectory
                if i == len(data) - 1:
                    full_record = {
                        "sample_idx": i,
                        "f1": f1_val,
                        "ground_truth": ground_truth,
                        "data_source": data_source,
                        "info_gain_reward": info_gain_reward,
                        "prompt": prompt_str,
                        "response": response_str,
                    }
                    full_file = os.path.join(
                        DEBUG_TRAJ_DIR, f"step_{self._debug_step}_full_last.json")
                    with open(full_file, 'w') as f:
                        json.dump(full_record, f, ensure_ascii=False, indent=2)

            # Per-step statistics: answer quality breakdown
            answer_text = ""
            if "<answer>" in response_str and "</answer>" in response_str:
                answer_text = response_str.split("<answer>")[-1].split("</answer>")[0].strip()
            if not hasattr(self, '_step_stats'):
                self._step_stats = {'total': 0, 'empty_answer': 0, 'placeholder': 0,
                                     'has_answer': 0, 'positive_f1': 0, 'format_error': 0}
            self._step_stats['total'] += 1
            if not answer_text:
                self._step_stats['empty_answer'] += 1
            elif 'YOUR ANSWER' in answer_text or 'YOUR THINKING' in answer_text:
                self._step_stats['placeholder'] += 1
            else:
                self._step_stats['has_answer'] += 1
            if f1_val > 0:
                self._step_stats['positive_f1'] += 1
            elif f1_val == -2.0:
                self._step_stats['format_error'] += 1

            # Collect successful samples (F1 > 0 and format correct, i.e. not -2.0)
            if f1_val > 0 and f1_val != -2.0:
                sft_samples.append({
                    "prompt": prompt_str,
                    "response": response_str,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "f1": f1_val,
                    "turns": len(info_gain_reward) + 1 if info_gain_reward else 1,
                    "info_gain_reward": info_gain_reward if info_gain_reward else [],
                })

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and val_type == 'f1':
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[data_source]", data_source, "[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    # Validation mode: score is dict
                    for key, value in score.items():
                        if key != 'scores':  # Skip verbose token-level scores
                            print(f"[{key}]", value)
                else:
                    # Training mode: score is list (token-level rewards)
                    # Only print non-zero count and last value (usually F1 score)
                    if isinstance(score, list) and len(score) > 0:
                        non_zero_count = sum(1 for s in score if s != 0)
                        last_value = score[-1] if score else 0
                        print(f"[score] {non_zero_count} non-zero rewards, final={last_value:.4f}")
                    else:
                        print("[score]", score)
                
                # Print turn count and info_gain_reward (for both training and validation)
                if info_gain_reward:
                    num_turns = len(info_gain_reward) + 1
                    print(f"[turns]", num_turns)
                    print(f"[info_gain_reward]", info_gain_reward)

        # Print per-step answer quality statistics
        if hasattr(self, '_step_stats') and self._step_stats['total'] > 0:
            s = self._step_stats
            print(f"[batch_stats] total={s['total']} positive_f1={s['positive_f1']} "
                  f"has_answer={s['has_answer']} placeholder={s['placeholder']} "
                  f"empty_answer={s['empty_answer']} format_error={s['format_error']}")
            self._step_stats = {'total': 0, 'empty_answer': 0, 'placeholder': 0,
                                 'has_answer': 0, 'positive_f1': 0, 'format_error': 0}

        # Save successful trajectories for future SFT
        if sft_samples:
            import time
            save_path = os.path.join(SFT_SAVE_DIR, f"sft_step_{int(time.time())}.jsonl")
            with open(save_path, 'a') as f:
                for sample in sft_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"[sft] Saved {len(sft_samples)} successful samples to {save_path}")

        if is_validation:
            return {
                "f1_scores": f1_scores,
                "em_scores": em_scores,
                "noformatf1_scores": noformatf1_scores,
                "reward_tensor": reward_tensor,
            }
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
