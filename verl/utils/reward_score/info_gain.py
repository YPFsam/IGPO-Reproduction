from openai import OpenAI
import re
import difflib
import string
import json
import os

def check_tags_balance(solution_str: str) -> bool:
    """Check if tags are properly paired
    
    Args:
        solution_str: String to check
    
    Returns:
        bool: Whether all tags are properly paired
    """
    # Tag pairs to check
    # Note: 'think' is NOT checked because the first <think$gt; is injected by Qwen3 chat template
    # into the prompt (not the response). The response only contains the closing </think$gt; for
    # the first turn, so tags will always appear imbalanced in the response alone.
    tags_to_check = ['code', 'tool_call', 'answer']
    
    for tag in tags_to_check:
        # Count start and end tags
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)
        
        # If start and end tag counts don't match, return False
        if start_count != end_count:
            return False
            
        # Check tag nesting order (ensure end tag doesn't appear before start tag)
        last_pos = -1
        while True:
            start_pos = solution_str.find(start_tag, last_pos + 1)
            if start_pos == -1:
                break
                
            end_pos = solution_str.find(end_tag, start_pos)
            if end_pos == -1:
                return False
                
            last_pos = end_pos
            
    return True

def preprocess_text(text: str) -> str:
    """Preprocess text for dataset scoring
    
    Processing steps:
    1. Convert to lowercase
    2. Remove punctuation (.,!?;:'"()[]{}...)
    3. Remove extra whitespace
    """
    # Replace punctuation with spaces
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    return text

def deal_multi_labels(ground_truth):
    for item in ground_truth:
        if item['label'].lower() == 'false':
            return 'false'
    return 'true'



def compute_f1(solution_str, ground_truth, data_source, val_type='f1') -> float:
    if data_source in ['Factbench', 'politifact', 'liar2']:
        ground_truth = json.loads(ground_truth)
        ground_truth = deal_multi_labels(ground_truth)
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    ground_truths = ground_truth.split("<|answer_split|>")
    # First check if tags are properly paired (format correctness)
    if not check_tags_balance(solution_str):
        
        if val_type == 'noformatf1':
            return 0
        else:
            return -2.0

    # Use regex to extract content from first <answer> tag
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            # Preprocess the answer
            answer_content = preprocess_text(answer_content)
        else:
            if val_type == 'noformatf1':
                return 0
            else:
                return -2.0
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        if val_type == 'noformatf1':
            return 0
        else:
            return -2.0
    
    max_score = 0.0
    
    for gt in ground_truths:
        # Preprocess ground truth
        gt = preprocess_text(gt)

        if val_type == 'em':
            if gt == answer_content:
                return 1.0
        else:
            # Tokenize answer and reference answer
            pred_tokens = set(answer_content.split())
            gt_tokens = set(gt.split())
            
            if not gt_tokens:  # Avoid division by zero
                continue
            if not pred_tokens:
                continue
            
            # Calculate common tokens count
            common_tokens = pred_tokens & gt_tokens
            
            # Calculate precision and recall
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
            
            # Calculate F1 score
            if precision + recall > 0:  # Avoid division by zero
                f1 = 2 * (precision * recall) / (precision + recall)
                max_score = max(max_score, f1)
            
    return max_score

def _char_pos_to_token_idx(char_pos, offset_mapping):
    """
    Find the token index corresponding to a character position
    
    Args:
        char_pos: Character position
        offset_mapping: offset_mapping returned by tokenizer, format [(start, end), ...]
    
    Returns:
        Corresponding token index
    """
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_pos < end:
            return i
        if char_pos < start:
            # char_pos is in the gap before current token, return previous token
            return max(0, i - 1)
    # If out of range, return last token
    return len(offset_mapping) - 1


def compute_score(solution_str, ground_truth, data_source, val_type='f1', info_gain_reward=[], tokenizer=None, is_validation=False):
    """【IGPO reward 核心】把 F1 和 info_gain 放到 response 序列的对应 token 上。

    最终输出的 scores 是 token 级别的奖励数组（长度 = token 数），大部分为 0：
      - 中间每一轮的最后一个 token ← info_gain_reward[i]（信息增益，内在奖励）
      - 最后一轮的最后一个 token  ← f1_score（外在奖励）

    这样 advantage 计算时就能区分"哪一轮搜索有价值"。

    关键技术：用 tokenizer 的 offset_mapping 做字符位置→token 位置的精确映射，
    避免 subword 分词导致的索引错位。

    Args:
        solution_str: 模型生成的完整 response（含多轮）
        ground_truth: 标准答案
        info_gain_reward: 每一轮的信息增益列表（长度 = 轮数 - 1，最后一轮只有 F1 没有 IG）
        tokenizer: HuggingFace tokenizer
        is_validation: 验证模式（额外返回 em/noformatf1 指标）
    """
    if tokenizer is None:
        raise ValueError("tokenizer cannot be None")

    alpha = 1.0

    # 算 F1（外在奖励，衡量最终答案对不对）
    if is_validation:
        f1_score = compute_f1(solution_str, ground_truth, data_source, val_type='f1')
        em_score = compute_f1(solution_str, ground_truth, data_source, val_type='em')
        noformatf1_score = compute_f1(solution_str, ground_truth, data_source, val_type='noformatf1')
    else:
        f1_score = compute_f1(solution_str, ground_truth, data_source, val_type)

    # 用 offset_mapping 做字符→token 精确映射
    encoding = tokenizer(solution_str, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding['input_ids']
    offset_mapping = encoding['offset_mapping']  # [(char_start, char_end), ...]

    tokens_size = len(token_ids)
    scores = [0.0] * tokens_size

    # If no tokens, return empty scores directly
    if tokens_size == 0:
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores

    # 多轮 response 的分隔符：每轮新一轮的 assistant 开头
    separator = "\n<|im_start|>assistant\n"

    # 找出每一轮的字符起止位置（用于把 IG 放到正确的轮末 token）
    turn_start_positions = []  # 每轮内容的起始字符位置
    turn_end_positions = []    # 每轮内容的结束字符位置

    # Find all separator positions
    sep_positions = []
    search_pos = 0
    while True:
        sep_pos = solution_str.find(separator, search_pos)
        if sep_pos == -1:
            break
        sep_positions.append(sep_pos)
        search_pos = sep_pos + 1

    if len(sep_positions) == 0:
        # 没有分隔符 → 整个 response 是单轮（GRPO 消融场景）
        turn_start_positions = [0]
        turn_end_positions = [len(solution_str)]
    else:
        # 第一个分隔符前的内容（如果有）
        if sep_positions[0] > 0:
            turn_start_positions.append(0)
            turn_end_positions.append(sep_positions[0])

        # 每个分隔符后的一轮
        for i, sep_pos in enumerate(sep_positions):
            turn_start = sep_pos + len(separator)
            turn_start_positions.append(turn_start)

            if i + 1 < len(sep_positions):
                turn_end = sep_positions[i + 1]
            else:
                turn_end = len(solution_str)
            turn_end_positions.append(turn_end)

    chats_size = len(turn_start_positions)

    # 退化场景：没有 info_gain（GRPO 消融）或只有单轮 → 只把 F1 放最后 token
    if info_gain_reward == [] or chats_size == 1:
        scores[-1] = alpha * f1_score

        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores


    # 校验：info_gain 数量应该 = 轮数 - 1（最后一轮只有 F1，没有 IG）
    if len(info_gain_reward) != chats_size - 1:
        print(f"info_gain.py: turn mismatch - chats_size={chats_size}, info_gain_len={len(info_gain_reward)}")
        # 数量不匹配 → 退化为只用 F1（防御性兜底）
        scores[-1] = alpha * f1_score

        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores

    # 把奖励放到每一轮的最后一个 token 上
    for i in range(chats_size):
        turn_end_char = turn_end_positions[i]

        # 找到这一轮最后一个字符对应的 token 索引
        # 注意：turn_end_char 是开区间，所以用 turn_end_char - 1
        if turn_end_char > 0:
            last_token_idx = _char_pos_to_token_idx(turn_end_char - 1, offset_mapping)
        else:
            last_token_idx = 0

        last_token_idx = min(last_token_idx, tokens_size - 1)

        # 核心赋值逻辑：
        if i < chats_size - 1:
            # 中间轮：放 info_gain（信息增益奖励）
            if i < len(info_gain_reward):
                ig_value = info_gain_reward[i]
                if ig_value == 0.0:
                    ig_value = 1e-10  # 避免 0 被下游 !=0 判断跳过
                scores[last_token_idx] = ig_value
            # else: skip this turn's info_gain (defensive, should not happen)
        else:
            scores[last_token_idx] = alpha * f1_score
    
    if is_validation:
        return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
    return scores
