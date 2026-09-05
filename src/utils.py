import re
import json
import torch
import string
import numpy as np
from typing import List, Union
from collections import Counter

from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------ llm model ------------------------------------------

def get_model_path(model_name, config):
    model_paths = config.get('backbone_paths', {})
    if model_name not in model_paths:
        raise ValueError(
            f'No path configured for backbone {model_name!r}. '
            f'Please add it under backbone_paths in config.yaml.'
        )
    return model_paths[model_name]


def get_model(model_name, config, max_new_tokens=128, dtype=torch.float32, device_map="auto"):
    """Load a model. `device_map='auto'` spreads across visible GPUs; pass
    `device_map={'': 'cuda:N'}` (or a string like `'cuda:0'`) to pin the whole
    model onto one GPU — required when DDP runs multiple ranks per node."""
    print(f'Loading {model_name} (dtype={dtype}, device_map={device_map})...')
    model_path = get_model_path(model_name, config)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    generation_config = dict(
        num_beams=1,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    return model, tokenizer, generation_config


def model_generate(prompt, model, tokenizer, generation_config):
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True
    )
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    output = model.generate(
        input_ids,
        attention_mask = torch.ones(input_ids.shape).to(model.device),
        **generation_config
    )
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text


def get_rewrite(passage, model=None, tokenizer=None, generation_config=None):
    rewrite_prompt = "Rewrite the following passage. While keeping the entities, proper nouns, and key details such as names, locations, and terminology intact, create a new version of the text that expresses the same ideas in a different way. Make sure the revised passage is distinct from the original one, but preserves the core meaning and relevant information.\n{passage}"
    return model_generate(rewrite_prompt.format(passage=passage), model, tokenizer, generation_config)


QA_PROMPT = "I will provide a passage of text, and you need to generate three different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.\n\
You need to generate the question and answer in the following format:\n\
[\n\
    {{\n\
        \"question\": \"What is the capital of France?\",\n\
        \"answer\": \"Paris\",\n\
        \"full_answer\": \"The capital of France is Paris.\"\n\
    }}, \n\
]\n\n\
This list should have at least three elements. You only need to output this list in the above format.\n\
Passage:\n\
{passage}"

def fix_qa(qa):
    if isinstance(qa, list):
        if len(qa) >= 3:
            qa = qa[:3]
            for data in qa:
                if "question" not in data or "answer" not in data or "full_answer" not in data:
                    return False, qa
                if isinstance(data["answer"], list):
                    data["answer"] = ", ".join(data["answer"])
                if isinstance(data["answer"], int):
                    data["answer"] = str(data["answer"])
                if data["answer"] is None:
                    data["answer"] = "Unknown"
            return True, qa
    return False, qa

def get_qa(passage, model_name, model=None, tokenizer=None, generation_config=None):

    def fix_json(output):
        if model_name == "llama3.2-1b-instruct":
            output = output[output.find("["):]
            if output.endswith(","):
                output = output[:-1]
            if not output.endswith("]"):
                output += "]"
        elif model_name == "llama3-8b-instruct":
            if "[" in output:
                output = output[output.find("["):]
            if "]" in output:
                output = output[:output.find("]")+1]
        return output

    TRY_TIMES = 30
    try_times = TRY_TIMES
    prompt = QA_PROMPT.format(passage=passage)
    output = None
    while try_times:
        print(f'tried times: {TRY_TIMES + 1 - try_times}', end='\r')
        output = model_generate(prompt, model, tokenizer, generation_config)
        output = fix_json(output)
        try:
            qa = json.loads(output)
            ret, qa = fix_qa(qa)
            if ret:
                return qa
            try_times -= 1
        except (json.JSONDecodeError, TypeError):
            try_times -= 1
    return output



def _get_prompt(question, passages=None, answer=None):
    question = question.strip()
    if not question.endswith('?'):
        question = question.strip() + '?'
    elif question.endswith(' ?'):
        question = (question[:-1]).strip() + '?'

    if passages and not isinstance(passages, list):
        passages = [passages]

    if answer is None:
        answer = ""
    else:
        answer = str(answer).strip()
        if not answer.endswith('.'):
            answer += "."
    return question, passages, answer

USER_PROMPT_LORA = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n\
{passages}\n\n\
Question: {question}"
ASSISTANT_PROMPT = "Answer: {answer}"

def get_prompt(tokenizer, question, passages=None, answer=None):
    question, passages, answer = _get_prompt(question, passages, answer)
    contexts = ""
    if passages:
        for pid, psg in enumerate(passages):
            contexts += f"Passage {pid+1}: {psg}\n"
    user_content = USER_PROMPT_LORA.format(question=question, passages=contexts)
    assistant_content = ASSISTANT_PROMPT.format(answer=answer)

    messages = [{
        "role": "user",
        "content": user_content,
    }]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True)
    inputs += tokenizer.encode(assistant_content, add_special_tokens=False)
    return inputs



# --------------------------------------- evaluate ----------------------------------------

class BaseDataset:
    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

def evaluate(pred, ground_truth, with_cot=False):
    if "Answer: " in pred:
        pred = pred[pred.find("Answer: ") + len("Answer: "):]
    if "The answer is" in pred:
        pred = pred[pred.find("The answer is") + len("The answer is"):]
    if not with_cot:
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip()
    else:
        if "The answer is" in pred:
            pred = pred[pred.find("The answer is") + len("The answer is"):]
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip()

    em = BaseDataset.exact_match_score(
        prediction=pred,
        ground_truth=ground_truth,
    )["correct"]
    f1_score = BaseDataset.f1_score(
        prediction=pred,
        ground_truth=ground_truth,
    )
    f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
    return {
        "em": str(em),
        "f1": str(f1),
        "prec": str(prec),
        "recall": str(recall),
    }

# --------------------------------------- parallel sharding ----------------------------------------

def shard_list(items, worker_id, num_workers):
    """Round-robin shard: return the subset of items assigned to this worker.

    Workers split work by index modulo num_workers, so given the same input list every
    worker handles a disjoint subset and the union covers everything exactly once.
    """
    assert 0 <= worker_id < num_workers, f'invalid sharding: w{worker_id}/{num_workers}'
    return [item for i, item in enumerate(items) if i % num_workers == worker_id]


def shard_indices(n, worker_id, num_workers):
    """Round-robin shard of range(n)."""
    assert 0 <= worker_id < num_workers, f'invalid sharding: w{worker_id}/{num_workers}'
    return [i for i in range(n) if i % num_workers == worker_id]
