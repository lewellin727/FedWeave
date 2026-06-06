import gc
import torch

from src.cla import MountedLoRAs
from src.r_matrix import compute_R_via_colbert
from src.utils import USER_PROMPT_LORA, model_generate


def cla_inference(question, base_model, cla_module, lora_paths, doc_passages, encoder, cla_cfg, tokenizer, generation_config):
    """Generate an answer with CLA-mounted base model + N doc-LoRAs + (R, scores) prior."""
    assert len(lora_paths) == len(doc_passages), f'cla_inference: lora_paths ({len(lora_paths)}) and doc_passages ({len(doc_passages)}) length mismatch'
    R, scores = compute_R_via_colbert(question, doc_passages, encoder, max_doc_len=cla_cfg['R_max_doc_len'], max_query_len=cla_cfg['R_max_query_len'])

    device = base_model.device
    base_dtype = next(base_model.parameters()).dtype
    prompt = USER_PROMPT_LORA.format(question=question, passages=None)

    with torch.no_grad():
        with MountedLoRAs(cla_module, lora_paths, R, scores, device=device, dtype=base_dtype):
            pred = model_generate(prompt, base_model, tokenizer, generation_config)

    torch.cuda.empty_cache()
    gc.collect()
    return pred
