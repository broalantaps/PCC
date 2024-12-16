import logging
import os
import time
from typing import Dict, List, Optional

import evaluate
import numpy as np
import torch

from huggingface_hub import HfApi
from model.model import PCC
from torch.utils.data import DataLoader
from transformers import Trainer,TrainerCallback
from utils.utils import DataCollator

logger = logging.getLogger(__name__)
hf_token = os.environ.get('HF_TOKEN', '')

def compute_rag_metrics(metrics,tokenizer, eval_pred):
    with torch.no_grad():
        loss_rag, logits,labels,ppl_loss = eval_pred
        predictions = torch.argmax(logits, axis=-1)
        rag_val_loss = loss_rag.mean().item()
        
        mask = labels!=-100
        labels = labels[mask]
        predictions = predictions[mask]
        
        pred_texts = tokenizer.decode(predictions, skip_special_tokens=True)
        # pred_texts = generate_text
        label_texts = tokenizer.decode(labels, skip_special_tokens=True)
        # print(label_texts)

        bleu_score_all = metrics["bleu"].compute(
            predictions=[pred_texts], references=[[label_texts]]
        )
        rouge = metrics["rouge"].compute(
            predictions=[pred_texts], references=[[label_texts]]
        )
        
        bleu_score = bleu_score_all["bleu"]
        bleu_score_1gram = bleu_score_all["precisions"][0]
        bleu_score_2gram = bleu_score_all["precisions"][1]
        bleu_score_3gram = bleu_score_all["precisions"][2]
        bleu_score_4gram = bleu_score_all["precisions"][3]

        rouge1 = rouge["rouge1"]
        rouge2 = rouge["rouge2"]
        rougeL = rouge["rougeL"]

    return {
        "eval_rag_val_loss": rag_val_loss,
        "eval_bleu": bleu_score,
        "eval_bleu_1gram": bleu_score_1gram,
        "eval_bleu_2gram": bleu_score_2gram,
        "eval_bleu_3gram": bleu_score_3gram,
        "eval_bleu_4gram": bleu_score_4gram,
        "eval_rouge1": rouge1,
        "eval_rouge2": rouge2,
        "eval_rougeL": rougeL
    }

def compute_metrics(metrics, tokenizer, eval_pred):
    with torch.no_grad():
        loss_nt,loss_ae, logits, labels,ppl_loss = eval_pred
        predictions = torch.argmax(logits, axis=-1)
        ae_val_loss = loss_ae.mean().item()
        nt_val_loss = loss_nt.mean().item()

        mask = labels != -100
        labels = labels[mask]
        predictions = predictions[mask]

        ae_perplexity = np.exp(ae_val_loss)
        nt_perplexity = np.exp(nt_val_loss)
        
        pred_texts = tokenizer.decode(predictions, skip_special_tokens=True)
        label_texts = tokenizer.decode(labels, skip_special_tokens=True)

        bleu_score_all = metrics["bleu"].compute(
            predictions=[pred_texts], references=[[label_texts]]
        )
        rouge = metrics["rouge"].compute(
            predictions=[pred_texts], references=[[label_texts]]
        )
        
        bleu_score = bleu_score_all["bleu"]
        bleu_score_1gram = bleu_score_all["precisions"][0]
        bleu_score_2gram = bleu_score_all["precisions"][1]
        bleu_score_3gram = bleu_score_all["precisions"][2]
        bleu_score_4gram = bleu_score_all["precisions"][3]

        rouge1 = rouge["rouge1"]
        rouge2 = rouge["rouge2"]
        rougeL = rouge["rougeL"]

    return {
        "eval_ae_val_loss": ae_val_loss,
        "eval_next_val_loss": nt_val_loss,
        "eval_bleu": bleu_score,
        "eval_bleu_1gram": bleu_score_1gram,
        "eval_bleu_2gram": bleu_score_2gram,
        "eval_bleu_3gram": bleu_score_3gram,
        "eval_bleu_4gram": bleu_score_4gram,
        "eval_rouge1": rouge1,
        "eval_rouge2": rouge2,
        "eval_rougeL": rougeL,
        "eval_ae_ppl": ae_perplexity,
        "eval_next_ppl": nt_perplexity,
    }


def speed_metrics(split, start_time, num_samples=None, num_steps=None, num_tokens=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    - num_steps: number of steps processed
    - num_tokens: number of tokens processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if runtime == 0:
        return result
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    if num_tokens is not None:
        tokens_per_second = num_tokens / runtime
        result[f"{split}_tokens_per_second"] = round(tokens_per_second, 3)
    return result

class LogCallBack(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        if logs is not None and state.is_world_process_zero:
            log_msg = (
                f"Epoch: {logs.get('epoch', 0):.2f} | "
                f"Step: {state.global_step} | "
                f"Loss: {logs.get('loss', 0):.4f} | "
                f"Grad Norm: {logs.get('grad_norm', 0):.4f} | "
                f"Learning Rate: {logs.get('learning_rate', 0):.2e} | "  # Using scientific notation
            )
            print(log_msg)
            
class BaseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = self.load_metrics()
        self.stage = self.args.stage
        
    def compute_loss(
        self,
        model: PCC,
        inputs: Dict[str, List[str]],
        **kwargs,
    ):
        # embed_txt = inputs["embed_txt"]
        compress_ids = inputs["compress_ids"]
        if self.stage == 1:
            llm_ids = inputs["llm_ids"]
            next_ids = inputs["next_ids"]
            prompt_text = inputs["prompt_text"]
            labels_ids = None
            # loss = model(compress_ids=compress_ids,llm_ids=llm_ids,labels_ids=None,
            #              prompt_text=prompt_text, next_ids=next_ids,task_type=None)
        else:
            llm_ids = inputs["query_answer_ids"]
            labels_ids = inputs["label_ids"]
            prompt_text = None
            next_ids = None
        
        loss = model(compress_ids=compress_ids,llm_ids=llm_ids,labels_ids=labels_ids,
                         prompt_text=prompt_text, next_ids=next_ids,task_type=None)
        
        return loss["loss"]

    def load_metrics(self):
        metrics = {
            "bleu": evaluate.load("bleu"),
            "rouge": evaluate.load("rouge"),
        }
        return metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        total_metrics = {}
        num_batches = 0
        self.model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                compress_ids = batch["compress_ids"]
                if self.stage == 2:
                    llm_ids = batch["query_answer_ids"]
                    labels_ids = batch["label_ids"]
                    next_ids = None
                    prompt_text = None
                    with torch.no_grad():
                        outputs_rag = self.model(compress_ids=compress_ids,llm_ids=llm_ids,labels_ids=labels_ids,
                         prompt_text=prompt_text, next_ids=None,task_type="rag")
                    loss_rag = outputs_rag["loss"]
                    logits_rag = outputs_rag[
                        "logits"
                    ]  # Adjust based on your model's specific output
                    labels_rag = outputs_rag["target"]  # Ensure your data includes labels
                    ppl_loss_rag = outputs_rag["ppl_loss"]
                    batch_metrics = compute_rag_metrics(
                        self.metrics,
                        self.model.decoder.tokenizer,
                        (loss_rag, logits_rag, labels_rag, ppl_loss_rag),
                    )
                    for k, v in batch_metrics.items():
                        if k in total_metrics:
                            total_metrics[k] += v
                        else:
                            total_metrics[k] = v
                    num_batches += 1
                    
                elif self.stage == 1:
                    prompt_text = batch["prompt_text"]
                    llm_ids = batch["llm_ids"]
                    next_ids = batch["next_ids"]
                   
                    with torch.no_grad():
                        outputs_ae = self.model(compress_ids=compress_ids,llm_ids=llm_ids,labels_ids=None,
                            prompt_text=prompt_text, next_ids=next_ids,task_type="ae")
                        outputs_nt = self.model(compress_ids=compress_ids,llm_ids=llm_ids,labels_ids=None,
                            prompt_text=prompt_text, next_ids=next_ids,task_type="next_token")

                    loss_ae = outputs_ae["loss"]
                    loss_nt = outputs_nt["loss"]
                    logits_ae = outputs_ae[
                        "logits"
                    ]  # Adjust based on your model's specific output
                    labels_ae = outputs_ae["target"]  # Ensure your data includes labels
                    ppl_loss_ae = outputs_ae["ppl_loss"]
                    batch_metrics = self.compute_metrics(
                        self.metrics,
                        self.model.decoder.tokenizer,
                        (loss_nt, loss_ae, logits_ae, labels_ae, ppl_loss_ae),
                    )
                    for k, v in batch_metrics.items():
                        if k in total_metrics:
                            total_metrics[k] += v
                        else:
                            total_metrics[k] = v
                    num_batches += 1


        average_metrics = {k: v / num_batches for k, v in total_metrics.items()}

        self.model.train()
        self.log(average_metrics)
        return average_metrics

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizers
        llm_save_dir = os.path.join(output_dir, "llm")
        os.makedirs(llm_save_dir, exist_ok=True)
        self.model.decoder.tokenizer.save_pretrained(llm_save_dir)
        
        if self.model.compressor.tokenizer is not None:
            self.model.compressor.tokenizer.save_pretrained(output_dir)
        
        # Optionally push to HuggingFace if token is available
        hf_token = os.environ.get('HF_TOKEN', '')
        if hf_token:
            try:
                # Determine repository names based on current timestamp or run configuration
                compress_name = f"memory-compressor-{int(time.time())}"
                llm_name = f"llm-{int(time.time())}"
                
                # Push models and tokenizers to HuggingFace
                self.model.compressor.model.push_to_hub(compress_name, use_auth_token=hf_token)
                self.model.compressor.tokenizer.push_to_hub(compress_name, use_auth_token=hf_token)
                
                # Optional: push LLM model and tokenizer
                # Uncomment if you want to push LLM model as well
                # self.model.decoder.model.push_to_hub(llm_name, use_auth_token=hf_token)
                # self.model.decoder.tokenizer.push_to_hub(llm_name, use_auth_token=hf_token)
                
            except Exception as e:
                logger.warning(f"Failed to push models to HuggingFace: {e}")
        
        # Save state dict and other model components
        super()._save(output_dir, state_dict)
 