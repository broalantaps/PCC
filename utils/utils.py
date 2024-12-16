import torch
import random
from typing import Dict, List, Union
from .argument import DataArguments
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
class DataCollator(object):
    def __init__(self, stage, compress_pad_token_id,llm_pad_token_id, compressor_type) -> None:
        self.compress_pad_token_id = compress_pad_token_id
        self.llm_pad_token_id = llm_pad_token_id
        self.stage = stage
        self.compressor_type = compressor_type

    @staticmethod
    def dynamicPadding(
        batch: List[Union[List[int], List[str]]], 
        pad_token_id: int
    ):
        max_list_len = max(len(ids) for ids in batch)
        padded_ids = [
            ids + [pad_token_id] * (max_list_len-len(ids)) for ids in batch
        ]
        return padded_ids
    
    def __call__(self, examples) -> Dict[str, List[str]]:
        if self.compressor_type == 'large':
            compress_ids = [text["llm_ids"] for text in examples]
        elif self.compressor_type == 'lite':
            compress_ids = [text["compress_ids"] for text in examples]
        else:
            raise ValueError("compressor_type must be large or small")
        
        if self.stage == 1:
            llm_ids = [text["llm_ids"] + [128001] for text in examples]
            next_ids = [text["next_ids"] + [128001] for text in examples]      # 128001 is eos token of llama3
            prompt_text = ["<ae>" for _ in examples]
            padded_llm_ids = self.dynamicPadding(llm_ids, self.llm_pad_token_id)
            padded_next_ids = self.dynamicPadding(next_ids, self.llm_pad_token_id)
        # rag
        else:
            query_answer_ids = [text["query_answer_ids"] + [128001] for text in examples]
            label_ids = [text["labels"] + [128001] for text in examples]
            padded_query_answer_ids = self.dynamicPadding(query_answer_ids, self.llm_pad_token_id)
            padded_label_ids = self.dynamicPadding(label_ids, -100)
            
        padded_compress_ids = self.dynamicPadding(compress_ids, self.compress_pad_token_id)

        if self.stage == 1:
            return {
                "compress_ids":padded_compress_ids,
                "llm_ids":padded_llm_ids,
                "next_ids":padded_next_ids,
                "prompt_text": prompt_text,
            }
        else:
            return {
                "compress_ids":padded_compress_ids,
                "query_answer_ids":padded_query_answer_ids,
                "label_ids":padded_label_ids,
            }
            
            
 