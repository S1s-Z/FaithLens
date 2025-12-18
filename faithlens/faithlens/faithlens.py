from nltk.tokenize import sent_tokenize
import numpy as np
import torch
import nltk
import random
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import os
from typing import List

from faithlens.prompts import USER_PROMPT,USER_PROMPT_OURS,SYSTEM_PROMPT
from vllm import LLM, SamplingParams
import logging


class LLMCheck:

    def __init__(self, model_id, tensor_parallel_size=1, max_tokens=1, cache_dir=None, enable_prefix_caching=False, max_model_len=None):
        logging.basicConfig(
            level=logging.INFO,  
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

        logging.info("Reminder: Please set the CUDA device before initializing the LLMCheck object.")


        self.model_id = model_id
        self.operating_mode="ours"

        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.max_model_len = 32768 if max_model_len is None else max_model_len 
        self.default_chunk_size = self.max_model_len - 300 
        self.cache_dir = cache_dir

        self.user_prompt = USER_PROMPT
        self.user_prompt_ours = USER_PROMPT_OURS
        self.system_prompt = SYSTEM_PROMPT
        self.enable_prefix_caching = enable_prefix_caching

        # Check if CUDA is available and get compute capability
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability[0] >= 8:
                self.dtype = torch.bfloat16
                logging.info("Using bfloat16 for LLM initialization.")
            else:
                self.dtype = torch.float16
                logging.info(f"GPU compute capability {compute_capability} < 8.0. Using float16 for LLM initialization.")
        else:
            if torch.cpu.is_available() and hasattr(torch.cpu, 'is_bf16_supported') and torch.cpu.is_bf16_supported():
                self.dtype = torch.bfloat16
                logging.info("CUDA not available. Using bfloat16 on CPU.")
            else:
                self.dtype = torch.float32
                logging.info("CUDA not available and CPU doesn't support bfloat16. Using float32 for LLM initialization.")
        
        self.llm = LLM(
            model=self.model_id, 
            dtype=self.dtype, 
            download_dir=self.cache_dir,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
            seed=2024,
            max_model_len=self.max_model_len,   # need to be adjusted based on the GPU memory available
            enable_prefix_caching=self.enable_prefix_caching
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.padding_side = "left"
        terminators = [
            self.tokenizer.eos_token_id,
        ]
        converted_token = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if converted_token is not None:
            terminators.append(converted_token)

        self.sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=self.max_tokens,
            stop_token_ids=terminators,
            logprobs=5
        )


    def sent_tokenize_with_newlines(self, text):
        blocks = text.split('\n')
        
        tokenized_blocks = [sent_tokenize(block) for block in blocks]
        tokenized_text = []
        for block in tokenized_blocks:
            tokenized_text.extend(block)
            tokenized_text.append('\n')  

        return tokenized_text[:-1] 
    

    def apply_chat_template(self, doc, claim):
        user_prompt = self.user_prompt_ours.replace("[DOCUMENT]", doc).replace("[CLAIM]", claim)
        message = [
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        return text

    
    def extract_explain_and_answer(self, response_text, marker="answer"):
        """Extract CoT content from <think></think> and answer from <{marker}></{marker}>"""
        import re
        
        # Extract CoT content from <think></think>
        explain_pattern = r'<reason>(.*?)</reason>'
        explain_match = re.search(explain_pattern, response_text, re.DOTALL)
        explain_text = explain_match.group(1).strip() if explain_match else ""
        
        # # Extract answer from <{marker}></{marker}>
        answer_pattern = rf'<{marker}>(.*?)</{marker}>'
        answer_match = re.search(answer_pattern, response_text, re.DOTALL)
        answer_text = answer_match.group(1).strip() if answer_match else ""

        # all_answers = re.findall(rf'<{marker}>(.*?)</{marker}>', response_text, re.DOTALL)
        # answer_text = all_answers[-1].strip() if all_answers else ""   

        return explain_text, answer_text

    def get_support_prob_ours(self, response, marker="answer"):
        """probs from vllm inference"""
        response_text = response.outputs[0].text.lower()
        try:
            support_prob=1.0 if (f"<{marker}> yes </{marker}>" in response_text) or (f"<{marker}>yes</{marker}>" in response_text) else 0.0
        except Exception as e:
            print("Error:", e)
            support_prob = random.random()
        
        original_response_text = response.outputs[0].text.lower()
        explain_text, answer_text = self.extract_explain_and_answer(original_response_text, marker)
        
        return support_prob, explain_text, answer_text, original_response_text


    def get_all_chunks_per_doc(self, doc, claim):
    
        def chunks(lst, n):
            """Yield successive chunks from lst with each having approximately n tokens.
            """
            current_chunk = []
            current_word_count = 0
            for sentence in lst:
                sentence_word_count = len(self.tokenizer(sentence, add_special_tokens=False)['input_ids'])
                if current_word_count + sentence_word_count > n:
                    yield ' '.join(current_chunk)
                    current_chunk = [sentence]
                    current_word_count = sentence_word_count
                else:
                    current_chunk.append(sentence)
                    current_word_count += sentence_word_count
            if current_chunk:
                yield ' '.join(current_chunk)

        if doc in self.doc_chunk_cache:
            doc_chunks = self.doc_chunk_cache[doc]
        else:
            doc_sents = self.sent_tokenize_with_newlines(doc)
            doc_sents = doc_sents or ['']
    
            doc_chunks = [chunk.replace(" \n ", '\n').strip() for chunk in chunks(doc_sents, self.chunk_size)]
            doc_chunks = [chunk for chunk in doc_chunks if chunk != '']
            self.doc_chunk_cache[doc] = doc_chunks
        if len(doc_chunks) == 0:
            doc_chunks = [''] 

        claim_repeat = [claim] * len(doc_chunks)

        return {'doc_chunks': doc_chunks, 'claim_repeat': claim_repeat}


    def score(self, docs: List[str], claims: List[str], chunk_size=None) -> List[float]:

        self.doc_chunk_cache = {}
        self.chunk_size = chunk_size if chunk_size else self.default_chunk_size

        assert self.chunk_size < self.max_model_len, \
            "chunk_size must be less than max_model_len so that MiniCheck can process the claim"

        all_prompts = []
        doc_claim_indices = []
        
        for index, (doc, claim) in tqdm(enumerate(zip(docs, claims)), total=len(docs), desc="Tokenizing"):

            chunks = self.get_all_chunks_per_doc(doc, claim)
            doc_chunks = chunks['doc_chunks']
            claim_repeat = chunks['claim_repeat']
            claim_sentences = self.split_into_sentences(claim)
            prompts = []
            for doc_chunk in doc_chunks:
                for sentence in claim_sentences:
                    prompt = self.apply_chat_template(doc_chunk, sentence)
                    prompts.append(prompt)
            all_prompts.extend(prompts)
            doc_claim_indices.extend([index] * len(prompts))
        
        responses = self.llm.generate(all_prompts, self.sampling_params, use_tqdm=False) 

        if self.operating_mode=="ours":
            ours_results = [self.get_support_prob_ours(responses[idx]) for idx in range(len(responses))]
            probs_per_chunk_sentence = [result[0] for result in ours_results] 
            explain_texts = [result[1] for result in ours_results]

        result_dict = {}
        for index, prob_per_chunk_sentence in zip(doc_claim_indices, probs_per_chunk_sentence):
            if index not in result_dict:
                result_dict[index] = []
            result_dict[index].append(prob_per_chunk_sentence)
        
        explain_dict = {}
        for index, explain in zip(doc_claim_indices, explain_texts):
            if index not in explain_dict:
                explain_dict[index] = []
            explain_dict[index].append(explain)


        probs_per_doc_claim_pair = [result_dict[index] for index in range(len(docs))]
        explain_set = [explain_dict[index] for index in range(len(docs))]

        pred_label, final_explains = [], []

        for idx in range(len(probs_per_doc_claim_pair)):

            doc = docs[idx]
            claim = claims[idx]

            claim_sentences = self.split_into_sentences(claim)
            num_chunks = len(self.get_all_chunks_per_doc(doc, claim)['doc_chunks'])
            num_sentences = len(claim_sentences)

            
            prob_matrix = np.array(probs_per_doc_claim_pair[idx]).reshape(num_chunks, num_sentences)

            max_prob_per_sentence = np.max(prob_matrix, axis=0)

            final_score = np.min(max_prob_per_sentence)


            doc_pred = 1 if final_score > 0.5 else 0
            pred_label.append(doc_pred)

            prompt_probs = probs_per_doc_claim_pair[idx]
            prompt_explains = explain_set[idx]
            filtered_explains = [
                exp for prob, exp in zip(prompt_probs, prompt_explains)
                if (prob > 0.5) == (doc_pred == 1)
            ]
            final_explains.append(" ".join(filtered_explains))

        return pred_label, final_explains

    def split_into_sentences(self, text: str) -> List[str]:
        # nltk.data.path.append("/mnt/user/sishuzheng/nltk_data")
        # nltk.data.path.append("/mnt/public/share/users/sishuzheng-share/nltk_data")
        return nltk.sent_tokenize(text)

class FaithLens:
    def __init__(self, model_name='ssz1111/FaithLens', max_model_len=None, batch_size=16, cache_dir=None, tensor_parallel_size=1, max_tokens=1, do_vllm=True, enable_prefix_caching=False) -> None:

        if not max_tokens or max_tokens<2048:
            max_tokens=2048
        if do_vllm:
            self.model = LLMCheck(
                model_id=model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len
            )
        else:
            raise RuntimeError(
                "vLLM is required for this implementation. Please set do_vllm=True to use vLLM backend. "
            )


    def score(self, docs: List[str], claims: List[str], chunk_size=None) -> List[float]:
        assert isinstance(docs, list) or isinstance(docs, np.ndarray), "docs must be a list or np.ndarray"
        assert isinstance(claims, list) or isinstance(claims, np.ndarray), "claims must be a list or np.ndarray"  

        return self._score_llmcheck(docs, claims, chunk_size)
        
    def _score_llmcheck(self, docs, claims, chunk_size):
        return self.model.score(docs, claims, chunk_size)
