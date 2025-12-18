# Adapt code from https://github.com/yuh-zha/AlignScore/tree/main

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
from minicheck.utils import SYSTEM_PROMPT, USER_PROMPT, USER_PROMPT_OURS
from typing import List
import logging


def sent_tokenize_with_newlines(text):
    blocks = text.split('\n')
    
    tokenized_blocks = [sent_tokenize(block) for block in blocks]
    tokenized_text = []
    for block in tokenized_blocks:
        tokenized_text.extend(block)
        tokenized_text.append('\n')  

    return tokenized_text[:-1]  


class Inferencer():
    def __init__(self, model_name, max_model_len, batch_size, cache_dir) -> None:
        
        self.model_name = model_name

        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

        if model_name == 'flan-t5-large':
            ckpt = 'lytang/MiniCheck-Flan-T5-Large'
            self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, cache_dir=cache_dir, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt, cache_dir=cache_dir)

            self.max_model_len=2048 if max_model_len is None else max_model_len
            self.max_output_length = 256
        
        else:
            if model_name == 'roberta-large':
                ckpt = 'lytang/MiniCheck-RoBERTa-Large'
                self.max_model_len=512 if max_model_len is None else max_model_len

            elif model_name == 'deberta-v3-large':
                ckpt = 'lytang/MiniCheck-DeBERTa-v3-Large'
                self.max_model_len=2048 if max_model_len is None else max_model_len
                
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            config = AutoConfig.from_pretrained(ckpt, num_labels=2, finetuning_task="text-classification", revision='main', token=None, cache_dir=cache_dir)
            config.problem_type = "single_label_classification"

            self.tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True, revision='main', token=None, cache_dir=cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                ckpt, config=config, revision='main', token=None, ignore_mismatched_sizes=False, cache_dir=cache_dir, device_map="auto")
        
        self.model.eval()
        self.batch_size = batch_size
        self.softmax = nn.Softmax(dim=-1)

    def inference_example_batch(self, doc: list, claim: list):
        """
        inference a example,
        doc: list
        claim: list
        using self.inference to batch the process
        """

        assert len(doc) == len(claim), "doc must has the same length with claim!"

        max_support_probs = []
        used_chunks = []
        support_prob_per_chunk = []
        
        for one_doc, one_claim in tqdm(zip(doc, claim), desc="Evaluating", total=len(doc)):
            output = self.inference_per_example(one_doc, one_claim)
            max_support_probs.append(output['max_support_prob'])
            used_chunks.append(output['used_chunks'])
            support_prob_per_chunk.append(output['support_prob_per_chunk'])
        
        return {
            'max_support_probs': max_support_probs,
            'used_chunks': used_chunks,
            'support_prob_per_chunk': support_prob_per_chunk
        }

    def inference_per_example(self, doc:str, claim: str):
        """
        inference a example,
        doc: string
        claim: string
        using self.inference to batch the process
        """
        def chunks(lst, n):
            """Yield successive chunks from lst with each having approximately n tokens.

            For flan-t5, we split using the white space;
            For roberta and deberta, we split using the tokenization.
            """
            if self.model_name == 'flan-t5-large':
                current_chunk = []
                current_word_count = 0
                for sentence in lst:
                    sentence_word_count = len(sentence.split())
                    if current_word_count + sentence_word_count > n:
                        yield ' '.join(current_chunk)
                        current_chunk = [sentence]
                        current_word_count = sentence_word_count
                    else:
                        current_chunk.append(sentence)
                        current_word_count += sentence_word_count
                if current_chunk:
                    yield ' '.join(current_chunk)
            else:
                current_chunk = []
                current_token_count = 0
                for sentence in lst:
                    sentence_word_count = len(self.tokenizer(
                        sentence, padding=False, add_special_tokens=False, 
                        max_length=self.max_model_len, truncation=True)['input_ids'])
                    if current_token_count + sentence_word_count > n:
                        yield ' '.join(current_chunk)
                        current_chunk = [sentence]
                        current_token_count = sentence_word_count
                    else:
                        current_chunk.append(sentence)
                        current_token_count += sentence_word_count
                if current_chunk:
                    yield ' '.join(current_chunk)

        doc_sents = sent_tokenize_with_newlines(doc)
        doc_sents = doc_sents or ['']

        doc_chunks = [chunk.replace(" \n ", '\n').strip() for chunk in chunks(doc_sents, self.chunk_size)]
        doc_chunks = [chunk for chunk in doc_chunks if chunk != '']

        '''
        [chunk_1, chunk_2, chunk_3, chunk_4, ...]
        [claim]
        '''
        claim_repeat = [claim] * len(doc_chunks)
        
        output = self.inference(doc_chunks, claim_repeat)
        
        return output

    def inference(self, doc, claim):
        """
        inference a list of doc and claim

        Standard aggregation (max) over chunks of doc

        Note: We do not have any post-processing steps for 'claim'
        and directly check 'doc' against 'claim'. If there are multiple 
        sentences in 'claim'. Sentences are not splitted and are checked 
        as a single piece of text.
        
        If there are multiple sentences in 'claim', we suggest users to 
        split 'claim' into sentences beforehand and prepares data like 
        (doc, claim_1), (doc, claim_2), ... for a multi-sentence 'claim'.

        **We leave the user to decide how to aggregate the results from multiple sentences.**

        Note: AggreFact-CNN is the only dataset that contains three-sentence 
        summaries and have annotations on the whole summaries, so we do not 
        split the sentences in each 'claim' during prediciotn for simplicity. 
        Therefore, for this dataset, our result is based on treating the whole 
        summary as a single piece of text (one 'claim').

        In general, sentence-level prediciton performance is better than that on 
        the full-response-level.
        """

        if isinstance(doc, str) and isinstance(claim, str):
            doc = [doc]
            claim = [claim]
        
        batch_input, _, batch_org_chunks = self.batch_tokenize(doc, claim)

        label_probs_list = []
        used_chunks = []

        for mini_batch_input, batch_org_chunk in zip(batch_input, batch_org_chunks):

            mini_batch_input = {k: v.to(self.model.device) for k, v in mini_batch_input.items()}

            with torch.no_grad():

                if self.model_name == 'flan-t5-large':
                    
                    decoder_input_ids = torch.zeros((mini_batch_input['input_ids'].size(0), 1), dtype=torch.long).to(self.model.device)
                    outputs = self.model(input_ids=mini_batch_input['input_ids'], attention_mask=mini_batch_input['attention_mask'], decoder_input_ids=decoder_input_ids)
                    logits = outputs.logits.squeeze(1)

                    # 3 for no support and 209 for support
                    label_logits = logits[:, torch.tensor([3, 209])].cpu()
                    label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
                
                else:

                    outputs = self.model(**mini_batch_input)
                    logits = outputs.logits
                    label_probs = F.softmax(logits, dim=1)    

                label_probs_list.append(label_probs)
                used_chunks.extend(batch_org_chunk)

        label_probs = torch.cat(label_probs_list)
        support_prob_per_chunk = label_probs[:, 1].cpu().numpy()
        max_support_prob = label_probs[:, 1].max().item()
        
        return {
            'max_support_prob': max_support_prob,
            'used_chunks': used_chunks,
            'support_prob_per_chunk': support_prob_per_chunk
        }

    def batch_tokenize(self, doc, claim):
        """
        input doc and claims are lists
        """
        assert isinstance(doc, list) and isinstance(claim, list)
        assert len(doc) == len(claim), "doc and claim should be in the same length."

        original_text = [self.tokenizer.eos_token.join([one_doc, one_claim]) for one_doc, one_claim in zip(doc, claim)]

        batch_input = []
        batch_concat_text = []
        batch_org_chunks = []
        for mini_batch in self.chunks(original_text, self.batch_size):
            if self.model_name == 'flan-t5-large':
                model_inputs = self.tokenizer(
                    ['predict: ' + text for text in mini_batch], 
                    max_length=self.max_model_len, 
                    truncation=True, 
                    padding=True, 
                    return_tensors="pt"
                ) 
            else:
                model_inputs = self.tokenizer(
                    [text for text in mini_batch], 
                    max_length=self.max_model_len, 
                    truncation=True, 
                    padding=True, 
                    return_tensors="pt"
                ) 
            batch_input.append(model_inputs) 
            batch_concat_text.append(mini_batch)  
            batch_org_chunks.append([item[:item.find('</s>')] for item in mini_batch]) 

        return batch_input, batch_concat_text, batch_org_chunks

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    def fact_check(self, doc, claim):

        outputs = self.inference_example_batch(doc, claim)
        return outputs['max_support_probs'], outputs['used_chunks'], outputs['support_prob_per_chunk']



class LLMCheckHF:
    """
    Versión de LLMCheck que utiliza la biblioteca estándar 'transformers' 
    de Hugging Face para la inferencia, en lugar de vLLM.
    """

    def __init__(self, model_id, batch_size=1, tensor_parallel_size=1, max_tokens=None, cache_dir=None, enable_prefix_caching=False, max_model_len=None):
        # (tensor_parallel_size y enable_prefix_caching se ignoran en esta implementación,
        #  pero se mantienen por compatibilidad de firma)

        import logging
        logging.basicConfig(
            level=logging.INFO,  
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

        logging.info("Reminder: Please set the CUDA device before initializing the LLMCheck object.")

        if model_id == 'Bespoke-MiniCheck-7B':
            self.model_id = 'bespokelabs/Bespoke-MiniCheck-7B'
            self.operating_mode="bespoke"
        elif model_id == 'Granite-Guardian-3.3-8B':
            self.model_id = 'ibm-granite/granite-guardian-3.3-8b'
            self.operating_mode="gg_hybrid"
        else:
            self.model_id = model_id
            self.operating_mode="ours"

        self.batch_size = batch_size
        self.max_model_len = 32768 if max_model_len is None else max_model_len # max input length (prompt + doc)
        self.default_chunk_size = self.max_model_len - 300 # reserve some space (hard coded) for the claim to be checked
        self.cache_dir = cache_dir

        self.user_prompt = USER_PROMPT
        self.user_prompt_ours = USER_PROMPT_OURS
        self.system_prompt = SYSTEM_PROMPT
        # self.enable_prefix_caching = enable_prefix_caching # No usado en HF estándar

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
        
        # --- Reemplazo de VLLM por Hugging Face Transformers ---
        logging.info(f"Loading tokenizer: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            cache_dir=self.cache_dir,
            trust_remote_code=True if self.model_id == 'bespokelabs/Bespoke-MiniCheck-7B' else False,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
             self.tokenizer.pad_token = self.tokenizer.eos_token
             logging.info("Tokenizer pad_token set to eos_token.")

        logging.info(f"Loading model: {self.model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            # download_dir=self.cache_dir,
            trust_remote_code=True if self.model_id == 'bespokelabs/Bespoke-MiniCheck-7B' else False,
            device_map="auto" # Carga el modelo distribuido en las GPUs disponibles
        )
        self.model.eval()
        logging.info("Model loaded successfully.")
        
        # --- Configuración de parámetros de generación (reemplaza SamplingParams) ---
        terminators = [
            self.tokenizer.eos_token_id,
        ]
        converted_token = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if converted_token is not None and converted_token != self.tokenizer.unk_token_id:
            terminators.append(converted_token)

        self.generation_params = {
            "temperature": 0.6,
            # "stop_token_ids": terminators,
        }

        if self.operating_mode == "bespoke":
            # 'bespoke' solo necesita el primer token 'yes'/'no' y sus probabilidades
            self.generation_params["max_new_tokens"] = max_tokens if max_tokens is not None else 1
            self.generation_params["output_scores"] = True
            self.generation_params["return_dict_in_generate"] = True
            
            # Pre-calcular IDs de token para "yes"
            yes_tokens = ['yes', 'Yes', ' yes', ' Yes']
            self.yes_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in yes_tokens]
            self.yes_token_ids = [tid for tid in self.yes_token_ids if tid is not None and tid != self.tokenizer.unk_token_id]
            logging.info(f"Operating mode 'bespoke'. Watching for token IDs: {self.yes_token_ids}")

        else:
            # 'gg_hybrid' y 'ours' necesitan generar texto (CoT + respuesta)
            self.generation_params["max_new_tokens"] = max_tokens if max_tokens is not None else 100 # Default más grande
            self.generation_params["output_scores"] = False
            self.generation_params["return_dict_in_generate"] = False # Solo necesitamos los IDs generados
            logging.info(f"Operating mode '{self.operating_mode}'. Max new tokens: {self.generation_params['max_new_tokens']}")

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def sent_tokenize_with_newlines(self, text):
        blocks = text.split('\n')
        
        tokenized_blocks = [sent_tokenize(block) for block in blocks]
        tokenized_text = []
        for block in tokenized_blocks:
            tokenized_text.extend(block)
            tokenized_text.append('\n')  

        return tokenized_text[:-1] 
    

    def apply_chat_template(self, doc, claim):
        # Esta función es idéntica a la original
        if self.operating_mode=="bespoke":
            user_prompt = self.user_prompt.replace("[DOCUMENT]", doc).replace("[CLAIM]", claim)
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        elif self.operating_mode=="gg_hybrid":
            documents = [{'doc_id':'0', 'text': doc}]
            messages = [{"role": "assistant", "content": claim}]
            guardian_config = {"criteria_id": "groundedness"}
            text = self.tokenizer.apply_chat_template(messages, guardian_config=guardian_config, documents=documents, think=True, tokenize=False, add_generation_prompt=True)
        elif self.operating_mode=="ours":
            user_prompt = self.user_prompt_ours.replace("[DOCUMENT]", doc).replace("[CLAIM]", claim)
            message = [
                {"role": "user", "content": user_prompt},
            ]
            text = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
           
        return text

    
    # get_support_prob (la versión de vllm) se elimina, ya que la lógica
    # de procesamiento de scores se moverá a 'score' para el procesamiento por lotes.
    
    def get_support_prob_hybrid_gg(self, response_text: str, marker="score"):
        """
        Procesa un string de respuesta (en lugar de un objeto de vllm).
        La lógica interna es idéntica a la original.
        """
        response_text = response_text.lower()
        try:
            # NOTA: La lógica original se preserva.
            # (1.0 si '<score> no </score>' está presente)
            support_prob=1.0 if f"<{marker}> no </{marker}>" in response_text else 0.0
        except Exception as e:
            print("Error:", e)
            support_prob = random.random()
        return support_prob

    def extract_cot_and_answer(self, response_text, marker="answer"):
        """Función de utilidad idéntica a la original."""
        import re
        
        # Extract CoT content from <think></think>
        cot_pattern = r'<think>(.*?)</think>'
        cot_match = re.search(cot_pattern, response_text, re.DOTALL)
        cot_text = cot_match.group(1).strip() if cot_match else ""
        
        # # Extract answer from <{marker}></{marker}>
        answer_pattern = rf'<{marker}>(.*?)</{marker}>'
        answer_match = re.search(answer_pattern, response_text, re.DOTALL)
        answer_text = answer_match.group(1).strip() if answer_match else ""

        return cot_text, answer_text

    def get_support_prob_ours(self, response_text: str, marker="answer"):
        """
        Procesa un string de respuesta (en lugar de un objeto de vllm).
        La lógica interna es idéntica a la original.
        """
        response_text_lower = response_text.lower()
        try:
            support_prob=1.0 if (f"<{marker}> yes </{marker}>" in response_text_lower) or (f"<{marker}>yes</{marker}>" in response_text_lower) else 0.0
        except Exception as e:
            print("Error:", e)
            support_prob = random.random()
        
        # Extrae CoT y respuesta del texto de respuesta original (no en minúsculas)
        cot_text, answer_text = self.extract_cot_and_answer(response_text, marker)
        
        return support_prob, cot_text, answer_text, response_text_lower


    def get_all_chunks_per_doc(self, doc, claim):
        """Función idéntica a la original."""
    
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
        """
        Método 'score' modificado para usar 'model.generate' de transformers 
        con procesamiento por lotes.
        """

        self.doc_chunk_cache = {}
        self.chunk_size = chunk_size if chunk_size else self.default_chunk_size

        assert self.chunk_size < self.max_model_len, \
            "chunk_size must be less than max_model_len so that MiniCheck can process the claim"

        all_prompts = []
        doc_claim_indices = []
        
        # 1. Preparación de Prompts (Idéntico al original)
        for index, (doc, claim) in tqdm(enumerate(zip(docs, claims)), total=len(docs), desc="Tokenizing"):
            chunks_data = self.get_all_chunks_per_doc(doc, claim)
            doc_chunks = chunks_data['doc_chunks']
            claim_repeat = chunks_data['claim_repeat']

            claim_sentences = self.split_into_sentences(claim)

            prompts = []
            for doc_chunk in doc_chunks:
                for sentence in claim_sentences:
                    prompt = self.apply_chat_template(doc_chunk, sentence)
                    prompts.append(prompt)

            all_prompts.extend(prompts)
            doc_claim_indices.extend([index] * len(prompts))

        # 2. Generación por Lotes (Reemplazo de llm.generate)
        probs_per_chunk_sentence = []
        cot_texts = []
        answer_texts = []
        ori_reponses_texts = []
        
        all_decoded_text_for_parsing = [] # Para modos 'gg_hybrid' y 'ours'

        for i in tqdm(range(0, len(all_prompts), self.batch_size), desc="Generating"):
            batch_prompts = all_prompts[i:i + self.batch_size]
            
            # Tokenizar el lote
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.max_model_len
            ).to(self.model.device)

            # Generar respuesta
            with torch.no_grad():
                response = self.model.generate(**inputs, **self.generation_params)

            # import logging
            # logging.basicConfig(
            #     level=logging.INFO,  
            #     format='%(asctime)s [%(levelname)s] %(message)s',
            #     handlers=[
            #         logging.StreamHandler()
            #     ]
            # )
            # logging.info(f"operating_mode: {self.operating_mode}.")

            # 3. Procesamiento de Salida por Lote
            if self.operating_mode == "bespoke":
                # 'response' es un dict que contiene 'scores'
                # Tomamos los scores del primer token generado
                scores = response.scores[0] # [batch_size, vocab_size]
                probs = F.softmax(scores, dim=-1)
                
                # Sumar las probabilidades de todos los tokens 'yes'
                support_probs = probs[:, self.yes_token_ids].sum(dim=-1)
                probs_per_chunk_sentence.extend(support_probs.cpu().tolist())
            
            else:
                # 'response' es solo el tensor de IDs generados
                # Decodificar solo el texto generado (excluyendo el prompt)
                input_len = inputs['input_ids'].shape[1]
                generated_ids_only = response[:, input_len:]
                
                decoded_text_batch = self.tokenizer.batch_decode(
                    generated_ids_only, 
                    skip_special_tokens=True
                )
                # logging.info(f"decoded {decoded_text_batch}.")
                all_decoded_text_for_parsing.extend(decoded_text_batch)
                # print(all_decoded_text_for_parsing)

        # 3b. Procesamiento de Texto (para modos no-'bespoke')
        if self.operating_mode != "bespoke":
            if self.operating_mode == "gg_hybrid":
                probs_per_chunk_sentence = [self.get_support_prob_hybrid_gg(text) for text in all_decoded_text_for_parsing]
            elif self.operating_mode == "ours":
                ours_results = [self.get_support_prob_ours(text) for text in all_decoded_text_for_parsing]
                probs_per_chunk_sentence = [result[0] for result in ours_results]
                cot_texts = [result[1] for result in ours_results]
                answer_texts = [result[2] for result in ours_results]
                ori_reponses_texts = [result[3] for result in ours_results]


        # 4. Agregación (Idéntica a la original)
        result_dict = {}
        for index, prob_per_chunk_sentence in zip(doc_claim_indices, probs_per_chunk_sentence):
            if index not in result_dict:
                result_dict[index] = []
            result_dict[index].append(prob_per_chunk_sentence)

        probs_per_doc_claim_pair = [result_dict.get(index, [0.0]) for index in range(len(docs))] # Usar .get con default
        pred_label, max_support_prob, used_chunk, support_prob_per_chunk = [], [], [], []

        for idx in range(len(probs_per_doc_claim_pair)):

            doc = docs[idx]
            claim = claims[idx]

            claim_sentences = self.split_into_sentences(claim)
            num_chunks = len(self.get_all_chunks_per_doc(doc, claim)['doc_chunks'])
            num_sentences = len(claim_sentences)

            # Asegurarse de que la matriz tenga la forma correcta, incluso si falta
            current_probs = probs_per_doc_claim_pair[idx]
            expected_len = num_chunks * num_sentences
            if len(current_probs) != expected_len:
                # Si hay un desajuste (ej. doc vacío), rellenar con 0s
                current_probs = current_probs + [0.0] * (expected_len - len(current_probs))
                current_probs = current_probs[:expected_len] # Truncar si es demasiado largo
            
            if num_chunks == 0 or num_sentences == 0:
                prob_matrix = np.array([]).reshape(num_chunks, num_sentences) # Matriz vacía
                final_score = 0.0 # Default para documentos o claims vacíos
            else:
                prob_matrix = np.array(current_probs).reshape(num_chunks, num_sentences)
                
                # Para cada sentencia, coger la prob max en todos los chunks
                max_prob_per_sentence = np.max(prob_matrix, axis=0)
    
                # La puntuación final es el mínimo de estos valores máximos
                final_score = np.min(max_prob_per_sentence) if max_prob_per_sentence.size > 0 else 0.0

            pred_label.append(1 if final_score > 0.5 else 0)
            max_support_prob.append(final_score)
            used_chunk.append(self.get_all_chunks_per_doc(doc, claim)['doc_chunks'])
            support_prob_per_chunk.append(prob_matrix)

        return pred_label, max_support_prob, used_chunk, support_prob_per_chunk, all_prompts, cot_texts, answer_texts, ori_reponses_texts

    def split_into_sentences(self, text: str) -> List[str]:
        # Corregido el error tipográfico (”。" -> "()")
        return nltk.sent_tokenize(text)

class LLMCheck:

    def __init__(self, model_id, tensor_parallel_size=1, max_tokens=1, cache_dir=None, enable_prefix_caching=False, max_model_len=None):
        from vllm import LLM, SamplingParams

        import logging
        logging.basicConfig(
            level=logging.INFO,  
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

        logging.info("Reminder: Please set the CUDA device before initializing the LLMCheck object.")

        if model_id == 'Bespoke-MiniCheck-7B':
            self.model_id = 'bespokelabs/Bespoke-MiniCheck-7B'
            self.operating_mode="bespoke"
        elif model_id == 'Granite-Guardian-3.3-8B':
            self.model_id = 'ibm-granite/granite-guardian-3.3-8b'
            self.operating_mode="gg_hybrid"
        else:
            self.model_id = model_id
            self.operating_mode="ours"

        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.max_model_len = 32768 if max_model_len is None else max_model_len # max input length (prompt + doc)
        self.default_chunk_size = self.max_model_len - 300 # reserve some space (hard coded) for the claim to be checked
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
            trust_remote_code=True if self.model_id == 'bespokelabs/Bespoke-MiniCheck-7B' else False,
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
        if self.operating_mode=="bespoke":
            user_prompt = self.user_prompt.replace("[DOCUMENT]", doc).replace("[CLAIM]", claim)
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        elif self.operating_mode=="gg_hybrid":
            documents = [{'doc_id':'0', 'text': doc}]
            messages = [{"role": "assistant", "content": claim}]
            guardian_config = {"criteria_id": "groundedness"}
            text = self.tokenizer.apply_chat_template(messages, guardian_config=guardian_config, documents=documents, think=True, tokenize=False, add_generation_prompt=True)
        elif self.operating_mode=="ours":
            user_prompt = self.user_prompt_ours.replace("[DOCUMENT]", doc).replace("[CLAIM]", claim)
            message = [
                {"role": "user", "content": user_prompt},
            ]
            text = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
           
        
        return text

    
    def get_support_prob(self, response):
        """probs from vllm inference"""
        import math
        support_prob = 0

        for token_prob in response.outputs[0].logprobs[0].values():
            decoded_token = token_prob.decoded_token
            if decoded_token.lower() == 'yes': 
                support_prob += math.exp(token_prob.logprob)
        
        return support_prob
    
    def get_support_prob_hybrid_gg(self, response, marker="score"):
        """probs from vllm inference"""
        response_text = response.outputs[0].text.lower()
        # print('response_text', response_text)
        try:
            support_prob=1.0 if f"<{marker}> no </{marker}>" in response_text else 0.0
        except Exception as e:
            print("Error:", e)
            support_prob = random.random()
        return support_prob

    def extract_cot_and_answer(self, response_text, marker="answer"):
        """Extract CoT content from <think></think> and answer from <{marker}></{marker}>"""
        import re
        
        cot_pattern = r'<reason>(.*?)</reason>'
        cot_match = re.search(cot_pattern, response_text, re.DOTALL)
        cot_text = cot_match.group(1).strip() if cot_match else ""
        
        # # Extract answer from <{marker}></{marker}>
        answer_pattern = rf'<{marker}>(.*?)</{marker}>'
        answer_match = re.search(answer_pattern, response_text, re.DOTALL)
        answer_text = answer_match.group(1).strip() if answer_match else ""


        return cot_text, answer_text

    def get_support_prob_ours(self, response, marker="answer"):
        """probs from vllm inference"""
        response_text = response.outputs[0].text.lower()
        try:
            support_prob=1.0 if (f"<{marker}> yes </{marker}>" in response_text) or (f"<{marker}>yes</{marker}>" in response_text) else 0.0
        except Exception as e:
            print("Error:", e)
            support_prob = random.random()
        
        # Extract CoT and answer from original response text (not lowercased)
        original_response_text = response.outputs[0].text.lower()
        cot_text, answer_text = self.extract_cot_and_answer(original_response_text, marker)
        
        return support_prob, cot_text, answer_text, original_response_text


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

            # 原本的
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
        
        responses = self.llm.generate(all_prompts, self.sampling_params) 

        if self.operating_mode=="bespoke":
            probs_per_chunk_sentence = [self.get_support_prob(responses[idx]) for idx in range(len(responses))]
        elif self.operating_mode=="gg_hybrid":
            probs_per_chunk_sentence = [self.get_support_prob_hybrid_gg(responses[idx]) for idx in range(len(responses))]
        elif self.operating_mode=="ours":
            ours_results = [self.get_support_prob_ours(responses[idx]) for idx in range(len(responses))]
            probs_per_chunk_sentence = [result[0] for result in ours_results]  # Extract only support_prob
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