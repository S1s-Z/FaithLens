import pandas as pd
import os
from faithlens.faithlens import FaithLens
from typing import List, Dict, Union


class FaithLensInfer:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        enable_prefix_caching: bool = False,
        max_model_len: int = 32768,
        max_tokens: int = 8192,
        **kwargs
    ):
        if device and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
        
        self.scorer = FaithLens(
            model_name=model_name,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
            **kwargs
        )
        self.model_name = model_name
        self.device = device


    def infer(
        self,
        docs: List[str] = None,
        claims: List[str] = None,
    ) -> Union[List[Dict], None]:

        if docs is None or claims is None:
            raise ValueError("The `docs` and `claims` parameter are required when not using a dataset.")
        if len(docs) != len(claims):
            raise ValueError(f"docs' length : ({len(docs)}) claims' length ({len(claims)}) do not match")

        # core
        try:
            pred_label, explanation_texts= self.scorer.score(
                docs=docs,
                claims=claims
            )
        except Exception as e:
            raise RuntimeError(f"Process Error: {str(e)}")

        output_data = []
        for idx in range(len(docs)):
            sample = {
                "doc": docs[idx],
                "claim": claims[idx],
                "pred_label": int(pred_label[idx]),
                "explanation": explanation_texts[idx]
            }
            output_data.append(sample)

        # return json results
        return output_data