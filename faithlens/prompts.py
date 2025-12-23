SYSTEM_PROMPT = """Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. Please assess the claim's consistency with the document by responding with either "Yes" or "No"."""


USER_PROMPT = """Document: [DOCUMENT]\nClaim: [CLAIM]"""



USER_PROMPT_OURS = """Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. 

- First, think step by step about whether all the information in the claim is fully supported by the document within <think> and </think> tags. 

- Then, please provide an easy-to-understand explanation for your answer within <reason> and </reason> tags.

- Finally, assess the claim's consistency with the document by responding with either "Yes" or "No" and wrap your final answer in <answer> and </answer> tags.

Document: [DOCUMENT]
Claim: [CLAIM]
"""