SYSTEM_PROMPT = """Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. Please assess the claim's consistency with the document by responding with either "Yes" or "No"."""


USER_PROMPT = """Document: [DOCUMENT]\nClaim: [CLAIM]"""



USER_PROMPT_OURS = """Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. 

- First, think step by step about whether all the information in the claim is fully supported by the document within <think> and </think> tags. 

- Then, please provide an easy-to-understand explanation for your answer within <reason> and </reason> tags.

- Finally, assess the claim's consistency with the document by responding with either "Yes" or "No" and wrap your final answer in <answer> and </answer> tags.

Document: [DOCUMENT]
Claim: [CLAIM]
"""




# USER_PROMPT_OURS = """Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. First, think step by step about whether all the information in the claim is fully supported by the document. Then, please assess the claim's consistency with the document by responding with either "Yes" or "No". Please wrap your final answer in <answer> and </answer>.\nDocument: [DOCUMENT]\nClaim: [CLAIM] """


# USER_PROMPT_OURS = """
# Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. 

# First, reference each relevant sentence from the document individually: 
# - For every sentence that relates to the claim (either supporting, contradicting, or being part of the relevant context), restate its key information in your own words and append its corresponding <cite>[Cx-Cx]</cite> tag (use [C_start-C_end], where start and end are the same for a single sentence). 
# - You may group multiple consecutive sentences into a single citation range if they express one coherent idea (e.g., <cite>[C3-C5]</cite>).
# Then, explain how these referenced parts support or contradict the claim. Clearly indicate whether the evidence aligns with or opposes the claim, and why.
# Please wrap your above content in <think> and </think>.


# Finally, please assess the claim's consistency with the document by responding with either "Yes" or "No". 
# Please wrap your final answer in <answer> and </answer>.

# Document: [DOCUMENT]
# Claim: [CLAIM]
# """
