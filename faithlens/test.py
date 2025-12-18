import os
os.environ['NLTK_DATA'] = './nltk'
from faithlens.inference import FaithLensInfer
import json

       
if __name__ == "__main__":
    detection_model = FaithLensInfer(model_name="/mnt/public/share/users/sishuzheng-share/minicheck_models/llama3.1_reason_rl_models-from3checkpoint_from3dataset_reason_reward_rererun/global_step_100_hf", device="cuda:1")
    single_result = detection_model.infer(
        docs=["Relegation-threatened Romanian club Ceahlaul Piatra Neamt have sacked Brazilian coach Ze Maria for the second time in a week. Former Brazil defender Ze Maria was fired on Wednesday after a poor run, only to be reinstated the next day after flamboyant owner Angelo Massone decided to 'give the coaching staff another chance.' But the 41-year-old former Inter Milan and Parma right back, capped 25 times by Brazil, angered Massone again after Ceahlaul were beaten 2-0 by mid-table FC Botosani on Saturday. Ze Maria represented Brazil on 25 occasions during an international career spanning five years . The result left Ceahlaul 16th in the standings, six points adrift of safety. Ze Maria replaced Florin Marin in January to become Ceahlaul's third coach this season. He will be replaced by Serbian Vanya Radinovic."],
        claims=["Former brazil defender ze maria was fired on wednesday after a poor run. The 41-year-old was reinstated the next day after flamboyant owner angelo massone decided to'give the coaching staff another chance' but the 41-year-old angered massone again after ceahlaul were beaten 2-0 by mid-table fc botosani on saturday."],
    )
    print("Single Result:")
    print(json.dumps(single_result, ensure_ascii=False, indent=2))