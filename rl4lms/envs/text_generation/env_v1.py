from cmath import inf
from typing import Dict, Tuple, Optional, List

import torch
from gym import Env, spaces
from gym.spaces.dict import Dict as DictSpace
from gym.spaces.discrete import Discrete
from rl4lms.data_pools.text_generation_pool import Sample
from rl4lms.envs.text_generation.reward import BatchedRewardFunction, RewardFunction
from rl4lms.envs.text_generation.observation import Observation
from transformers import AutoTokenizer
from rl4lms.core_components.sampler import PrioritySampler
import json
from rl4lms.envs.text_generation.passage_retrieval import Retriever
import re
from .choiceeval import hits
from .bing_search import langchain_LLM
import time
import requests

class TextGenEnv(Env):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        reward_function: RewardFunction,
        samples: Tuple[List[Sample], float],
        max_episode_length: int = 512,
        priority_scale: float = 0.0,
        max_prompt_length: Optional[int] = None,
        terminate_on_eos: bool = False,
        context_start_token: Optional[int] = None,
        prompt_truncation_side: str = "left",
    ):
        """
        A generic RL environment to generate textual sequences.
        For eg: text generation, summarization, machine translation, text simplification
        Args:
            tokenizer (AutoTokenizer): pre-trained tokenizer
            reward_function (RewardFunction): reward functiom
            samples (Tuple[List[Sample], float]): list of samples
            max_episode_length (int, optional): Max steps to the model Defaults to 512.
            priority_scale (float, optional): weight for the priority sampler Defaults to 0.0.
            max_prompt_length (Optional[int], optional): maximum prompt length. Defaults to None.
            terminate_on_eos (bool, optional): whether to terminate on EOS. Defaults to False.
            context_start_token (bool, optional): start token for the context (For Encoder-Decoder models! )
            prompt_truncation_side (str): truncation side for prompt text (Defaults to "left")
        """
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.max_steps = max_episode_length
        self._max_text_length = (
            max_prompt_length if max_prompt_length else tokenizer.model_max_length
        )
        self._terminate_on_eos = terminate_on_eos
        self._context_start_token = context_start_token
        self._prompt_truncation_side = prompt_truncation_side
        super().__init__()

        # set the observation and action space here
        self._vocab_size = tokenizer.vocab_size
        self.observation_space = DictSpace(
            {
                # we have to provide fixed sized inputs (padded) because sb3 support for DictObsersevation is limited
                # while creating rollout buffers, observations are concatenated for each key
                "prompt_or_input_encoded_pt": spaces.Box(
                    low=0, high=self._vocab_size, shape=(self._max_text_length,)
                ),
                "prompt_or_input_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(self._max_text_length,)
                ),
                "context_encoded_pt": spaces.Box(
                    low=0, high=self._vocab_size, shape=(self.max_steps,)
                ),
                "context_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(self.max_steps,)
                ),
                "input_encoded_pt": spaces.Box(
                    low=0,
                    high=self._vocab_size,
                    shape=(self._max_text_length + self.max_steps,),
                ),
                "input_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(self._max_text_length + self.max_steps,)
                ),
            }
        )
        self.action_space = Discrete(n=self._vocab_size)
        # see https://github.com/huggingface/transformers/issues/4875 : rounding up to nearest power of 2 for better GPU efficiency
        if 'mt5' in self.tokenizer.name_or_path:
            n = 250112
            self.action_space = Discrete(n=n)
        elif 't5' in self.tokenizer.name_or_path:
            n = 32128
            self.action_space = Discrete(n=n)
        self.sampler_for_replaying = PrioritySampler(priority_scale=priority_scale)
        for sample, weight in samples: # all training data are add
            # sample是一个Sample类, 里面有prompt_or_input_text是输入，reference是answer；weight都是1.0
            # print(sample)
            self.sampler_for_replaying.add(sample, weight)

        # check the tokenizer and add padding tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # TBD: configure this
        self.tokenizer.truncation_side = "left"  # TBD: configure this

        # init tracking variables
        self.__current_sample = None
        self.__current_obs = None
        self.__time_step = None

        # self.retriever = Retriever()
        # self.retriever.setup_retriever()
        self.bing_gpt4 = langchain_LLM(path = '/mnt/workspace/user/gaojingsheng/LLM/OpenAI/logs', model_name='gpt4')
        
        # self.question_query_prompt = "Please first determine whether the following question requires additional relevant knowledge retrieval. If so, rewrite the question to better fit retrieval query. Output in the format: {'retrieval': 'yes', 'query': query}. If no retrieval is needed, please answer the question directly, in the format {'retrieval': no, 'answer': answer}. The question is: ".lower()
        # self.question_query_prompt_prefix = "Please first determine whether answering the following question requires additional relevant knowledge. If so, rewrite the question to better fit retrieval query. Output in the format: {'query': **}. If no additional knowledge is needed, please answer the question directly, output in the format '{'answer': **}'. Question: " 
        # self.question_query_prompt_suffix = " Output: " 
        # self.retrieval_answer_prompt = "Please answer this question in the format of '{'answer': **}', you can refer to relevant knowledge: {} The question is: {}"
        
    def bing_search(self, topic):

        url_bing = "http://8.130.105.10:58000/api/v1/bing_api?only_search=1&query=" + topic
        # try:
        old_time = time.time()
        response = requests.get(url_bing, timeout=3)
        response_json = response.json()
        search_time = time.time() - old_time

        res_text = []

        # print(response_json.keys()) # dict_keys(['code', 'msg', 'exactqa', 'result'])
        for item in response_json['result']:
            try:
                single_data = item['title'] + ": " + item['snippet']
                res_text.append(single_data)
            except:
                pass
        obs= " ".join(res_text)
        return obs, search_time

    def step_origin(self, action: int) -> Tuple[Dict[str, torch.tensor], int, bool, dict]:
        self.__time_step += 1

        # previous obs
        previous_obs = self.__current_obs

        # just update the context tensor and gets the new observation
        self.__current_obs = self.__current_obs.update(action, self.tokenizer)

        # decide if the episode is finished or not
        done = (action == self.tokenizer.eos_token_id and self._terminate_on_eos) or (
            self.__time_step == self.max_steps
        )

        # compute reward
        if not isinstance(self.reward_function, BatchedRewardFunction):
            reward = (
                None
                if self.reward_function is None
                else self.reward_function(
                    previous_obs,
                    action,
                    self.__current_obs,
                    done,
                    self.__current_obs.meta_info,
                )
            )
        else:
            reward = -inf  # will be overridden later

        # populate additional info
        info = {
            "output": self.__current_obs.context_text,
            "action_history": self.__current_obs.action_history,
            "reference_text": self.__current_obs.target_or_reference_texts,
            "prompt_text": self.__current_obs.prompt_or_input_text,
            "prev_output": previous_obs.context_text,
            "meta_info": previous_obs.meta_info,
        }

        return self.__current_obs.to_dict(), reward, done, info

    def match(self, input_str):
        # Pattern to find the content between "Question:" and "Output:"
        pattern = r"Question:\s*(.*?)\s*Output:"
        match = re.search(pattern, input_str)

        if match:
            extracted_content = match.group(1)
        else:
            extracted_content = ""
        return extracted_content

    def is_valid_format(self, input_str):
        try:
            input_dict = json.loads(input_str)
            if "retrieval" in input_dict and ("answer" in input_dict or "query" in input_dict):
                return True
            else:
                return False
        except json.JSONDecodeError:
            return False
    
    def is_valid_format_v2(self, input_str):

        if "query:" == input_str[:6] or "answer:" == input_str[:7]:

            return True
        else:

            return False
            
    def retrieval_answer_prompt(self, knowledge, question):
        
        return "Please answer this question in the format of '{'answer': **}', you can refer to relevant knowledge: " + knowledge + ". The question is: " + question

    def step(self, action: int) -> Tuple[Dict[str, torch.tensor], int, bool, dict]:
        self.__time_step += 1

        # previous obs
        previous_obs = self.__current_obs

        # just update the context tensor and gets the new observation
        self.__current_obs = self.__current_obs.update(action, self.tokenizer)

        # decide if the episode is finished or not
        done = (action == self.tokenizer.eos_token_id and self._terminate_on_eos) or (
            self.__time_step == self.max_steps
        )
        if done:
            predicted = self.__current_obs.context_text
            print("predicted sentence is: ", predicted)
            if not self.is_valid_format_v2(predicted):
                answer_done = True
                # retrieval_done = False
                reward = -2
            else:
                # output_dict = json.loads(predicted)
                # print(predicted) 
                # if "answer" in output_dict:
                if "answer: " == predicted[:8]:
                    # self.__current_obs.context_text = output_dict["answer"]
                    self.__current_obs.context_text = predicted[8:]
                    reward = (
                        None
                        if self.reward_function is None
                        else self.reward_function(
                            previous_obs,
                            action,
                            self.__current_obs,
                            done,
                            self.__current_obs.meta_info, 
                        )
                    )
                    answer_done = True
                    # retrieval_done = False
                    print("reward is: ", reward)
                # elif output_dict["retrieval"] == "yes":
                elif "query: " == predicted[:7]:
                    # if "query" in output_dict:
                    # retrieve_document = self.retriever.search_document(output_dict["query"], 1)

                    # retrieve_document = self.retriever.search_document(predicted[7:], 1)
                    # retrieve_text = retrieve_document[0]["text"]
                    # retrieve_text = self.bing_gpt4(predicted[7:])

                    retrieve_text, search_time = self.bing_search(predicted[7:])

                    # sample = self.retrieval_answer_prompt.format(retrieve_text, self.__current_obs.prompt_or_input_text.replace(self.question_query_prompt,""))
                    # sample = self.retrieval_answer_prompt.format(retrieve_text, self.match(self.__current_obs.prompt_or_input_text))
                    # sample = self.retrieval_answer_prompt(retrieve_text, self.match(self.__current_obs.prompt_or_input_text))

                    if hits(self.__current_obs.target_or_reference_texts[0], retrieve_text, dn=0, dl=False) != 0:
                        # reward = -0.1
                        # reward = 1
                        # reward = 0.1
                        reward = 0 # -0.05
                    else:
                        # reward = -0.1
                        # reward = -0.5
                        # reward = 0.5
                        reward = -0.1
                        
                    # exachange the observation 
                    sample = Sample(id=f"retrieval_{self.__time_step}",
                           prompt_or_input_text=self.retrieval_answer_prompt(retrieve_text, self.match(self.__current_obs.prompt_or_input_text)),
                           references=[self.__current_obs.target_or_reference_texts[0]]
                           )
                    # print("sample reset to: ", sample)
                    # here maybe clear the action history
                    action_history = self.__current_obs.action_history
                    self.__current_obs = self.reset_retrieval(sample)
                    self.__current_obs.action_history = action_history

                    answer_done = False
                    # if action == self.tokenizer.eos_token_id:
                    #     retrieval_done = True
                    # else:
                    #     retrieval_done = False

                else:
                    answer_done = True
                    # retrieval_done = False
                    reward = -2 # -10
                    
        else:
            answer_done = False
            # retrieval_done = False
            reward = 0 
            
        # populate additional info
        info = {
            "output": self.__current_obs.context_text,
            "action_history": self.__current_obs.action_history,
            "reference_text": self.__current_obs.target_or_reference_texts,
            "prompt_text": self.__current_obs.prompt_or_input_text,
            "prev_output": previous_obs.context_text,
            "meta_info": previous_obs.meta_info,
        }
        
        return self.__current_obs.to_dict(), reward, answer_done, info


    def reset(self, sample: Sample = None) -> Dict[str, torch.tensor]:
        """
        Resets the environment and starts a new episode
        """
        # gets a new sample if not provided
        if sample is None:
            sample = self.sampler_for_replaying.sample(size=1)[0]
        # Sample: "please first determine whether the following question requires additional relevant knowledge retrieval. if so, rewrite the question to better fit retrieval query. output in the format: {'retrieval': 'yes', 'query': query}. if no retrieval is needed, please answer the question directly, in the format {'retrieval': no, 'answer': answer}. the question is: what is the name of the de versely family house? \\n  at madeline hall, an old mansion-house near southampton belonging to the wealthy de versely family, lives an elderly spinster miss delmar, the aunt of the earl de versely and captain delmar. miss delmar invites arabella mason, the daughter of a deceased, well-liked steward to stay with her as a lower-class guest in the house. captain delmar is known to visit his aunt at madeline hall frequently, accompanied by his valet ben keene, who is also a private marine. captain delmar eventually suggests that ben should propose to arabella, and the two marry in secret, to the frustration of miss delmar and arabella's mother. the captain is able to smooth over the situation with his aunt, even after it is discovered that arabella was six months pregnant at the time of the marriage. she later gives birth to a boy, who takes the captain's christian name and ben's surname--the titular percival keene.the family moves to chatham, after ben is ordered back with his detachment. arabella opens up a successful shop and circulating library below her house, enlisting the help of her mother and sister, amelia. percival becomes well known in town from his mischievous pranks on officers and other strangers, often encouraged by his aunt amelia. however, percival's mother and grandmother are less fond of his disregard for manners, and insist on sending him to school after an episode in which he bites his grandmother. percival reports to the school house of mr. o'gallagher, a poor irish scholar, who rules his class with a system of severe corporal punishment. mr. o'gallagher routinely bullies percival by stealing his lunch, leading perciv
        # print("sample is: ", sample)
        self.__current_sample = sample

        # init the observation
        self.__current_obs = Observation.init_from_sample(
            sample,
            self.tokenizer,
            self._max_text_length,
            self.max_steps,
            self._prompt_truncation_side,
            self._context_start_token,
            sample.meta_data,
        )

        # start the time step counter
        self.__time_step = 0

        dict_observation = self.__current_obs.to_dict()
        return dict_observation

    def reset_retrieval(self, sample: Sample = None) -> Dict[str, torch.tensor]:
        """
        Resets the environment and starts a new episode
        """
        # gets a new sample if not provided
        if sample is None:
            sample = self.sampler_for_replaying.sample(size=1)[0]
        # Sample: "please first determine whether the following question requires additional relevant knowledge retrieval. if so, rewrite the question to better fit retrieval query. output in the format: {'retrieval': 'yes', 'query': query}. if no retrieval is needed, please answer the question directly, in the format {'retrieval': no, 'answer': answer}. the question is: what is the name of the de versely family house? \\n  at madeline hall, an old mansion-house near southampton belonging to the wealthy de versely family, lives an elderly spinster miss delmar, the aunt of the earl de versely and captain delmar. miss delmar invites arabella mason, the daughter of a deceased, well-liked steward to stay with her as a lower-class guest in the house. captain delmar is known to visit his aunt at madeline hall frequently, accompanied by his valet ben keene, who is also a private marine. captain delmar eventually suggests that ben should propose to arabella, and the two marry in secret, to the frustration of miss delmar and arabella's mother. the captain is able to smooth over the situation with his aunt, even after it is discovered that arabella was six months pregnant at the time of the marriage. she later gives birth to a boy, who takes the captain's christian name and ben's surname--the titular percival keene.the family moves to chatham, after ben is ordered back with his detachment. arabella opens up a successful shop and circulating library below her house, enlisting the help of her mother and sister, amelia. percival becomes well known in town from his mischievous pranks on officers and other strangers, often encouraged by his aunt amelia. however, percival's mother and grandmother are less fond of his disregard for manners, and insist on sending him to school after an episode in which he bites his grandmother. percival reports to the school house of mr. o'gallagher, a poor irish scholar, who rules his class with a system of severe corporal punishment. mr. o'gallagher routinely bullies percival by stealing his lunch, leading perciv
        self.__current_sample = sample

        # init the observation
        self.__current_obs = Observation.init_from_sample(
            sample,
            self.tokenizer,
            self._max_text_length,
            self.max_steps,
            self._prompt_truncation_side,
            self._context_start_token,
            sample.meta_data, 
        )

        # start the time step counter
        self.__time_step = 0

        dict_observation = self.__current_obs
        return dict_observation

    def render(self):
        pass

    def close(self):
        pass

    def add_sample(self, sample: Sample, weight: int = 1.0):
        self.sampler_for_replaying.add(sample, weight)
