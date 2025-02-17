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
import re
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
        retrieval_cost: float = -0.2,
        retrieval_engine: str = "bing",
        prompt_truncation_side: str = "left",
        search_num: int = 1,
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
        for sample, weight in samples: 
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

        self.prompt_input_retrieval = "You will be presented with a question. If you know the answer, please respond directly. If you don't know the answer, use the Bing search engine to find the necessary information and then answer the question based on your observation. \n\nQuestion: {input}\n\nObservation: {retrieval}\n\nPlease format your output as follows:\n\n1. If you choose to answer the question directly, please use: \"[Answer] YOUR_ANSWER\"\n2. If you choose to use the Bing search engine, please use: \"[Search] YOUR_SEARCH_QUERY\"\n\nPlease output:\n"

        self.retrieval_cost = retrieval_cost
        self.retrieval_engine = retrieval_engine
        self.search_num = search_num
        
            
    def bing_search(self, topic):
        
        k = self.search_num
        URL_BING = "https://www.bingapis.com/api/v7/search?q={query}&appid=yourapi&count="

        url_bing = URL_BING.format(query=topic) + str(k)
        old_time = time.time()
        response = requests.get(url_bing, timeout=3)
        response_json = response.json()

        search_time = time.time() - old_time
        res_text = []
        web_pages = response_json.get('webPages', {}).get('value', [])
        for page  in web_pages:
            try:
                single_data = page['name'] + ": " + page['snippet']
                
                res_text.append(single_data)
                try:
                    for link in page.get('deepLinks', []):
                        single_data = link['name'] + ": " + link['snippet']
                        res_text.append(single_data)
                except:
                    pass
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

    def matchv3(self, input_str):
        pattern = r"\n\nQuestion: \s*(.*?)\s*\n\nPlease format"
        match = re.search(pattern, input_str)

        if match:
            extracted_content = match.group(1)
        else:
            extracted_content = ""
        return extracted_content

    def get_first_word_and_remaining(self, sentence):
        parts = sentence.split(maxsplit=1)
        first_word = parts[0]
        remaining_sentence = parts[1] if len(parts) > 1 else ""
        return first_word, remaining_sentence.strip()

    
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
            predicted = self.__current_obs.context_text.replace("[PAD]", "")

            # print("input question is: ", self.__current_obs.prompt_or_input_text)
            print("predicted sentence is: ", predicted)
            try:
                answer_query, remaining_sentence = self.get_first_word_and_remaining(predicted)
            except:
                answer_query = ""
                remaining_sentence = ""
                # print("predicted sentence is: ", predicted)
                print("predict format error")

            if answer_query == "[Answer]":

                self.__current_obs.context_text = remaining_sentence
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


            elif answer_query == "[Search]":

                # retrieve_text, search_time = self.bing_search(remaining_sentence)
                try:
                    retrieve_text, _ = self.bing_search(remaining_sentence)

                except:
                    retrieve_text = ""
                    search_time = 0
                    print(remaining_sentence + "could not be searched")

                reward = self.retrieval_cost
                
                sample = Sample(id=f"retrieval_{self.__time_step}",
                        prompt_or_input_text=self.prompt_input_retrieval.format(input=self.matchv3(self.__current_obs.prompt_or_input_text), retrieval=retrieve_text),
                        references=[self.__current_obs.target_or_reference_texts[0]]
                        )
                action_history = self.__current_obs.action_history
                self.__current_obs = self.reset_retrieval(sample)
                self.__current_obs.action_history = action_history

                answer_done = False

            else:
                answer_done = True
                reward = -1 

            print("reward is: ", reward)
            
        else:
            answer_done = False
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
