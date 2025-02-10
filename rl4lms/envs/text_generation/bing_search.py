import requests
import os
import time
from typing import Any
import json

os.environ['OPENAI_API_KEY'] = '082b19bc29364b1bb39d2d9fb9b757d4'
os.environ['OPENAI_API_TYPE'] = 'azure'
# os.environ['OPENAI_API_VERSION'] = '2023-05-15'
# os.environ['OPENAI_API_VERSION'] = '2023-06-01-preview'
os.environ['OPENAI_API_VERSION'] = '2023-09-01-preview'

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

def load_json(filename):
    # data = []
    # 按行读取并解析JSON数据
    with open(filename, 'r') as file:
        # for line in file:
        #     json_object = json.loads(line)
        #     data.append(json_object)
        data = json.load(file)
    return data

def save_json(data, filename):

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

    return 

class LLM:
    def __init__(self, folder_name='test', path='/mnt/workspace/user/gaojingsheng/LLM/OpenAI/logs'):
        self.folder_name = folder_name
        self.log_file = self.get_log_file(path)

    def get_log_file(self, path):
        run_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        folder = os.path.join(path, self.folder_name)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f'{run_date}.json')

    def save_logs(self, system_message, prompt, output):
        log = {}
        log['system'] = system_message
        log['input'] = prompt
        log['output'] = output
        with open(self.log_file, 'a') as f:
            json.dump(log, f, ensure_ascii=False)
            f.write('\n')

class langchain_LLM(LLM):
    def __init__(self,  
                temperature = 0,
                max_tokens = 250,
                model_name = "gpt-35-turbo",
                folder_name = 'retrieval',
                path = '/mnt/workspace/user/gaojingsheng/LLM/OpenAI/logs'
        ) -> None:
        path = f'{path}/{model_name}'
        super().__init__(folder_name, path)
        deployment_name = model_name
        print(model_name)
        self.prompt_template = "Please strictly summarize the content related to the Query based on the information above directly. No additional descriptions are needed, and please do not answer in bullet points. Keep it within 200 words. Query: {}. Summary: "
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            model_name=model_name,
            temperature=temperature,
            azure_endpoint='https://new-llm.openai.azure.com/',
        )
        
    def __call__(self, query='', system_message = 'You are ChatGPT, a large language model trained by OpenAI.', stop=None) -> Any:
        retrieval_result = self.bing_search(query)
        gpt4_input = self.prompt_template.format(retrieval_result).replace('\"', '')
        
        messages = [
            SystemMessage(content=system_message)
        ]
        messages.append(
            HumanMessage(
                content=gpt4_input
            )
        )
        output = self.llm(messages, stop=stop).content 
        super().save_logs(system_message, query, output)
        return output
    
    def bing_search(self, query):
        url_bing = "http://8.130.105.10:58000/api/v1/bing_api?only_search=1&query=" + query
        try:
            response = requests.get(url_bing, timeout=3)
            response_bing = response.json()
            
            retrieval_results = response_bing['result']
        
        except Exception:
            print("error")
            # logger.error('请求bingsearch失败！')
            # logger.error(traceback.format_exc())
            retrieval_results = [""]

        return str(retrieval_results)

def bing(topic):
    url_bing = "http://8.130.105.10:58000/api/v1/bing_api?only_search=1&query=" + topic
    try:
        response = requests.get(url_bing, timeout=3)
        response_bing = response.json()
        
        return response_bing['result']
    
    except Exception:
        print("error")
        # logger.error('请求bingsearch失败！')
        # logger.error(traceback.format_exc())
        return []

# def process_bing(input_str):
    
#     return 


def bing_search(topic):
    
    k = 4
    
    URL_BING = "https://www.bingapis.com/api/v7/search?q={query}&appid=371E7B2AF0F9B84EC491D731DF90A55719C7D209&mkt=zh-CN&offset=0&count="

    url_bing = URL_BING.format(query=topic) + str(k)
    old_time = time.time()
    # try:
    response = requests.get(url_bing, timeout=3)
    response_json = response.json()
    # except:
    #     print(topic + " could not be found")
    #     response_json = None

    search_time = time.time() - old_time
    # print(response_json)
    res_text = []
    web_pages = response_json.get('webPages', {}).get('value', [])
    # print(web_pages)
    for page  in web_pages:
        # print("page is: ", page)
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
    
    return obs
if __name__ == "__main__":
    # bing_gpt4 = langchain_LLM(path = '/mnt/workspace/user/gaojingsheng/LLM/OpenAI/logs', model_name='gpt4')
    # prompt_template = "Please strictly summarize the content related to the Query based on the information above directly. No additional descriptions are needed, and please do not answer in bullet points. Keep it within 200 words. Query: {}. Summary: "
    # question = "When is it revealed that Serena was drugged in the storyline?"
    # retrieval_result = bing(question)[0]
    # output =  gpt4(prompt_template.format(retrieval_result)).replace('\"', '')
    # print(output) 
    print(bing_search("When is it revealed that Serena was drugged in the storyline?"))