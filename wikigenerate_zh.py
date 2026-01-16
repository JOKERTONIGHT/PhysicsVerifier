import json
import openai
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('/data1/zengzheni/save_checkpoint/bge-large-zh-v1.5/')

def chatgpt(content, model='gpt-4o', cont_dics=None):
    #time.sleep(1)
    try:
        if cont_dics is not None:
            messages = cont_dics
        else:
            messages=[
                {"role": "user", "content": content}
            ]
        response = openai.ChatCompletion.create(
            #engine=engine,#gpt35
            model=model,
            messages=messages,
            max_tokens=2048,#512
            stop=None,
        )
        response = response['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        #pdb.set_trace()
        response = ''
    print(response)
    return response

'''
with open('original_table.json','r',encoding='utf-8') as g :
    directory=json.load(g)

with open('diabetes_content.json', 'r', encoding='utf-8') as file:
    articles = [json.loads(line) for line in file]
'''



def split(obj,max_length):
    titles=list(obj.keys())
    for key,value in obj.items():
        if isinstance(value, dict):
            split(obj[key],max_length)
        if isinstance(value, str):
            if(len(value)>max_length):
                if(key=="其它"):
                    #split_prompt=f"""You will receive a lot of texts related to diabete. Each text begins with a number in brackets as its serial number. Your task is to classify these texts and come up with a separate title for each class. Please output a title and the serial number of the corresponding texts on each line. The title and serial number are separated by '//', and multiple serial numbers are separated by ','. \nSample output: Output value growth//2,3,5\nGreen agriculture//0,1\nNote: 1. The output must comply with the format requirements and wrap after the serial number set of each title; 2. You can divide the texts into NO MORE THAN 4 classes. \nBelow are multiple texts: \n{value}\nPlease answer the classification results directly in the required format without any extra words."""
                    split_prompt = f"""你将看到一些与火箭推进剂相关的专业文本。每一段文本以中括号里的数字作为其序号开头。你的任务是将这些文本分类并拟定对应的标题。请你在每行输出一个标题以及文本对应的序号。标题和序列号应该以'//'隔开，多个序号应以','隔开。\n示例输出：产值增长//2,3,5\n绿色农业//0,1\n注意：1. 输出必须符合格式要求，将所有序列号都匹配到一个标题之后；2. 你最多可以将所有文本划分为4个标题。\n下面是多句专业文本：\n{value}\n请你直接按照格式要求回答你的分类结果，不要说多余的话。"""
                    messages=[{"role":"user","content":split_prompt}]
                    gpt_answer=chatgpt("", cont_dics=messages)
                   # print(gpt_answer)
                    classes=gpt_answer.replace("\n\n","\n").split("\n")
                    if(len(classes)==1):
                        return 
                    articles=value.split("\n")
                    for new_class in classes:
                        parts=new_class.split("//")
                        new_title=parts[0]
                        numbers=parts[1].split(",")
                        new_content=""
                        for number in numbers:
                            prefix="["+number+"]"
                            for article in articles:
                                if article.startswith(prefix):
                                    new_content+=article
                                    new_content+="\n"
                        obj[new_title]=new_content
                    obj["others"]=""
                    break
                else:
                    #split_prompt=f"""You will receive a lot of texts related to {key}. Each text begins with a number in brackets as its serial number. Your task is to classify these texts and come up with a separate title for each class. Please output a title and the serial number of the corresponding texts on each line. The title and serial number are separated by '//', and multiple serial numbers are separated by ','. \nSample output: Output value growth//2,3,5\nGreen agriculture//0,1\nNote: 1. The output must comply with the format requirements and wrap after the serial number set of each title; 2. You have to divide the text into more than 1 classes. 3. You can divide the texts into NO MORE THAN 4 classes. 4.Each text can only have one category.\nBelow are multiple texts: \n{value}\nPlease answer the classification results directly in the required format without any extra words."""
                    split_prompt = f"""你将看到一些与火箭推进剂相关的专业文本。每一段文本以中括号里的数字作为其序号开头。你的任务是将这些文本分类并拟定对应的标题。请你在每行输出一个标题以及文本对应的序号。标题和序列号应该以'//'隔开，多个序号应以','隔开。\n示例输出：产值增长//2,3,5\n绿色农业//0,1\n注意：1. 输出必须符合格式要求，仅在某类所有序号之后转行；2. 你应该将所有文本划分在多个类，但是最多可以将划分为4个标题；3. 每句文本只能归属于一个类。\n下面是多句专业文本：\n{value}\n请你直接按照格式要求回答你的分类结果，不要说多余的话。"""
                    messages=[{"role":"user","content":split_prompt}]
                    gpt_answer=chatgpt("", cont_dics=messages)
                    classes=gpt_answer.replace("\n\n","\n").split("\n")
                    if (len(classes)==1):
                        return 
                    new_dict={}
                    articles=value.split("\n")
                    print(classes)
                    for new_class in classes:
                    
                        parts=new_class.split("//")
                        new_title=parts[0]
                        numbers=parts[1].split(",")
                        new_content=""
                        for number in numbers:
                            prefix="["+number+"]"
                            for article in articles:
                                if article.startswith(prefix):
                                    new_content+=article
                                    new_content+="\n"
                        new_dict[new_title]=new_content
                    obj[key]=new_dict




def merge(obj,max_num):
    if isinstance(obj, str):
            return
    titles=list(obj.keys())
   # print(titles)
    if "其它" in titles:
        titles.remove("其它")
    for key,value in obj.items():
        if isinstance(value, dict):
            merge(obj[key],max_num)
    
    if(len(titles)>max_num):
        number_list = [f"[{i}]{string}" for i, string in enumerate(titles)]
        #merge_prompt=f"""You will receive a lot of sub-headings related to {key}. Each sub_heading begins with a number in brackets as its serial number. Your task is to classify these sub-headings and come up with a separate heading for each class. Please output a heading and the serial number of the corresponding sub-headings on each line. The title and serial number are separated by '//', and multiple serial numbers are separated by ','. \nSample output: Output value growth//2,3,5\nGreen agriculture//0,1\nNote: 1. The output must comply with the format requirements and wrap after the serial number set of each title; 2. You can divide the sub-headings into NO MORE THAN 4 classes. \nBelow are multiple sub_headings: \n{number_list}\nPlease answer the classification results directly in the required format without any extra words."""
        merge_prompt=f"""你将见到多个与{key}相关的副标题。每一个标题开头中括号里的数字是它的序号。你的任务是将这些副标题分类并为每个类拟定一个新的主标题。请在每行输出一个主标题和其对应的副标题序号。标题和序号应该以'//'隔开，多个序号之间应该以','隔开。\n示例输出：产值增长//2,3,5\n绿色农业//0,1\n注意：1. 你的输出必须满足格式要求，仅在某类的所有副标题的序号后转行；2. 你可以将这些副标题归总为最多4类。\n下面是多个副标题：\n{number_list}\n请你按照格式要求直接输出你的分类结果，不要说多余的话。"""
        messages=[{"role":"user","content":merge_prompt}]
        gpt_answer=chatgpt('',cont_dics=messages)
        new_titles=gpt_answer.replace("\n\n","\n").split("\n")
        for title in new_titles:
            new_dict={}
            parts=title.split("//")
            new_title=parts[0]
            numbers=parts[1].split(",")
            if len(numbers)==1:
                continue
            else:
                for number in numbers:
                    num=int(number)
                    old_title=titles[num]
                    new_dict[old_title]=obj[old_title]
                    del obj[old_title]
                obj[new_title]=new_dict
    

            
def search(obj,text,num):
    titles=list(obj.keys())
    if isinstance(obj, str):
        return
    if isinstance(obj, dict):
        #select_title_prompt=f"""You will receive a section heading library and a text for diabetes trials. Please follow the steps below: 1. Understand the text and find the heading that match the text from the section heading library. 2. Summarize the content from the text that is related to the heading. 3. Each output must contain two parts: 'subsection heading&&Summary content'. 4. If there is no any heading that matches the text, output 'None'. \n Sample output1: Symptoms of COVID-19&&A small number of patients with COVID-19 may experience severe headaches. \n Sample output2:  Oral medication&&The trial found that the combination of Nirmatrelvir and Ritonavir 2 was significantly better than using them separately in relieving symptoms such as breathing difficulties and severe headaches caused by COVID-19. \nNote: 1. Output must be in the required format, and only wrap after the summary content; 2. Select headings that are most related to the text, otherwise output 'None' when there is no any related heading;  \nHeading library: {titles}\nText: {text}\n\nPlease direct answer without extra words.""" 
        select_title_prompt=f"""你将看到一个标题库和与火箭推进剂相关的一段文本。请你按照如下步骤操作：1. 理解文本并从标题库中找到其最匹配的标题；2. 将文本中与该标题相关的内容进行简要总结；3. 每个输出应该包含两个部分：'标题&&内容总结'；4. 如果没有任何标题与当前文本相匹配，则输出'None'。\n示例输出1：推进剂燃烧过程&&推进剂在燃烧室内完全燃烧，各产物处于热平衡和化学平衡状态，燃烧产物的流动膨胀过程简化为理想的等熵流动过程，喷管进口处和出口处的能量守恒。\n示例输出2：推进剂性能参数&&推进剂比冲的大小直接取决于燃烧室内热焓与喷管排气热焓之差，比冲值反映了推进系统所能提供能量的大小。\n注意：1. 输出必须满足格式要求，仅在某标题内容总结后转行；2. 选择与文本最为匹配的标题，如果没有任何相关标题时才可输出'None'。\n标题库如下：{titles}\n文本如下：{text}\n\n请你直接输出答案，不要说任何多余的话。"""
        messages=[{"role":"user","content":select_title_prompt}]
        title=chatgpt('', cont_dics=messages)
        print(title)
        ##生成新的标题
        if(title=="None"or "None" in title):
            #summary_prompt=f"""Please generate a summary based on the text provided.\n Sample output: \n The trial found that the combination of Nirmatrelvir and Ritonavir 2 was significantly better than using them separately in relieving symptoms such as breathing difficulties and severe headaches caused by COVID-19.\nText: {text}\n\nPlease directly respond with your summary without any redundant words."""
            summary_prompt=f"""请你根据所提供的文本生成一段总结，一两句话即可。文本：{text}\n请你直接回复你的总结，不要说多余的话。"""
            messages=[{"role":"user","content":summary_prompt}]
            summary=chatgpt('', cont_dics=messages)
            summary="["+str(num)+"]"+summary+"\n"
            if("其它" in titles):
                obj["其它"]+=summary
            else:
                obj["其它"]=summary
        else:
            title=title.replace("\n","")
            line=title.split("&&")
            name=line[0]
            summary="["+str(num)+"]"+line[1]+"\n"
            obj_=obj[name]      
            if isinstance(obj_, dict):
                search(obj[name],text,num)
            if isinstance(obj_, str):
                obj[name]+=summary

                
def merge_similar_keys(data, similarity_threshold=0.8):
    """
    遍历嵌套字典，合并字符串相似的叶子结点键的值到第一个键，并删除其他相似的键。

    :param data: 待处理的嵌套字典
    :param similarity_threshold: 相似度阈值，介于0到1之间，默认值为0.8
    """
    key_locations = {}  # 记录已处理的键及其代表键

    def are_keys_similar(key1, key2):
        # 计算两个键的相似度
        embeddings_a = model.encode(key1,convert_to_tensor=True).unsqueeze(0)
        embeddings_b = model.encode(key2,convert_to_tensor=True).unsqueeze(0)
        similarity=torch.nn.functional.cosine_similarity(embeddings_a, embeddings_b,dim=-1)
        return similarity >= similarity_threshold

    def find_representative_key(key):
        # 查找已有的相似键的代表键
        for rep_key in key_locations:
            if are_keys_similar(key, rep_key):
                print(key+","+rep_key+"\n")
                return rep_key
        return None

    def traverse(d):
        keys_to_delete = []
        for key in list(d.keys()):
            if(key=="其它"):
                continue  ##跳过others
            value = d[key]
            if isinstance(value, dict):
                traverse(value)  # 递归遍历子字典
            else:
                rep_key = find_representative_key(key)
                if rep_key:
                    # 如果找到相似的代表键，合并值
                    first_parent, first_key = key_locations[rep_key]
                    first_parent[first_key] += value  # 根据需要更改合并方式
                    keys_to_delete.append(key)  # 标记当前键待删除
                else:
                    # 记录新键的首次出现位置
                    key_locations[key] = (d, key)
        # 删除标记的键
        for key in keys_to_delete:
            del d[key]

    traverse(data)


def find_value_by_key(data, target_key):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                return value
            elif isinstance(value, (dict, list)):
                result = find_value_by_key(value, target_key)
                if result is not None:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = find_value_by_key(item, target_key)
            if result is not None:
                return result
    return None  # 如果没有找到，返回 None


def process(d,nested_dict):
    for key, value in d.items():
        if isinstance(value, dict):
            # 如果值是字典，递归调用
            process(value,nested_dict)
        elif isinstance(value,str):
            # 如果值是数字类型，执行加1
            d[key]=find_value_by_key(nested_dict,key)
    return d
        # 如果值不是字典或数字，则不做操作

def clear_dict_values(d):
    """
    递归将嵌套字典的所有值替换为空字符串 ""。
    :param d: 输入的嵌套字典
    :return: 替换值后的字典
    """
    for key, value in d.items():
        if isinstance(value, dict):
            clear_dict_values(value)  # 如果值是字典，递归调用自身
        else:
            d[key] = ""  # 将非字典的值替换为空字符串
    return d

def remove_others_key(data):
    if isinstance(data, dict):
        # Create a list of keys to delete to avoid modifying the dictionary during iteration
        keys_to_delete = [key for key in data if (key == "others") or (key == "Other issues for diabetes") or data[key]==""]
        # Remove the 'others' keys
        for key in keys_to_delete:
            del data[key]

        # Recursively call the function for nested dictionaries
        for key, value in data.items():
            remove_others_key(value)

    elif isinstance(data, list):
        # If the value is a list, apply the function to each item
        for item in data:
            remove_others_key(item)

    return data


def reorganize(data):
    directory=clear_dict_values(data)
    directory=remove_others_key(directory)
    prompt=f"""Given a nested json directory, remove duplicates and merge the directory, reorganize the directory structure, and output a new json directory.
directory:{directory}
Please directly respond without any redundant words. Make sure the leaf nodes of the json directory are not changed."""
    prompt=f"""给定一个交联的json字典，你需要把冗余的部分去除或相互合并，重新整理字典结构，并输出新的json字典。\n字典：{directory}请你直接输出json字典，不要有任何多余的话。你需保证json字典的叶节点名称没有改变。"""
    messages=[{"role":"user","content":prompt}]
    new_d=chatgpt('', model='gpt-5-chat', cont_dics=messages)
    new_d=new_d.replace("```json","").replace("```","")
    new_d=json.loads(new_d)
    directory=process(new_d,data)
    print(new_d)
    return directory



with open('tjj0.json','r',encoding='utf-8') as g :
    directory=json.load(g)

books = json.load(open('book_segments.json'))
ranks = json.load(open('rank_dic_more.json'))

ranks0 = json.load(open('rank_dic.json'))
banned = []
for ky in ranks0.keys():
    banned+=ranks[ky][:3]
banned1 = []
for st in banned:
    banned1.append(st)
    if st-1 not in banned:
        banned1.append(st-1)
    if st+1 not in banned:
        banned1.append(st+1)

srts = []
for ky in ranks.keys():
    srts+=ranks[ky][39:42]
srts1 = []
for st in srts:
    if st in banned1:
        continue
    srts1.append(st)
    if st-1 not in srts:
        srts1.append(st-1)
    if st+1 not in srts:
        srts1.append(st+1)
import numpy as np
srts = np.sort(srts1)
print(srts, len(srts))
import pdb

articles = {}
for idx,st in enumerate(srts):
    articles[idx] = {'content':books[str(st)], 'id':st}
    #texts.append('【'+str(st)+'】'+books[str(st)])

def generate_wiki(directory, articles):
    for i in range(0,len(articles)):
        article=articles[i]
        text=article["content"]
        num=article["id"]
        try:
            search(directory,text,num)
        except Exception as exc:
            error_message = f"Error: {str(exc)}"
            print(num)
            print(error_message)
            continue
            

        if(i%20==0):
            name=f"""output_14/step_{article["id"]}.json"""
            with open(name, 'w') as file:
                json.dump(directory, file, indent=4, ensure_ascii=False)

            if (i%40!=0):
                continue
            split(directory,2400)
            name1=f"""output_14/split_{article["id"]}.json"""
            with open(name1, 'w') as file:
                json.dump(directory, file, indent=4, ensure_ascii=False)
    

            merge_similar_keys(directory, similarity_threshold=0.9)
            name3=f"""output_14/unrepeat_{article["id"]}.json"""
            with open(name3, 'w') as file:
                json.dump(directory, file, indent=4, ensure_ascii=False)

            merge(directory,15)
            name2=f"""output_14/merge_{article["id"]}.json"""
            with open(name2, 'w') as file:
                json.dump(directory, file, indent=4, ensure_ascii=False)
        
     #   if(article["id"]==1001):
      #      directory=reorganize(directory)
       #     name2=f"""output7/reorganize_{article["id"]}.json"""
       #     with open(name2, 'w') as file:
       #         json.dump(directory, file, indent=4)
        
            
generate_wiki(directory, articles)
