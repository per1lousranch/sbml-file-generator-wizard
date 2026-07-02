import ollama
import pymupdf
import numpy as np
import json
import os.path
import PySimpleGUI as sg
import time

# function for extracting paragraphs
# PARAMETERS:
# filenames: a list of strings which are filenames for information to be extracted from
def parse_file(filenames: list[str]):
    extracted_docs = [] 

    # iterating through filenames
    for file in filenames:
        doc = pymupdf.open(file) # PyMuPDF allows for text extraction via paragraphs

        # iterating through pages of an individual document
        for i in range(35): # upper limit so far: 36
            if i == 2 or i == 3: # NOTE: remove if not using the multi specification or any similar doc with hyperlinked contents pages
                continue # for some reason code crashes if trying to parse the contents pages of the multi specification

            page = doc[i]

            paragraph_lst = page.get_text("blocks") # 'blocks' parameter allows extraction based on paragraphs

            for lst in paragraph_lst:
                chunk = lst[4] # 5th index contains actual text, so we only append what's there

                if len(chunk) >= 56: # used to ensure only sentences and no titles are passed through, calculated from taking the average character per word of 4 with a lower bound of 14 words per sentences (4 x 14 = 56)
                    extracted_docs.append(chunk) # could also check if there are punctuation in the chunk?
    
    return extracted_docs

# function for getting embeddings of extracted text
# PARAMETERS:
# model_name: string which contains the name of the model to be used for embedding (must be installed via Ollama)
# paragraphs: list of strings which contains extracted paragraphs for embeddings to be created for
def get_embeddings(model_name: str, paragraphs: list[str]):
    embeddings = ''

    # if embeddings file already exists, no need to create again since they will be the smae
    if os.path.exists('embeddings.json'):
        with open('embeddings.json', 'r') as file: # instead just open the file and load them
            embeddings = json.load(file)
    else: # otherwise, create them from scratch
        batch = ollama.embed(model = model_name, input = paragraphs)

        embeddings = batch['embeddings'] # batch is a dictionary, the key 'embeddings' actually returns the embeddings

        # cache to save time in future iterations of program
        with open('embeddings.json', 'w') as file:
            json.dump(embeddings, file)

    return embeddings

# function for executing the chat
# PARAMETERS:
# model_name: string for the name of the converstaional model
# embedding_name: string for the name of the embedding model (calculating embedding for prompt)
# embeddings: list of list of ints which we will use to compare embedding of prompt against via cosine similarity
# paragraphs: list of strings that we will index to get information for model
def rag_continuous_chat(model_name: str, embedding_name: str, embeddings: list[list[int]], paragraphs: list[str]):
    system_prompt = '''You are an SBML Multi expert; first, search the information provided at the 
    end of this string and do not deviate from it to try and find the answer. If you cannot answer a question based 
    on the information, use your pre-trained knowledge to answer the question, but explicitly state that your answer is from 
    your own pre-trained knowledge. The information is here: '''
    message_list = [{'role': 'system', 'content': system_prompt}]

    layout = [
        [sg.Text(text = "RAG Chat Application")],
        [sg.Text("Chat: "), sg.Input(), sg.OK()],
        [sg.Multiline(key = 'output', size = (60, 30))]
    ]

    window = sg.Window(title = "RAG Chat Application", layout = layout, margins = (300, 150))

    while True:
        event, values = window.read()
        
        if event == sg.WIN_CLOSED:
            break
        else:
            # window['output'].update("Thinking... please wait...") # doesn't work???

            user_prompt = values[0]

            single = ollama.embed(model = embedding_name, input = user_prompt) # generate embedding for prompt
            single_embed = single['embeddings'][0] # only one embed, so we take the 1st item in the list which is the embed

            def cosine_similarity():
                single_norm = np.linalg.norm(single_embed)

                similarity = []

                # comparing user prompt embedding against embeddings of embeded chunks
                for i in range(len(embeddings)):
                    single_lst_norm = np.linalg.norm(embeddings[i])

                    dot_product = np.dot(single_embed, embeddings[i])
                    mult = single_norm * single_lst_norm
                    calc = dot_product / mult

                    similarity.append([calc, i])
                
                # closer the cosine similarity is to 1, the more similar, so sort in reverse
                similarity.sort(reverse = True)

                return similarity
            
            # return the best 10 chunks
            similarity = cosine_similarity()[:10]

            final = []
            # uncomment to see what context is getting passed into the model
            for item in similarity:
                print(item[0], item[1]) # item[1] is index
                print("Content: " + paragraphs[item[1]]) # since the indexes are the same for embeddings and paragraphs, we get the content from paragraphs
                final.append(paragraphs[item[1]]) # appending chunks final list to be appended to system prompt
            
            message_list[0]['content'] = system_prompt + " ".join(final) # adding chunks to system prompt
            message_list.append({'role': 'user', 'content': user_prompt}) # add user prompt message list
            response = ollama.chat(model = model_name, messages = message_list, options = {'temperature': 1, 'top_k': 64, 'top_p': 0.95}, stream = True) # get model's response

            str_response = ""

            # mimics generating text
            for chunk in response:
                # print(chunk['message']['content'], end='', flush=True)
                str_response += chunk['message']['content']
            
            window['output'].update(str_response)


            # for formatting
            print("\n")
            
            # add model's message into converstaion history (kind of broken)
            message_list.append({'role': 'assistant', 'content': str_response})
    
            
def sbml_generation_continous_chat(model_name: str):
    system_prompt = '''You are an SBML Multi expert; generate SBML Multi XML files based on the following image provided. 
    Only provide the completed file, no other text.'''
    message_list = [{'role': 'system', 'content': system_prompt}]

    layout = [
        [sg.Text(text = "SBML Generation Chat Application")],
        [sg.FileBrowse("Select image (or paste path)", target = 'image_input'), sg.Input('Paste image path here.', key = 'image_input'), sg.OK(key = 'input1')],
        [sg.Multiline('Generated text will appear here...', key = 'output', size = (90, 30))],
        [sg.FileSaveAs(target = 'save_output', key = 'save'), sg.Input('Paste target save location here.', key = 'save_output'), sg.OK(key = 'input2'), sg.Text(text = '', key = 'save_status')]
    ]

    window = sg.Window(title = "SBML Generation Chat Application", layout = layout, margins = (300, 150))

    while True:
        event, values = window.read()
        
        if event == sg.WIN_CLOSED:
            break
        elif event == 'image_input' or event == 'input1':
            if window.find_element_with_focus().key == 'save_output':
                with open(values['save'], 'w') as file:
                    file.write(values['output'].get())

                window['save_status'].update("Saved successfully!")
                window.refresh()

                time.sleep(3)
                
                window['save_status'].update("")
                window.refresh()
            else:
                image_path = values['image_input']

                message_list.append({'role': 'user', 'content': "Generate an SBML multi file of the provided image.", 'images': [image_path]})

                response = ollama.chat(model = model_name, messages = message_list, think = True, stream = False) # get model's response, thinking set to true

                window['output'].update(response.message.content)
        elif event == 'save_output' or event == 'input2':
            with open(values['save'], 'w') as file:
                file.write(values['output'])

            window['save_status'].update("Saved successfully!")
            window.refresh()

            time.sleep(3)

            window['save_status'].update("")
            window.refresh()

def main():
    # paragraphs = parse_file(['SBML_Core_Specification.pdf', 'SBML_Multi_Specification.pdf'])

    layout = [
        [sg.Text(text = "SBML File Generator Wizard")],
        [sg.Button(button_text = "1. RAG (questions about specifications)")],
        [sg.Button(button_text = "2. SBML Generation (generating file)")],
    ]

    window = sg.Window(title = "SBML File Generator Wizard", layout = layout, margins = (100, 100))

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == "1. RAG (questions about specifications)":
            paragraphs = parse_file(['SBML_Multi_Correct.pdf'])
            embeddings = get_embeddings('qwen3-embedding:8b', paragraphs)
            rag_continuous_chat('gemma3:4b', 'qwen3-embedding:8b', embeddings, paragraphs)
        elif event == "2. SBML Generation (generating file)":
            sbml_generation_continous_chat("minimax-m3:cloud") 

if __name__ == "__main__":
    main()
