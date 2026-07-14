import ollama
import pymupdf
import numpy as np
import json
import os.path
import PySimpleGUI as sg
import time
from libsbml import *
import ai_server
from langchain_core.messages import *
import base64
from dotenv import load_dotenv
import os

def configure():
    load_dotenv()

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


# function for executing the RAG question asking chat
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

    window = sg.Window(title = "RAG Chat Application", layout = layout, margins = (300, 150), resizable = True)

    while True:
        event, values = window.read()
        
        if event == sg.WIN_CLOSED:
            break
        else:
            # window['output'].update("Generating... please wait...") # doesn't work???

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
    

# function for executing the SBML generation feature
# PARAMETERS:
# model_name: string which contins the model name to be used for generating the file and fixing errors
def sbml_generation_continous_chat():
    server_connection=ai_server.ServerClient(
        access_key = os.getenv('access_key'), 
        secret_key = os.getenv('secret_key'),
        base = "https://genai.niaid.nih.gov/Monolith/api"
    )

    # currently connected to GPT 5.5
    model = ai_server.ModelEngine(engine_id = os.getenv('engine_id'))

    lc_llm = model.to_langchain_chat_model()

    system_prompt = '''If provided with an image, enerate a SBML Multi XML file based on the image. If provided with a list 
    of errors, try to fix the errors in the file to abide by SBML Multi specification and generate the entire fixed file 
    again; do not change anything else in the file when fixing errors apart from what is outlined in the errors.
    Output the final result in raw text. Do not use markdown, code blocks, or any other formatting.''' # system prompt

    message_list = []
    message_list.append(SystemMessage(content = system_prompt))

    layout = [ # layout for defining elements in the GUI window
        [sg.Text(text = "SBML Generation Chat Application")],
        [sg.FileBrowse("Select image or SBML (or paste path)", target = 'path_input'), sg.Input('Paste image path here.', key = 'path_input'), sg.OK(key = 'input1'), sg.Text("", key = 'thinking_status')],
        [sg.Multiline('Generated text/imported file will appear here.', key = 'output', size = (90, 30), horizontal_scroll = True), sg.Multiline("Errors found during validation will appear here.", key = 'errors', size = (60, 30), horizontal_scroll = True)],
        [sg.FileSaveAs(target = 'save_output', key = 'save'), sg.Input('Paste target save location here.', key = 'save_output'), sg.OK(key = 'input2'), sg.Text(text = '                                                       ', key = 'save_status'), sg.Button("Validate SBML file", key = 'validate'), sg.Button("Submit validations to LLM", key = 'submit_validations'), sg.Text("", key = 'validation_status')]
    ]

    window = sg.Window(title = "SBML Generation Chat Application", layout = layout, margins = (240, 150), resizable = False) # defining the window, resizable false for now...

    while True: # loop for running the window
        event, values = window.read() # event records what event occured, values record the values of elements at the time of the event
        
        if event == sg.WIN_CLOSED: # when the window is closed
            break # break out of while True loop
        elif event == 'path_input' or event == 'input1': # event for hitting enter on image input box or OK button next to it
            if window.find_element_with_focus().key == 'save_output': # for some reason, hitting 'enter' on the save file input box triggers this event
                with open(values['save'], 'w') as file: # so this if statement ensures that whatever input box has focus is properly triggered 
                    file.write(values['output'].get()) # writing result to file

                update_text_element(window, "save_status", "                                                       ", "Saved successfully!                        ", 3) # status message
            else:
                path = values['path_input'] # taking the file path

                extension = path[-4:]

                if extension == ".xml" or extension == "sbml":
                    with open(path, "r") as file:
                        window['output'].update(file.read())
                else:
                    with open(path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

                    command = [
                        {"type": "text", "text": "Generate an SBML multi file of the provided image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}, # data URI
                        },
                    ]

                    message_list.append(HumanMessage(content = command))

                    window['thinking_status'].update("Generating...")
                    window.refresh()

                    response = lc_llm.invoke(message_list)

                    window['output'].update(response.content) # updating the box with the 

                    update_text_element(window, 'thinking_status', "", "Finished!", 3)
        elif event == 'save_output' or event == 'input2': # event for hitting enter on save file path box or OK button next to it
            with open(values['save'], 'w') as file:
                file.write(values['output']) # saving what's in output to file

            update_text_element(window, "save_status", "                                                       ", "Saved successfully!                        ", 3) # updating status message
        elif event == 'validate': # event for pressing 'Validate SBML file'
            reader = SBMLReader() # initialize SBMLReader

            doc = reader.readSBMLFromString(values['output']) # read from string in output

            error_log = doc.getErrorLog()

            errors = error_log.toString()

            if errors == "": # errors empty
                window['errors'].update("No errors found.")
            else:
                window['errors'].update(errors)

            window.refresh()
        elif event == 'submit_validations': # event for pressing 'Submit validations to LLM'
            if values['errors'] == 'Errors found during validation will appear here.' or values['errors'] == 'No errors found.': # case where no errors found or validation not ran yet
                update_text_element(window, "validation_status", "", "No errors found to fix.", 3)
            else:
                window['validation_status'].update("Fixing errors...")
                window.refresh()

                command = [
                    {"type": "text", "text": "Errors: " + values['errors'] + ". File: " + values['output']},
                ]

                message_list.append(HumanMessage(content = command))

                response = lc_llm.invoke(message_list)

                window['output'].update(response.content)

                update_text_element(window, "validation_status", "", "Finished!", 3)


# function for updating certain text elements to show current status of application
# PARAMETERS:
# window: current PySimpleGUI window
# target: string for the key of the target element to be updated
# before: string of original state of the text element that should be restored after it changes
# after: string of state to temporarily change elemnet to
# wait: integer for how many seconds the change should be shown before reverting back
def update_text_element(window, target: str, before: str, after: str, wait: int):
    window[target].update(after)
    window.refresh()

    time.sleep(wait)

    window[target].update(before)
    window.refresh()


def main():
    configure()

    layout = [
        [sg.Text(text = "SBML File Generator Wizard", expand_x = True, expand_y = True, justification = 'center')],
        [sg.Button(button_text = "1. Questions about specifications", key = '1', expand_x = True, expand_y = True)],
        [sg.Button(button_text = "2. SBML generation", key = '2', expand_x = True, expand_y = True)],
    ]

    window = sg.Window(title = "SBML File Generator Wizard", layout = layout, margins = (100, 100), resizable = True)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == "1":
            paragraphs = parse_file(['SBML_Multi_Correct.pdf'])
            embeddings = get_embeddings('qwen3-embedding:8b', paragraphs)
            rag_continuous_chat('gemma3:4b', 'qwen3-embedding:8b', embeddings, paragraphs)
        elif event == "2":
            sbml_generation_continous_chat()

            
if __name__ == "__main__":
    main()
