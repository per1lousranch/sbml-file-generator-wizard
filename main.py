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
import libsbml
import sys


def configure():
    load_dotenv()


# function for extracting paragraphs for SBML multi specifically
# PARAMETERS:
# filenames: a list of strings which are filenames for information to be extracted from
def rag_parse_file(filenames: list[str]):
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

# function for extracting paragraphs in general
# PARAMETERS:
# filenames: a list of strings which are filenames for information to be extracted from
def parse_file(filenames: list[str]):
    extracted_docs = ""

    # iterating through filenames
    for file in filenames:
        doc = pymupdf.open(file) # PyMuPDF allows for text extraction via paragraphs

        # iterating through pages of an individual document
        for i in range(doc.page_count): # upper limit so far: 36
            page = doc[i]

            paragraph_lst = page.get_text("blocks") # 'blocks' parameter allows extraction based on paragraphs

            for lst in paragraph_lst:
                chunk = lst[4] # 5th index contains actual text, so we only append what's there

                #if len(chunk) >= 56: # used to ensure only sentences and no titles are passed through, calculated from taking the average character per word of 4 with a lower bound of 14 words per sentences (4 x 14 = 56)
                extracted_docs += chunk # could also check if there are punctuation in the chunk?
    
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
    server_connection = ai_server.ServerClient(
        access_key = os.getenv('access_key'), 
        secret_key = os.getenv('secret_key'),
        base = "https://genai.niaid.nih.gov/Monolith/api"
    )

    # currently connected to GPT 5.5
    model = ai_server.ModelEngine(engine_id = os.getenv('engine_id'))

    lc_llm = model.to_langchain_chat_model()

    system_prompt = '''
    ### OVERALL BEHAVIOR:
    If provided with an image, generate a Level 3 Version 1 Release 1 SBML Multi XML file based on the image. If provided with a list 
    of errors, try to fix the errors in the file to abide by SBML Multi specification and generate the entire fixed file 
    again; do not change anything else in the file when fixing errors apart from what is outlined in the errors.
    
    ### FORMATTING:
    Output the final result in raw text. Do not use markdown, code blocks, or any other formatting.
    
    ### SBML MULTI REQUIREMENTS:
    Every molecule and speciesType must have 1 or more binding sites. Remember to include mcp tags when necessary.
    Compartments must have the isType attribute. SpecieisFeatureType must have 2 or more possible values. Reactions must have either:
    
    - 2 reactants and 1 product
    - 1 reactant and 2 products
    - 1 reactant and 1 product
    - 1 reactant and no products
    
    If a species in a reaction appears in both the reactants and the product, that species should be a modifier.
    
    2 different species types cannot share the exact same speciesTypeInstances and InSpeciesTypeBonds.''' # system prompt

    message_list = []
    message_list.append(SystemMessage(content = system_prompt))

    layout = [ # layout for defining elements in the GUI window
        [sg.Text(text = "SBML Generation Chat Application")],
        [sg.FileBrowse("Select image or SBML (or paste path)", target = 'path_input'), sg.Input('Paste image path here.', key = 'path_input')],
        [sg.FileBrowse("Select manuscript (or paste path)", target = 'manuscript_input'), sg.Input('Paste manuscript path here.', key = 'manuscript_input'), sg.OK(key = 'input1'), sg.Text("", key = 'thinking_status')],
        [sg.Multiline('Generated text/imported file will appear here.', key = 'output', size = (90, 30), horizontal_scroll = True), sg.Multiline("Errors found during validation will appear here.", key = 'errors', size = (60, 30), horizontal_scroll = True), sg.Multiline("Chat will appear here.", key = 'chat', size = (60, 30), horizontal_scroll = True)],
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
                manuscript_present = True

                try:
                  context = parse_file([values['manuscript_input']])
                except:
                  manuscript_present = False

                path = values['path_input'] # taking the file path

                extension = path[-4:]

                if extension == ".xml" or extension == "sbml":
                    with open(path, "r") as file:
                        window['output'].update(file.read())
                else:
                    with open(path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

                    command = [
                        {"type": "text", "text": '''Generate an SBML multi file of the provided image.
                        Use the below SBML multi files as guides on how to write and structure SBML multi:

                        FILE 1: G-protein-coupled receptor (GPCR) signaling

                        <?xml version="1.0" encoding="UTF-8"?>
                        <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" xmlns:multi="http://www.sbml.org/sbml/level3/version1/multi/version1" level="3" version="1" multi:required="true">
                          <model>
                            <listOfUnitDefinitions>
                              <unitDefinition id="micron_square_per_sec">
                                <listOfUnits>
                                  <unit kind="metre" exponent="2" scale="-6" multiplier="1"/>
                                  <unit kind="second" exponent="-1" scale="0" multiplier="1"/>
                                </listOfUnits>
                              </unitDefinition>
                              <unitDefinition id="per_sec">
                                <listOfUnits>
                                  <unit kind="second" exponent="-1" scale="0" multiplier="1"/>
                                </listOfUnits>
                              </unitDefinition>
                              <unitDefinition id="litre_per_mole_per_sec">
                                <listOfUnits>
                                  <unit kind="litre" exponent="1" scale="0" multiplier="1"/>
                                  <unit kind="mole" exponent="-1" scale="0" multiplier="1"/>
                                  <unit kind="second" exponent="-1" scale="0" multiplier="1"/>
                                </listOfUnits>
                              </unitDefinition>
                            </listOfUnitDefinitions>
                            <listOfCompartments>
                              <compartment id="membrane" name="membrane" spatialDimensions="2" constant="true" multi:isType="true">
                                <multi:listOfCompartmentReferences>
                                  <multi:compartmentReference multi:compartment="outside_membrane"/>
                                  <multi:compartmentReference multi:compartment="inside_membrane"/>
                                </multi:listOfCompartmentReferences>
                              </compartment>
                              <compartment id="outside_membrane" name="Outside membrane" constant="true" multi:isType="true"/>
                              <compartment id="inside_membrane" name="inside membrane" constant="true" multi:isType="true"/>
                              <compartment id="free_diffusing" name="free diffusing" spatialDimensions="3" constant="true" multi:isType="true"/>
                              <compartment id="any" name="any" constant="true" multi:isType="true"/>
                            </listOfCompartments>
                            <listOfSpecies>
                              <species id="cpx_000001" name="Receptor_1" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000001">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000001"/>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000002"/>
                                </multi:listOfOutwardBindingSites>
                              </species>
                              <species id="cpx_000002" name="Ligand_1" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000002">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000003"/>
                                </multi:listOfOutwardBindingSites>
                              </species>
                              <species id="cpx_000003" name="Ligand.Receptor_1" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000005">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000001"/>
                                </multi:listOfOutwardBindingSites>
                              </species>
                              <species id="cpx_000004" name="Galpha_1" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000003">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000004"/>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000005"/>
                                </multi:listOfOutwardBindingSites>
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:speciesFeatureType="mcp_000004_GTP" multi:occur="1">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="mcp_000004_GTP_off"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                              <species id="cpx_000005" name="Gbetagamma_1" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000004">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000006"/>
                                </multi:listOfOutwardBindingSites>
                              </species>
                              <species id="cpx_000006" name="Galpha.Gbetagamma_1" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000006">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000004"/>
                                </multi:listOfOutwardBindingSites>
                              </species>
                              <species id="cpx_000007" name="Galpha.Gbetagamma_2" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000006">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000004"/>
                                </multi:listOfOutwardBindingSites>
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:speciesFeatureType="mcp_000004_GTP" multi:occur="1">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="mcp_000004_GTP_off"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                              <species id="cpx_000008" name="Galpha_2" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000003">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000004"/>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000005"/>
                                </multi:listOfOutwardBindingSites>
                              </species>
                              <species id="cpx_000009" name="Galpha.Gbetagamma_3" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000006">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000004"/>
                                </multi:listOfOutwardBindingSites>
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:speciesFeatureType="mcp_000004_GTP" multi:occur="1">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="mcp_000004_GTP_on"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                              <species id="cpx_000010" name="Receptor_unbnd" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000005">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000001"/>
                                </multi:listOfOutwardBindingSites>
                              </species>
                              <species id="cpx_000011" name="Galpha.Gbetagamma_unbnd" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000006">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000004"/>
                                </multi:listOfOutwardBindingSites>
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:speciesFeatureType="mcp_000004_GTP" multi:occur="1">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="mcp_000004_GTP_off"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                              <species id="cpx_000012" name="Galpha.Gbetagamma.Ligand.Receptor_1" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000007"/>
                              <species id="cpx_000013" name="Galpha.Receptor_1" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000008">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000002"/>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000005"/>
                                </multi:listOfOutwardBindingSites>
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:speciesFeatureType="mcp_000004_GTP" multi:occur="1">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="mcp_000004_GTP_on"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                              <species id="cpx_000014" name="Receptor_2" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000001">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000001"/>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000002"/>
                                </multi:listOfOutwardBindingSites>
                              </species>
                              <species id="cpx_000015" name="Galpha_3" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000003">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000004"/>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000005"/>
                                </multi:listOfOutwardBindingSites>
                              </species>
                              <species id="cpx_000016" name="Galpha.Receptor_2" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000008">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000002"/>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000005"/>
                                </multi:listOfOutwardBindingSites>
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:speciesFeatureType="mcp_000004_GTP" multi:occur="1">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="mcp_000004_GTP_off"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                              <species id="cpx_000017" name="Galpha.Gbetagamma.Ligand.Receptor_2" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000007">
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:speciesFeatureType="mcp_000004_GTP" multi:occur="1">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="mcp_000004_GTP_off"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                              <species id="cpx_000018" name="Galpha.Gbetagamma.Ligand.Receptor_3" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000007">
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:speciesFeatureType="mcp_000004_GTP" multi:occur="1">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="mcp_000004_GTP_on"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                              <species id="cpx_000019" name="Galpha_4" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000003">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000004"/>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000005"/>
                                </multi:listOfOutwardBindingSites>
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:speciesFeatureType="mcp_000004_GTP" multi:occur="1">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="mcp_000004_GTP_on"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                              <species id="cpx_000020" name="Galpha_5" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000003">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000004"/>
                                  <multi:outwardBindingSite multi:bindingStatus="unbound" multi:component="bst_000005"/>
                                </multi:listOfOutwardBindingSites>
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:speciesFeatureType="mcp_000004_GTP" multi:occur="1">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="mcp_000004_GTP_off"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                              <species id="cpx_000021" name="Galpha_GTP_all" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000003">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000004"/>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000005"/>
                                </multi:listOfOutwardBindingSites>
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:speciesFeatureType="mcp_000004_GTP" multi:occur="1">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="mcp_000004_GTP_on"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                              <species id="cpx_000022" name="Galpha_7" compartment="any" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false" multi:speciesType="cps_000003">
                                <multi:listOfOutwardBindingSites>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000004"/>
                                  <multi:outwardBindingSite multi:bindingStatus="either" multi:component="bst_000005"/>
                                </multi:listOfOutwardBindingSites>
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:speciesFeatureType="mcp_000004_GTP" multi:occur="1">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="mcp_000004_GTP_off"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                            </listOfSpecies>
                            <listOfParameters>
                              <parameter id="par_1" value="0.1" units="micron_square_per_sec" constant="true"/>
                              <parameter id="par_2" value="100" units="micron_square_per_sec" constant="true"/>
                              <parameter id="par_3" value="0.001" units="micron_square_per_sec" constant="true"/>
                            </listOfParameters>
                            <listOfReactions>
                              <reaction id="trn_000002" name="Galpha auto-GTPase" reversible="false" fast="false">
                                <listOfReactants>
                                  <speciesReference id="spr_cpx_000019" species="cpx_000019" constant="false"/>
                                </listOfReactants>
                                <listOfProducts>
                                  <speciesReference species="cpx_000020" constant="false"/>
                                </listOfProducts>
                                <kineticLaw>
                                  <math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <apply>
                                      <times/>
                                      <ci> k </ci>
                                      <ci> cpx_000019 </ci>
                                    </apply>
                                  </math>
                                  <listOfLocalParameters>
                                    <localParameter id="k" value="0.3" units="per_sec"/>
                                  </listOfLocalParameters>
                                </kineticLaw>
                              </reaction>
                              <reaction id="trn_000001" name="Rec mediated Galpha GDP GTP exchange" reversible="false" fast="false">
                                <listOfReactants>
                                  <speciesReference id="spr_cpx_000017" species="cpx_000017" constant="false"/>
                                </listOfReactants>
                                <listOfProducts>
                                  <speciesReference species="cpx_000018" constant="false"/>
                                </listOfProducts>
                                <kineticLaw>
                                  <math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <apply>
                                      <times/>
                                      <ci> k </ci>
                                      <ci> cpx_000017 </ci>
                                    </apply>
                                  </math>
                                  <listOfLocalParameters>
                                    <localParameter id="k" value="3" units="per_sec"/>
                                  </listOfLocalParameters>
                                </kineticLaw>
                              </reaction>
                              <reaction id="cpi_000002" name="G protein recombination" reversible="false" fast="false" compartment="free_diffusing">
                                <listOfReactants>
                                  <speciesReference id="spr1_cpx_000004" name="Galpha_1" species="cpx_000004" constant="false"/>
                                  <speciesReference id="spr2_cpx_000005" name="Gbetagamma_1" species="cpx_000005" constant="false"/>
                                </listOfReactants>
                                <listOfProducts>
                                  <speciesReference species="cpx_000006" constant="false"/>
                                </listOfProducts>
                                <kineticLaw>
                                  <math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <apply>
                                      <times/>
                                      <ci> k </ci>
                                      <ci> cpx_000004 </ci>
                                      <ci> cpx_000005 </ci>
                                    </apply>
                                  </math>
                                  <listOfLocalParameters>
                                    <localParameter id="k" value="1000000" units="litre_per_mole_per_sec"/>
                                  </listOfLocalParameters>
                                </kineticLaw>
                              </reaction>
                              <reaction id="cpi_000003" name="G protein recruitment" reversible="false" fast="false" compartment="free_diffusing">
                                <listOfReactants>
                                  <speciesReference id="spr1_cpx_000010" name="Receptor_unbnd" species="cpx_000010" constant="false"/>
                                  <speciesReference id="spr2_cpx_000011" species="cpx_000011" constant="false"/>
                                </listOfReactants>
                                <listOfProducts>
                                  <speciesReference species="cpx_000012" constant="false"/>
                                </listOfProducts>
                                <kineticLaw>
                                  <math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <apply>
                                      <times/>
                                      <ci> k </ci>
                                      <ci> cpx_000010 </ci>
                                      <ci> cpx_000011 </ci>
                                    </apply>
                                  </math>
                                  <listOfLocalParameters>
                                    <localParameter id="k" value="10000" units="litre_per_mole_per_sec"/>
                                  </listOfLocalParameters>
                                </kineticLaw>
                              </reaction>
                              <reaction id="cpi_000001" name="Receptor ligation" reversible="false" fast="false" compartment="free_diffusing">
                                <listOfReactants>
                                  <speciesReference id="spr1_cpx_000001" name="Receptor_1" species="cpx_000001" constant="false"/>
                                  <speciesReference id="spr2_cpx_000002" name="Ligand_1" species="cpx_000002" constant="false"/>
                                </listOfReactants>
                                <listOfProducts>
                                  <speciesReference species="cpx_000003" constant="false"/>
                                </listOfProducts>
                                <kineticLaw>
                                  <math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <apply>
                                      <times/>
                                      <ci> k </ci>
                                      <ci> cpx_000001 </ci>
                                      <ci> cpx_000002 </ci>
                                    </apply>
                                  </math>
                                  <listOfLocalParameters>
                                    <localParameter id="k" value="10000000" units="litre_per_mole_per_sec"/>
                                  </listOfLocalParameters>
                                </kineticLaw>
                              </reaction>
                              <reaction id="cpd_000002" name="[Galpha.Gbetagamma_1]-dissociation" reversible="false" fast="false">
                                <listOfReactants>
                                  <speciesReference id="spr_cpx_000007" species="cpx_000007" constant="false"/>
                                </listOfReactants>
                                <listOfProducts>
                                  <speciesReference name="Galpha_2" species="cpx_000008" constant="false"/>
                                  <speciesReference name="Gbetagamma_1" species="cpx_000005" constant="false"/>
                                </listOfProducts>
                                <kineticLaw>
                                  <math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <apply>
                                      <times/>
                                      <ci> k </ci>
                                      <ci> cpx_000007 </ci>
                                    </apply>
                                  </math>
                                  <listOfLocalParameters>
                                    <localParameter id="k" value="0.01" units="per_sec"/>
                                  </listOfLocalParameters>
                                </kineticLaw>
                              </reaction>
                              <reaction id="cpd_000003" name="[Galpha.Gbetagamma_3]-dissociation" reversible="false" fast="false">
                                <listOfReactants>
                                  <speciesReference id="spr_cpx_000009" species="cpx_000009" constant="false"/>
                                </listOfReactants>
                                <listOfProducts>
                                  <speciesReference name="Galpha_2" species="cpx_000008" constant="false"/>
                                  <speciesReference name="Gbetagamma_1" species="cpx_000005" constant="false"/>
                                </listOfProducts>
                                <kineticLaw>
                                  <math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <apply>
                                      <times/>
                                      <ci> k </ci>
                                      <ci> cpx_000009 </ci>
                                    </apply>
                                  </math>
                                  <listOfLocalParameters>
                                    <localParameter id="k" value="10" units="per_sec"/>
                                  </listOfLocalParameters>
                                </kineticLaw>
                              </reaction>
                              <reaction id="cpd_000005" name="[Galpha.Receptor_2]-dissociation" reversible="false" fast="false">
                                <listOfReactants>
                                  <speciesReference id="spr_cpx_000016" species="cpx_000016" constant="false"/>
                                </listOfReactants>
                                <listOfProducts>
                                  <speciesReference name="Receptor_2" species="cpx_000014" constant="false"/>
                                  <speciesReference name="Galpha_3" species="cpx_000015" constant="false"/>
                                </listOfProducts>
                                <kineticLaw>
                                  <math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <apply>
                                      <times/>
                                      <ci> k </ci>
                                      <ci> cpx_000016 </ci>
                                    </apply>
                                  </math>
                                  <listOfLocalParameters>
                                    <localParameter id="k" value="0.01" units="per_sec"/>
                                  </listOfLocalParameters>
                                </kineticLaw>
                              </reaction>
                              <reaction id="cpd_000001" name="[Ligand.Receptor_1]-dissociation" reversible="false" fast="false">
                                <listOfReactants>
                                  <speciesReference id="spr_cpx_000003" species="cpx_000003" constant="false"/>
                                </listOfReactants>
                                <listOfProducts>
                                  <speciesReference name="Receptor_1" species="cpx_000001" constant="false"/>
                                  <speciesReference name="Ligand_1" species="cpx_000002" constant="false"/>
                                </listOfProducts>
                                <kineticLaw>
                                  <math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <apply>
                                      <times/>
                                      <ci> k </ci>
                                      <ci> cpx_000003 </ci>
                                    </apply>
                                  </math>
                                  <listOfLocalParameters>
                                    <localParameter id="k" value="0.1" units="per_sec"/>
                                  </listOfLocalParameters>
                                </kineticLaw>
                              </reaction>
                              <reaction id="cpd_000004" name="Rec GalphaGTP dissoc" reversible="false" fast="false">
                                <listOfReactants>
                                  <speciesReference id="spr_cpx_000013" species="cpx_000013" constant="false"/>
                                </listOfReactants>
                                <listOfProducts>
                                  <speciesReference name="Receptor_2" species="cpx_000014" constant="false"/>
                                  <speciesReference name="Galpha_3" species="cpx_000015" constant="false"/>
                                </listOfProducts>
                                <kineticLaw>
                                  <math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <apply>
                                      <times/>
                                      <ci> k </ci>
                                      <ci> cpx_000013 </ci>
                                    </apply>
                                  </math>
                                  <listOfLocalParameters>
                                    <localParameter id="k" value="10" units="per_sec"/>
                                  </listOfLocalParameters>
                                </kineticLaw>
                              </reaction>
                            </listOfReactions>
                            <multi:listOfSpeciesTypes>
                              <multi:speciesType multi:id="mol_000003" multi:name="Galpha" multi:compartment="membrane">
                                <annotation>diffusionCoefficient:par_1</annotation>
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_mcp_000004" multi:name="Galpha_inside-membrane" multi:speciesType="mcp_000004"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                              <multi:speciesType multi:id="mcp_000004" multi:name="Galpha_inside-membrane" multi:compartment="inside_membrane">
                                <multi:listOfSpeciesFeatureTypes>
                                  <multi:speciesFeatureType multi:id="mcp_000004_GTP" multi:name="GTP" multi:occur="1">
                                    <multi:listOfPossibleSpeciesFeatureValues>
                                      <multi:possibleSpeciesFeatureValue multi:id="mcp_000004_GTP_on" multi:name="on"/>
                                      <multi:possibleSpeciesFeatureValue multi:id="mcp_000004_GTP_off" multi:name="off"/>
                                    </multi:listOfPossibleSpeciesFeatureValues>
                                  </multi:speciesFeatureType>
                                </multi:listOfSpeciesFeatureTypes>
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_bst_000004" multi:name="Receptor binding site" multi:speciesType="bst_000004"/>
                                  <multi:speciesTypeInstance multi:id="sti_bst_000005" multi:name="Gbetagamma binding site" multi:speciesType="bst_000005"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                              <multi:bindingSiteSpeciesType multi:id="bst_000004" multi:name="Receptor binding site"/>
                              <multi:bindingSiteSpeciesType multi:id="bst_000005" multi:name="Gbetagamma binding site"/>
                              <multi:speciesType multi:id="mol_000004" multi:name="Gbetagamma" multi:compartment="membrane">
                                <annotation>diffusionCoefficient:par_1</annotation>
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_mcp_000005" multi:name="Gbetagamma_inside-membrane" multi:speciesType="mcp_000005"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                              <multi:speciesType multi:id="mcp_000005" multi:name="Gbetagamma_inside-membrane" multi:compartment="inside_membrane">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_bst_000006" multi:name="Gbetagamma_site_1" multi:speciesType="bst_000006"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                              <multi:bindingSiteSpeciesType multi:id="bst_000006" multi:name="Gbetagamma_site_1"/>
                              <multi:speciesType multi:id="mol_000002" multi:name="Ligand" multi:compartment="free_diffusing">
                                <annotation>diffusionCoefficient:par_2</annotation>
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_mcp_000003" multi:name="Ligand_component_1" multi:speciesType="mcp_000003"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                              <multi:speciesType multi:id="mcp_000003" multi:name="Ligand_component_1" multi:compartment="free_diffusing">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_bst_000003" multi:name="Ligand_site_1" multi:speciesType="bst_000003"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                              <multi:bindingSiteSpeciesType multi:id="bst_000003" multi:name="Ligand_site_1"/>
                              <multi:speciesType multi:id="mol_000001" multi:name="Receptor" multi:compartment="membrane">
                                <annotation>diffusionCoefficient:par_3</annotation>
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_mcp_000001" multi:name="Intracellular Domain" multi:speciesType="mcp_000001"/>
                                  <multi:speciesTypeInstance multi:id="sti_mcp_000002" multi:name="Extracellular Domain" multi:speciesType="mcp_000002"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                              <multi:speciesType multi:id="mcp_000001" multi:name="Intracellular Domain" multi:compartment="inside_membrane">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_bst_000001" multi:name="G protein recruitment site" multi:speciesType="bst_000001"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                              <multi:bindingSiteSpeciesType multi:id="bst_000001" multi:name="G protein recruitment site"/>
                              <multi:speciesType multi:id="mcp_000002" multi:name="Extracellular Domain" multi:compartment="outside_membrane">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_bst_000002" multi:name="Ligand site" multi:speciesType="bst_000002"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                              <multi:bindingSiteSpeciesType multi:id="bst_000002" multi:name="Ligand site"/>
                              <multi:speciesType multi:id="cps_000003" multi:name="Galpha">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000003_1_mol_000003" multi:name="Galpha" multi:speciesType="mol_000003"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                              <multi:speciesType multi:id="cps_000006" multi:name="Galpha.Gbetagamma">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000006_1_mol_000003" multi:name="Galpha" multi:speciesType="mol_000003"/>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000006_2_mol_000004" multi:name="Gbetagamma" multi:speciesType="mol_000004"/>
                                </multi:listOfSpeciesTypeInstances>
                                <multi:listOfInSpeciesTypeBonds>
                                  <multi:inSpeciesTypeBond multi:bindingSite1="sti_bst_000006" multi:bindingSite2="sti_bst_000005"/>
                                </multi:listOfInSpeciesTypeBonds>
                              </multi:speciesType>
                              <multi:speciesType multi:id="cps_000007" multi:name="Galpha.Gbetagamma.Ligand.Receptor">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000007_1_mol_000001" multi:name="Receptor" multi:speciesType="mol_000001"/>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000007_2_mol_000002" multi:name="Ligand" multi:speciesType="mol_000002"/>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000007_3_mol_000003" multi:name="Galpha" multi:speciesType="mol_000003"/>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000007_4_mol_000004" multi:name="Gbetagamma" multi:speciesType="mol_000004"/>
                                </multi:listOfSpeciesTypeInstances>
                                <multi:listOfInSpeciesTypeBonds>
                                  <multi:inSpeciesTypeBond multi:bindingSite1="sti_bst_000003" multi:bindingSite2="sti_bst_000002"/>
                                  <multi:inSpeciesTypeBond multi:bindingSite1="sti_bst_000006" multi:bindingSite2="sti_bst_000005"/>
                                  <multi:inSpeciesTypeBond multi:bindingSite1="sti_bst_000004" multi:bindingSite2="sti_bst_000001"/>
                                </multi:listOfInSpeciesTypeBonds>
                              </multi:speciesType>
                              <multi:speciesType multi:id="cps_000008" multi:name="Galpha.Receptor">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000008_1_mol_000001" multi:name="Receptor" multi:speciesType="mol_000001"/>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000008_2_mol_000003" multi:name="Galpha" multi:speciesType="mol_000003"/>
                                </multi:listOfSpeciesTypeInstances>
                                <multi:listOfInSpeciesTypeBonds>
                                  <multi:inSpeciesTypeBond multi:bindingSite1="sti_bst_000004" multi:bindingSite2="sti_bst_000001"/>
                                </multi:listOfInSpeciesTypeBonds>
                              </multi:speciesType>
                              <multi:speciesType multi:id="cps_000004" multi:name="Gbetagamma">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000004_1_mol_000004" multi:name="Gbetagamma" multi:speciesType="mol_000004"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                              <multi:speciesType multi:id="cps_000002" multi:name="Ligand">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000002_1_mol_000002" multi:name="Ligand" multi:speciesType="mol_000002"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                              <multi:speciesType multi:id="cps_000005" multi:name="Ligand.Receptor">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000005_1_mol_000001" multi:name="Receptor" multi:speciesType="mol_000001"/>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000005_2_mol_000002" multi:name="Ligand" multi:speciesType="mol_000002"/>
                                </multi:listOfSpeciesTypeInstances>
                                <multi:listOfInSpeciesTypeBonds>
                                  <multi:inSpeciesTypeBond multi:bindingSite1="sti_bst_000003" multi:bindingSite2="sti_bst_000002"/>
                                </multi:listOfInSpeciesTypeBonds>
                              </multi:speciesType>
                              <multi:speciesType multi:id="cps_000001" multi:name="Receptor">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="sti_cps_000001_1_mol_000001" multi:name="Receptor" multi:speciesType="mol_000001"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                            </multi:listOfSpeciesTypes>
                          </model>
                        </sbml>
                        
                        FILE 2: toy protein with one site, Y, whose phosphorylation state can be U or P, plus one reaction

                        <?xml version="1.0" encoding="UTF-8"?>
                        <sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"
                              xmlns:multi="http://www.sbml.org/sbml/level3/version1/multi/version1"
                              level="3" version="1" multi:required="true">

                          <model id="multi_phosphorylation_example" name="Toy SBML Multi phosphorylation example" substanceUnits="mole" timeUnits="second" extentUnits="mole">

                            <listOfUnitDefinitions>
                              <unitDefinition id="per_second">
                                <listOfUnits>
                                  <unit kind="second" exponent="-1" scale="0" multiplier="1"/>
                                </listOfUnits>
                              </unitDefinition>
                            </listOfUnitDefinitions>

                            <listOfCompartments>
                              <compartment id="cytosol" name="cytosol" spatialDimensions="3" size="1" units="litre" constant="true" multi:isType="false"/>
                            </listOfCompartments>

                            <!-- Multi package: species-type definitions. -->
                            <multi:listOfSpeciesTypes>
                              <!-- A binding/site species type with a discrete feature: phosphorylation state. -->
                              <multi:bindingSiteSpeciesType multi:id="st_Protein_site_Y" multi:name="Y site">
                                <multi:listOfSpeciesFeatureTypes>
                                  <multi:speciesFeatureType multi:id="sft_phosphorylation" multi:name="phosphorylation" multi:occur="1">
                                    <multi:listOfPossibleSpeciesFeatureValues>
                                      <multi:possibleSpeciesFeatureValue multi:id="U" multi:name="unphosphorylated"/>
                                      <multi:possibleSpeciesFeatureValue multi:id="P" multi:name="phosphorylated"/>
                                    </multi:listOfPossibleSpeciesFeatureValues>
                                  </multi:speciesFeatureType>
                                </multi:listOfSpeciesFeatureTypes>
                              </multi:bindingSiteSpeciesType>

                              <!-- A protein species type containing the Y site as a component. -->
                              <multi:speciesType multi:id="st_Protein" multi:name="Protein">
                                <multi:listOfSpeciesTypeInstances>
                                  <multi:speciesTypeInstance multi:id="Y" multi:name="Y" multi:speciesType="st_Protein_site_Y"/>
                                </multi:listOfSpeciesTypeInstances>
                              </multi:speciesType>
                            </multi:listOfSpeciesTypes>

                            <listOfSpecies>
                              <species id="prot_U" name="Protein(Y~U)" compartment="cytosol" initialAmount="1e-6" substanceUnits="mole" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false" multi:speciesType="st_Protein">
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:id="sf_prot_U_phosphorylation" multi:speciesFeatureType="sft_phosphorylation" multi:occur="1" multi:component="Y">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="U"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>

                              <species id="prot_P" name="Protein(Y~P)" compartment="cytosol" initialAmount="0" substanceUnits="mole" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false" multi:speciesType="st_Protein">
                                <multi:listOfSpeciesFeatures>
                                  <multi:speciesFeature multi:id="sf_prot_P_phosphorylation" multi:speciesFeatureType="sft_phosphorylation" multi:occur="1" multi:component="Y">
                                    <multi:listOfSpeciesFeatureValues>
                                      <multi:speciesFeatureValue multi:value="P"/>
                                    </multi:listOfSpeciesFeatureValues>
                                  </multi:speciesFeature>
                                </multi:listOfSpeciesFeatures>
                              </species>
                            </listOfSpecies>

                            <listOfReactions>
                              <reaction id="phosphorylation" name="phosphorylation" reversible="false" fast="false">
                                <listOfReactants>
                                  <speciesReference id="sr_prot_U" species="prot_U" stoichiometry="1" constant="true"/>
                                </listOfReactants>
                                <listOfProducts>
                                  <speciesReference id="sr_prot_P" species="prot_P" stoichiometry="1" constant="true"/>
                                </listOfProducts>
                                <kineticLaw>
                                  <math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <apply>
                                      <times/>
                                      <ci> k_phos </ci>
                                      <ci> prot_U </ci>
                                    </apply>
                                  </math>
                                  <listOfLocalParameters>
                                    <localParameter id="k_phos" value="0.1" units="per_second"/>
                                  </listOfLocalParameters>
                                </kineticLaw>
                              </reaction>
                            </listOfReactions>
                          </model>
                        </sbml>
                        '''},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}, # data URI
                        },
                    ]

                    if manuscript_present:
                        print("extra context used")
                        command.append({"type": "text", "text": "Use the additional information from a document to assist in creating a more detailed SBML Multi file. Indicate whether or not you used the additional information in the 1st line with an XML comment. Do not take in any information from the document if it is not relevant to the information in the image. The information from the document is here: " + context})

                    message_list.append(HumanMessage(content = command))

                    window['thinking_status'].update("Generating...")
                    window.refresh()

                    response = lc_llm.invoke(message_list, thinking = True, thinking_budget = "medium")

                    window['output'].update(response.content) # updating the box with the 

                    update_text_element(window, 'thinking_status', "", "Finished!", 3)
        elif event == 'save_output' or event == 'input2': # event for hitting enter on save file path box or OK button next to it
            with open(values['save'], 'w') as file:
                file.write(values['output']) # saving what's in output to file

            update_text_element(window, "save_status", "                                                       ", "Saved successfully!                        ", 3) # updating status message
        elif event == 'validate': # event for pressing 'Validate SBML file'
            reader = SBMLReader() # initialize SBMLReader

            doc = reader.readSBMLFromString(values['output']) # read from string in output

            print("Internal consistency: " + str(doc.checkInternalConsistency()))
            print("Consistency: " + str(doc.checkConsistency()))

            stream = libsbml.ostringstream()
            doc.printErrors(stream)
            output = stream.str()

            print(output)

            if output == "": # errors empty
                window['errors'].update("No errors found.")
            else:
                window['errors'].update(output)

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
            paragraphs = rag_parse_file(['SBML_Multi_Correct.pdf'])
            embeddings = get_embeddings('qwen3-embedding:8b', paragraphs)
            rag_continuous_chat('gemma3:4b', 'qwen3-embedding:8b', embeddings, paragraphs)
        elif event == "2":
            sbml_generation_continous_chat()

            
if __name__ == "__main__":
    main()
