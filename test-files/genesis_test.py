# import the ai platform package
import ai_server
from langchain_core.messages import *
import base64
from dotenv import load_dotenv
import os

def configure():
    load_dotenv()

configure()

# pass in your access and secret keys to authenticate
server_connection=ai_server.ServerClient(
    access_key = os.getenv('access_key'), 
    secret_key = os.getenv('secret_key'),
    base="https://genai.niaid.nih.gov/Monolith/api"
)

# model connection to GPT 5.5
model = ai_server.ModelEngine(engine_id = os.getenv('engine_id'))

lc_llm = model.to_langchain_chat_model()

history = []
history.append(SystemMessage(content = "You only speak in French."))

'''
image_path = os.getenv('image_path')
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8") # 

command = [
    {"type": "text", "text": "What is the folllowing image?"},
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}, # data URI
    },
    ]
'''

while True:
    command = input("Chat (say 'exit' to exit): ")

    if command == "exit":
        print("Exiting...")
        break

    history.append(HumanMessage(content = command))

    output = lc_llm.invoke(history)

    str_response = ""

    '''
    for chunk in lc_llm.stream(history):
        content = chunk.content
        str_response += content
        print(content, end = '', flush = True)
    '''

    print(output.content)
    
    history.append(AIMessage(content = output.content))
    
    print("\n")
    
'''
history.append(HumanMessage(content = command))

output = lc_llm.invoke(history)

str_response = ""

for chunk in lc_llm.stream(history):
    content = chunk.content
    str_response += content
    print(content, end = '', flush = True)

history.append(AIMessage(content = output.content))

print("\n")
'''

'''
while True:
    # command = input("Chat (say 'exit' to exit): ")
    command = [
        {"type": "text", "text": "Describe this image in detail."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"},
        },
        ]

    if command == "exit":
        print("Exiting...")
        break

    history.append(HumanMessage(content = command))

    output = lc_llm.invoke(history)

    str_response = ""

    for chunk in lc_llm.stream(history):
        content = chunk.content
        str_response += content
        print(content, end = '', flush = True)
    
    history.append(AIMessage(content = output.content))
    
    print("\n")
'''


# print(output.content)

# define a question and grab the engine id from the server
# output = model.ask(command = "What model are you?")


