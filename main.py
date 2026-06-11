import ollama

def continuous_chat():
    model_name = "llama2"
    # message_list = [{'role': 'system', 'content': 'You are a helpful AI chatbot.'}]
    message_list = []


    while True:
        user_prompt = input("Chat (say 'exit' to exit): ")
        
        if user_prompt == 'exit':
            print('Exiting...')
            break
        else:
            message_list.append({'role': 'user', 'content': user_prompt})
            response = ollama.chat(model = model_name, messages=message_list, stream=True)

            str_response = ""

            for chunk in response:
                print(chunk['message']['content'], end='', flush=True)
                str_response += chunk['message']['content']

            print("\n")
            
            message_list.append({'role': 'assistant', 'content': str_response})

def main():
    continuous_chat()

if __name__ == "__main__":
    main()
