import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class AIChatbot:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
    def generate_response(self, prompt, max_length=100, temperature=0.7):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def chatbot():
    print("Hi, I'm an AI-powered chatbot. Type 'quit' to exit.")
    bot = AIChatbot()
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Bot: Bye, take care. See you soon!")
            break
        response = bot.generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chatbot()
