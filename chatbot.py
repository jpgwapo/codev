from transformers import GPT2LMHeadModel, GPT2Tokenizer

def chatbot_model(user_input):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # input
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt', padding='longest', truncation=True, max_length=512)
    
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()

    reply_length = 100

    model.config.pad_token_id = model.config.eos_token_id

    outputs = model.generate(inputs, attention_mask=attention_mask, max_length=reply_length, num_return_sequences=1, no_repeat_ngram_size=2)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

user_prompt = input("You: ")
print("ChatBot: ", chatbot_model(user_prompt))
