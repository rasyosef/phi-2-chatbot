import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# The huggingface model id for Microsoft's phi-2 model
checkpoint = "microsoft/phi-2"

# Download and load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)

# Text generation pipeline
phi2 = pipeline("text-generation", tokenizer=tokenizer, model=model)

# Function that accepts a prompt and generates text using the phi2 pipeline
def generate(prompt, chat_history, max_new_tokens):

  instruction = "You are a helpful assistant to 'User'. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
  final_prompt = f"Instruction: {instruction}\n"

  for sent, received in chat_history:
    final_prompt += "User: " + sent + "\n"
    final_prompt += "Assistant: " + received + "\n"

  final_prompt += "User: " + prompt + "\n"
  final_prompt += "Output:"

  generated_text = phi2(final_prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
  response = generated_text.split("Output:")[1]

  if "User:" in response:
    response = response.split("User:")[0]

  if "Assistant:" in response:
    response = response.split("Assistant:")[1].strip()

  chat_history.append((prompt, response))

  return "", chat_history

# Chat interface with gradio
with gr.Blocks() as demo:
  gr.Markdown("""
  # Phi-2 Chatbot Demo
  This chatbot was created using Microsoft's 2.7 billion parameter [phi-2](https://huggingface.co/microsoft/phi-2) Transformer model. 
  
  In order to reduce the response time on this hardware, `max_new_tokens` has been set to `24` in the text generation pipeline. With this default configuration, it takes approximately `60 seconds` for each response to be generated. Use the slider below to increase or decrease the length of the generated text.
  """)

  tokens_slider = gr.Slider(8, 128, value=24, label="Maximum new tokens", info="A larger `max_new_tokens` parameter value gives you longer text responses but at the cost of a slower response time.")

  chatbot = gr.Chatbot()
  msg = gr.Textbox()

  clear = gr.ClearButton([msg, chatbot])
  msg.submit(fn=generate, inputs=[msg, chatbot, tokens_slider], outputs=[msg, chatbot])

demo.launch()