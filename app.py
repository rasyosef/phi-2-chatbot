import gradio as gr

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    pipeline,
)
from threading import Thread

# The huggingface model id for Microsoft's phi-2 model
checkpoint = "microsoft/phi-2"

# Download and load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
)

# Text generation pipeline
phi2 = pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    device_map="cpu",
)


# Function that accepts a prompt and generates text using the phi2 pipeline
def generate(message, chat_history, max_new_tokens):
    instruction = "You are a helpful assistant to 'User'. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    final_prompt = f"Instruction: {instruction}\n"

    for sent, received in chat_history:
        final_prompt += "User: " + sent + "\n"
        final_prompt += "Assistant: " + received + "\n"

    final_prompt += "User: " + message + "\n"
    final_prompt += "Output:"

    if (
        len(tokenizer.tokenize(final_prompt))
        >= tokenizer.model_max_length - max_new_tokens
    ):
        final_prompt = "Instruction: Say 'Input exceeded context size, please clear the chat history and retry!' Output:"

    # Streamer
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300.0
    )
    thread = Thread(
        target=phi2,
        kwargs={
            "text_inputs": final_prompt,
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
        },
    )
    thread.start()

    generated_text = ""
    for word in streamer:
        generated_text += word
        response = generated_text.strip()

        if "User:" in response:
            response = response.split("User:")[0].strip()

        if "Assistant:" in response:
            response = response.split("Assistant:")[1].strip()

        yield response


# Chat interface with gradio
with gr.Blocks() as demo:
    gr.Markdown(
        """
  # Phi-2 Chatbot Demo
  This chatbot was created using Microsoft's 2.7 billion parameter [phi-2](https://huggingface.co/microsoft/phi-2) Transformer model. 
  
  In order to reduce the response time on this hardware, `max_new_tokens` has been set to `21` in the text generation pipeline. With this default configuration, it takes approximately `60 seconds` for the response to start being generated, and streamed one word at a time. Use the slider below to increase or decrease the length of the generated text.
  """
    )

    tokens_slider = gr.Slider(
        8,
        128,
        value=21,
        label="Maximum new tokens",
        info="A larger `max_new_tokens` parameter value gives you longer text responses but at the cost of a slower response time.",
    )

    chatbot = gr.ChatInterface(
        fn=generate,
        additional_inputs=[tokens_slider],
        stop_btn=None,
        examples=[["Who is Leonhard Euler?"]],
    )

demo.queue().launch()
