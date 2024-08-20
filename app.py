import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    pipeline,
)
from threading import Thread
import gradio as gr

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# The huggingface model id for phi-2 instruct model
checkpoint = "rasyosef/phi-2-instruct-v0.1"

# Download and load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, torch_dtype=torch.float32, device_map=DEVICE
)


# Text generation pipeline
phi2 = pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=[tokenizer.eos_token_id],
    device_map=DEVICE,
)


# Function that accepts a prompt and generates text using the phi2 pipeline
def generate(message, chat_history, max_new_tokens=64):

    history = [
        {
            "role": "system",
            "content": "You are Phi, a helpful AI assistant made by Microsoft and RasYosef. User will you give you a task. Your goal is to complete the task as faithfully as you can.",
        }
    ]

    for sent, received in chat_history:
        history.append({"role": "user", "content": sent})
        history.append({"role": "assistant", "content": received})

    history.append({"role": "user", "content": message})
    # print(history)

    if len(tokenizer.apply_chat_template(history)) > 512:
        yield "chat history is too long"
    else:
        # Streamer
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=300.0,
        )
        thread = Thread(
            target=phi2,
            kwargs={
                "text_inputs": history,
                "max_new_tokens": max_new_tokens,
                "streamer": streamer,
            },
        )
        thread.start()

        generated_text = ""
        for word in streamer:
            generated_text += word
            response = generated_text.strip()

            yield response


# Chat interface with gradio
with gr.Blocks() as demo:
    gr.Markdown(
        """
  # Phi-2 Chat Demo
  This chatbot was created using a finetuned version of Microsoft's 2.7 billion parameter Phi 2 transformer model, [phi-2-instruct-v0.1](https://huggingface.co/rasyosef/phi-2-instruct-v0.1) that has underwent a post-training process that incorporates both **supervised fine-tuning** and **direct preference optimization** for instruction following.
  """
    )

    tokens_slider = gr.Slider(
        8,
        256,
        value=64,
        label="Maximum new tokens",
        info="A larger `max_new_tokens` parameter value gives you longer text responses but at the cost of a slower response time.",
    )

    chatbot = gr.ChatInterface(
        chatbot=gr.Chatbot(height=400),
        fn=generate,
        additional_inputs=[tokens_slider],
        stop_btn=None,
        examples=[
            ["Hi"],
            ["What's the German word for 'car'?"],
            [
                "Molly and Abigail want to attend a beauty and modeling contest. They both want to buy new pairs of shoes and dresses. Molly buys a pair of shoes which costs $40 and a dress which costs $160. How much should Abigail budget if she wants to spend half of what Molly spent on the pair of shoes and dress?"
            ],
        ],
    )

demo.queue().launch(debug=True)
