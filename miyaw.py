import openai
import gradio as gr
from PIL import Image
import pytesseract
import numpy as np

messages = [{"role": "system", "content": "sen problem ve soru çözersin "}]

def get_image_text(image: np.ndarray) -> str:
    image_pil = Image.fromarray(image)
    return pytesseract.image_to_string(image_pil)

def CustomChatGPT(user_input, key, image):
    openai.api_key = key
    if image is not None:
        user_input = get_image_text(image)

    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response.choices[0].message["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

inputs = [
    gr.inputs.Textbox(lines=2, label="promptun"),
    gr.inputs.Textbox(lines=1, label="OpenAI Key"),
    gr.inputs.Image(label="fotoğraf yükleme")
]

demo = gr.Interface(fn=CustomChatGPT, inputs=inputs, outputs="text", title="soru çözme")

demo.launch(share=True)
