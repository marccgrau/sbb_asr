from transformers import pipeline
import gradio as gr

pipe = pipeline(model="marccgrau/whisper-small-init") 

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper Small SBB Shunting Communication",
    description="Realtime demo for speech recognition of shunting communication.",
)

iface.launch()