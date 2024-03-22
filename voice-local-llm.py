from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from vosk import Model,KaldiRecognizer
from huggingface_hub import hf_hub_download
import pyttsx3
import pyaudio
import json

sttmodel = Model(r"/home/ai-project/aiproject/myai/models/vosk-model-en-in-0.5")
recogniser = KaldiRecognizer(sttmodel, 16000)
mic = pyaudio.PyAudio()
sttstream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

MODELS_PATH = "./models"
model_path = hf_hub_download(   
        repo_id= "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        resume_download=True,
        cache_dir=MODELS_PATH,)

kwargs = {
        "model_path": model_path,
        "temperature": 0.7,
        "top_p" : 1,
        "n_ctx": 2048,
        "callback_manager" : callback_manager,
        "max_tokens": 100,
        "verbose" : True, 
        "n_batch": 512,  # set this based on your GPU & CPU RAM
    }

prompt_template = "<s>[INST] "+"""You are a helpful robot assistant,
you will answer user questions by thinking step by step. 
Give out short answers. 
Human: {question}
Assistant:"""+" [/INST]"

prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
llm = LlamaCpp(**kwargs)
llm_chain = LLMChain(prompt=prompt, llm=llm,)

def tts(text):  
    sttstream.stop_stream() 
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate-50)
    engine.say(text)
    engine.runAndWait()

def stt():
    sttstream.start_stream()
    while True:
        data = sttstream.read(4096)
        if recogniser.AcceptWaveform(data):
            result = recogniser.Result()
            result_json = json.loads(result)
            text_content = result_json.get("text", "")
            if text_content.strip():  
                return text_content 
            
print("initialization completed system is now ready to respond ...... ")

while True:
    qes = stt() 
    print("user:",qes)
    ans = llm_chain.run(qes)
    print("robot:",ans)
    tts(ans)
