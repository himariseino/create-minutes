# MP4ファイルの取り込みと音声認識
import whisper

audio_file_name = "audio.mp4"

model = whisper.load_model("base")
result = model.transcribe(audio_file_name)
print(result["text"])

# テキスト要約と議事録作成
from langchain.chains import LLMChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.5)
prompt = "以下の会議内容を要約し、議事録を作成してください: {result['text']}"

chain = LLMChain(llm=llm, prompt=prompt)
summary = chain.run(result['text'])

# 