import asyncio
import websockets
import json
import openai
import base64
import shutil
import os
import subprocess

# Define API keys and voice ID
OPENAI_API_KEY = 'sk-'
ELEVENLABS_API_KEY = 'your api'
VOICE_ID = 'your_voiceID'
MODEL_ID = 'eleven_multilingual_v2'

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

def is_installed(lib_name):
    return shutil.which(lib_name) is not None

async def text_chunker(chunks):
    """Split text into chunks, ensuring to not break sentences."""
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ", "、", "。")
    buffer = ""

    async for text in chunks:
        if text is None:
            continue
        if buffer.endswith(splitters):
            yield buffer + " "
            buffer = text
        elif text.startswith(splitters):
            yield buffer + text[0] + " "
            buffer = text[1:]
        else:
            buffer += text

    if buffer:
        yield buffer + " "

async def stream(audio_stream):
    """Stream audio data using mpv player."""
    if not is_installed("mpv"):
        raise ValueError(
            "mpv not found, necessary to stream audio. "
            "Install instructions: https://mpv.io/installation/"
        )

    mpv_process = subprocess.Popen(
        ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    print("Started streaming audio")
    async for chunk in audio_stream:
        if chunk:
            mpv_process.stdin.write(chunk)
            mpv_process.stdin.flush()

    if mpv_process.stdin:
        mpv_process.stdin.close()
    mpv_process.wait()

async def text_to_speech_input_streaming(model_id, voice_id, text_iterator):
    """Send text to ElevenLabs API and stream the returned audio."""
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps({
                    "text": " ",
                    "voice_settings": {"stability": 0.5, "similarity_boost": True},
                    "xi_api_key": ELEVENLABS_API_KEY,
                }))

                async def listen():
                    """Listen to the websocket for audio data and stream it."""
                    while True:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            if data.get("audio"):
                                yield base64.b64decode(data["audio"])
                            elif data.get('isFinal'):
                                break
                        except websockets.exceptions.ConnectionClosed:
                            print("Connection closed")
                            break

                listen_task = asyncio.create_task(stream(listen()))

                async for text in text_chunker(text_iterator):
                    await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))

                await websocket.send(json.dumps({"text": ""}))

                await listen_task
                break  # Exit the loop if everything went well

        except websockets.exceptions.ConnectionClosed:
            print("Reconnecting...")
            await asyncio.sleep(1)  # Wait a bit before reconnecting

async def chat_completion(query):
    """Retrieve text from OpenAI and pass it to the text-to-speech function."""
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model='gpt-3.5-turbo', messages=[
            {'role': 'user', 'content': query}
        ],
        temperature=1, stream=True
    )

    async def text_iterator():
        async for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content'):
                    yield delta.content
            else:
                break

    await text_to_speech_input_streaming(MODEL_ID, VOICE_ID, text_iterator())

# Main execution
if __name__ == "__main__":
    questions = [
        'こんにちは',
        '何か興味深いこと言って',
        'What is your favorite movie youve ever wachted',
        'If you can get a holiday, where would you like to go?',
        'そろそろ終わるお時間です',
    ]
    for q in questions:
        print(q)
        asyncio.run(chat_completion(q))
