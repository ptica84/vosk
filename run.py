import logging
import asyncio
import sys
import json
from vosk import Model, KaldiRecognizer
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.server import AsyncServer, AsyncEventHandler

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s')
_LOGGER = logging.getLogger(__name__)

RAW_INFO = {
    "type": "info",
    "data": {
        "asr": [{
            "name": "vosk",
            "description": "Vosk STT",
            "attribution": {"name": "Vosk", "url": "https://alphacephei.com/vosk/"},
            "installed": True,
            "version": "1.0.0",
            "models": [{
                "name": "vosk-model-ru",
                "description": "Russian Model",
                "attribution": {"name": "Vosk", "url": ""},
                "installed": True,
                "languages": ["ru"],
                "version": "1.0.0"
            }]
        }]
    }
}

class VoskEventHandler(AsyncEventHandler):
    def __init__(self, recognizer, reader, writer):
        super().__init__(reader, writer)
        self.recognizer = recognizer
        peer = writer.get_extra_info('peername')
        _LOGGER.info(f"=== СОЕДИНЕНИЕ: {peer} ===")

    async def handle_event(self, event: Event) -> bool:
        _LOGGER.info(f"Получен запрос от HA: {event.type}")
        
        # ИСПРАВЛЕНИЕ: Слушаем 'describe', отвечаем 'info'
        if event.type == "describe":
            _LOGGER.info("Отправляю метаданные...")
            await self.write_event(Event(type=RAW_INFO["type"], data=RAW_INFO["data"]))
            return True
        
        if AudioStart.is_type(event.type):
            self.recognizer.Reset()
            return True
        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.recognizer.AcceptWaveform(chunk.audio)
            return True
        elif AudioStop.is_type(event.type):
            result = json.loads(self.recognizer.FinalResult())
            text = result.get("text", "")
            _LOGGER.info(f"Распознано: {text}")
            await self.write_event(Transcript(text=text).event())
            return False
        
        return True

async def main():
    _LOGGER.info("1. Загрузка модели Vosk (это займет время)...")
    model = Model("/model")
    
    _LOGGER.info("2. Предварительная инициализация распознавателя...")
    recognizer = KaldiRecognizer(model, 16000)
    
    _LOGGER.info("3. Запуск сервера...")
    server = AsyncServer.from_uri("tcp://0.0.0.0:10420")
    _LOGGER.info("=== СЕРВЕР WYOMING ГОТОВ И СЛУШАЕТ ПОРТ 10420 ===")
    
    await server.run(lambda r, w: VoskEventHandler(recognizer, r, w))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        _LOGGER.error(f"ОШИБКА: {e}")