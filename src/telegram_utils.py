import asyncio
from aiogram import Bot
from aiogram.methods.send_message_draft import SendMessageDraft
from aiogram.methods.send_message import SendMessage
from aiogram.exceptions import TelegramRetryAfter
from dotenv import load_dotenv
import os

# Assume you have your bot token
load_dotenv()
token = os.getenv("TELEGRAM_BOT_TOKEN")
bot = Bot(token=token)

async def stream_to_telegram(chat_id, draft_id, llm_stream):
    full_text = ""
    last_update_time = 0
    # Update every 500ms to stay within Telegram's 1-req/sec limit
    update_interval = 0.5 

    try:
        async for chunk in llm_stream:
            full_text += chunk
            current_time = asyncio.get_event_loop().time()
            
            # Only send an update if enough time has passed
            if current_time - last_update_time > update_interval:
                try:
                    await bot(SendMessageDraft(chat_id=chat_id, draft_id=draft_id, text=full_text))
                    last_update_time = current_time
                except TelegramRetryAfter as e:
                    # If we hit a 429 here, handle it properly!
                    await asyncio.sleep(e.retry_after)
                except Exception as e:
                    print(f"Draft update error: {e}")
    finally:
        try:
            await bot.send_message(chat_id=chat_id, text=full_text)
        except Exception as e:
            print(f"Finalization failed: {e}")

    return full_text