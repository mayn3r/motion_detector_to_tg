import asyncio
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums.parse_mode import ParseMode
from aiogram.client.default import DefaultBotProperties
from loguru import logger
from src.app.schemas import settings
# from src.app.middlewares import middlewares
# from src.app.routers import routers
from src.detector import MotionDetector

async def main():
    logger.info(f"Используется прокси: {settings.proxy_url}")
    
    session = AiohttpSession(proxy=settings.proxy_url)
    bot = Bot(
        token=settings.bot_token.get_secret_value(), 
        session=session, 
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()

    await bot.delete_webhook(drop_pending_updates=True)
    logger.debug("Webhook удален")

    loop = asyncio.get_running_loop()
    detector = MotionDetector(bot=bot, loop=loop)
    await asyncio.to_thread(detector.start)
    
    logger.success("Бот и детектор движения запущены")
    
    try:
        await dp.start_polling(bot)
    except asyncio.CancelledError:
        logger.info("Поллинг остановлен")
    finally:
        # 4. Корректное завершение работы (Graceful Shutdown)
        logger.info("⏳ Остановка детектора движения...")
        detector.stop()           # Даем команду циклу while остановиться
        await bot.session.close() # Закрываем сессию aiohttp, убираем warning "Unclosed client session"
        logger.success("Бот корректно остановлен")

if __name__ == "__main__":
    asyncio.run(main())