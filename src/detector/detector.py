import cv2
import time
import asyncio
import os
import tempfile
from aiogram import Bot
from aiogram.types import FSInputFile
from loguru import logger
from src.app.schemas import settings

class MotionDetector:
    def __init__(self, bot: Bot, loop: asyncio.AbstractEventLoop):
        self.bot = bot
        self.loop = loop  # Сохраняем ссылку на главный event loop для безопасных вызовов
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        
        # Параметры для записи видео
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 20  # Если камера не отдает FPS, берем 20
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Состояние записи
        self.is_recording = False
        self.recording_start_time = 0
        self.video_writer = None
        self.current_video_path = ""
        
        # Настройки детекции
        self.COOLDOWN = settings.cooldown + 2.0  # + 2 сек буфер, чтобы не начинать новую запись сразу
        self.AREA_THRESHOLD = settings.area_threshold
        self.last_motion_time = 0

        if not self.cap.isOpened():
            logger.error("Не удалось открыть веб-камеру")
            raise RuntimeError("Не удалось открыть веб-камеру")
            
        logger.info(f"Камера инициализирована: {self.width}x{self.height} @ {self.fps} FPS")
        
    async def send_info(self):
        await self.bot.send_message(
                chat_id=settings.chat_id,
                text=f"🚨 Обнаружено движение! Идет запись {settings.cooldown} сек видео"
            )

    async def send_alert(self, video_path: str):
        """Асинхронная отправка сообщения и видео в Telegram"""
        try:
            # # 1. Отправляем текстовое уведомление
            # await self.bot.send_message(
            #     chat_id=settings.chat_id,
            #     text="<b>🚨 Обнаружено движение!</b>\n🎥 Видео прилагается ниже."
            # )
            
            # 2. Отправляем видеофайл
            await self.bot.send_video(
                chat_id=settings.chat_id,
                video=FSInputFile(path=video_path, filename="file.mp4")
            )
            logger.success("Видео успешно отправлено в Telegram")
            
        except Exception as e:
            logger.error(f"Ошибка отправки в Telegram: {e}")
        finally:
            # 3. Обязательно удаляем временный файл после отправки (или ошибки)
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.debug(f"Временный файл удален: {video_path}")
                

    def start(self):
        ret, prev_frame = self.cap.read()
        if not ret:
            logger.error("Ошибка: не удалось прочитать первый кадр")
            self.cap.release()
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

        logger.info("Детектор движения запущен в отдельном потоке.")

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Не удалось получить кадр. Завершение работы детектора.")
                break

            # Детекция движения
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            frame_diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            current_time = time.time()
            motion_detected = False
            
            for contour in contours:
                if cv2.contourArea(contour) > self.AREA_THRESHOLD:
                    motion_detected = True
                    break
            
            # Логика записи видео
            if motion_detected and not self.is_recording:
                if current_time - self.last_motion_time > self.COOLDOWN:
                    logger.success(f"Обнаружено движение! Начинаю запись {settings.cooldown}-секундного видео...")
                    
                    asyncio.run_coroutine_threadsafe(
                        self.send_info(), 
                        self.loop
                    )
                    
                    # Создаем временный файл для видео
                    fd, self.current_video_path = tempfile.mkstemp(suffix=".mp4")
                    os.close(fd)
                    
                    # Инициализируем VideoWriter (кодек mp4v широко поддерживается)
                    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(
                        str(self.current_video_path), fourcc, self.fps, (self.width, self.height)
                    )
                    
                    self.is_recording = True
                    self.recording_start_time = current_time
                    self.last_motion_time = current_time # Сбрасываем кулдаун

            # Если идет запись, сохраняем кадр
            if self.is_recording:
                self.video_writer.write(frame) # type: ignore
                
                # Проверяем, прошло ли 10 секунд
                if time.time() - self.recording_start_time >= self.COOLDOWN-2:
                    self.video_writer.release() # type: ignore
                    self.is_recording = False
                    logger.info("Запись завершена. Отправка в Telegram...")
                    
                    # БЕЗОПАСНЫЙ вызов асинхронной функции из синхронного потока!
                    asyncio.run_coroutine_threadsafe(
                        self.send_alert(str(self.current_video_path)), 
                        self.loop
                    )

            prev_gray = gray.copy()

            # Опционально: показываем окно (раскомментируйте для локальной отладки)
            # cv2.imshow("Детектор движения", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     self.is_running = False
            #     break

        # Очистка при завершении
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
        self.cap.release()
        # cv2.destroyAllWindows()
        logger.info("Детектор движения остановлен.")

    def stop(self):
        """Метод для безопасной остановки цикла"""
        self.is_running = False