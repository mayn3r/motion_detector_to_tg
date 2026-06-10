import cv2
import time
import asyncio
import os
import tempfile
import threading
import subprocess
import sounddevice as sd
import soundfile as sf
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
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 20
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Состояние записи
        self.is_recording = False
        self.recording_start_time = 0
        self.video_writer = None
        
        # Пути к временным файлам
        self.current_video_path = ""
        self.current_audio_path = ""
        self.final_video_path = ""
        self.audio_thread = None
        
        # Настройки детекции
        self.COOLDOWN = settings.cooldown + 2.0  # + 2 сек буфер
        self.AREA_THRESHOLD = settings.area_threshold 
        self.last_motion_time = 0
        
        # Настройки аудио
        self.audio_sample_rate = 44100

        if not self.cap.isOpened():
            logger.error("Не удалось открыть веб-камеру")
            raise RuntimeError("Не удалось открыть веб-камеру")
            
        logger.info(f"Камера инициализирована: {self.width}x{self.height} @ {self.fps} FPS")
    
    async def send_info(self):
        await self.bot.send_message(
            chat_id=settings.chat_id,
            text=f"🚨 Обнаружено движение! Идет запись {settings.cooldown} сек видео со звуком..."
        )

    async def send_alert(self, video_path: str):
        """Асинхронная отправка сообщения и видео в Telegram"""
        try:
            await self.bot.send_video(
                chat_id=settings.chat_id,
                video=FSInputFile(path=video_path, filename="motion_alert.mp4")
            )
            logger.success("Видео со звуком успешно отправлено в Telegram")
        except Exception as e:
            logger.error(f"Ошибка отправки в Telegram: {e}")
        finally:
            # Удаляем ВСЕ временные файлы после отправки
            self._cleanup_temp_files()

    def _record_audio(self, file_path: str, duration: float):
        """Функция для записи аудио в отдельном потоке"""
        logger.debug(f"Начало записи аудио на {duration} сек...")
        try:
            # Записываем аудио заданной длительности
            audio_data = sd.rec(
                int(duration * self.audio_sample_rate), 
                samplerate=self.audio_sample_rate, 
                channels=1, 
                dtype='int16'
            )
            sd.wait()  # Ждем окончания записи
            sf.write(file_path, audio_data, self.audio_sample_rate)
            logger.debug("Запись аудио завершена.")
        except Exception as e:
            logger.error(f"Ошибка записи аудио: {e}")

    def _merge_audio_video(self, video_path: str, audio_path: str, output_path: str):
        """Объединяет немое видео и аудио с помощью FFmpeg"""
        logger.debug("Объединение видео и аудио через FFmpeg...")
        try:
            # -y: перезаписывать файл без спроса
            # -c:v copy: копируем видеопоток без перекодирования (очень быстро)
            # -c:a aac: кодируем аудио в формат AAC (стандарт для MP4)
            # -shortest: заканчиваем файл, когда заканчивается самый короткий поток (видео или аудио)
            command = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                output_path
            ]
            # Запускаем FFmpeg и скрываем его вывод, если нет ошибок
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg ошибка: {result.stderr}")
            else:
                logger.success("Видео и аудио успешно объединены!")
        except FileNotFoundError:
            logger.error("Не найден FFmpeg! Убедитесь, что он установлен и добавлен в PATH.")
        except Exception as e:
            logger.error(f"Ошибка при объединении файлов: {e}")

    def _cleanup_temp_files(self):
        """Безопасное удаление всех временных файлов"""
        for path in [self.current_video_path, self.current_audio_path, self.final_video_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"Временный файл удален: {path}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить файл {path}: {e}")

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
            
            # Логика записи видео и аудио
            if motion_detected and not self.is_recording:
                if current_time - self.last_motion_time > self.COOLDOWN:
                    logger.success(f"Обнаружено движение! Начинаю запись {settings.cooldown}-секундного видео со звуком...")
                    
                    asyncio.run_coroutine_threadsafe(self.send_info(), self.loop)
                    
                    # 1. Создаем временные файлы
                    fd_vid, self.current_video_path = tempfile.mkstemp(suffix=".mp4")
                    os.close(fd_vid)
                    
                    fd_aud, self.current_audio_path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd_aud)
                    
                    fd_final, self.final_video_path = tempfile.mkstemp(suffix=".mp4")
                    os.close(fd_final)
                    
                    # 2. Инициализируем VideoWriter
                    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(
                        str(self.current_video_path), fourcc, self.fps, (self.width, self.height)
                    )
                    
                    # 3. Вычисляем точную длительность записи (как в вашем оригинальном коде)
                    record_duration = self.COOLDOWN - 2.0 
                    
                    # 4. Запускаем запись аудио в отдельном потоке
                    self.audio_thread = threading.Thread(
                        target=self._record_audio, 
                        args=(self.current_audio_path, record_duration)
                    )
                    self.audio_thread.start()
                     
                    self.is_recording = True
                    self.recording_start_time = current_time
                    self.last_motion_time = current_time # Сбрасываем кулдаун
    
            # Если идет запись, сохраняем кадр
            if self.is_recording:
                self.video_writer.write(frame) # type: ignore
                
                # Проверяем, прошло ли нужное время
                if time.time() - self.recording_start_time >= (self.COOLDOWN - 2.0):
                    self.video_writer.release() # type: ignore
                    self.is_recording = False
                    
                    # Ждем, пока поток записи аудио гарантированно завершится и сохранит файл
                    if self.audio_thread and self.audio_thread.is_alive():
                        self.audio_thread.join()
                    
                    logger.info("Запись завершена. Объединение и отправка в Telegram...")
                    
                    # Объединяем видео и аудио
                    self._merge_audio_video(
                        self.current_video_path, 
                        self.current_audio_path, 
                        self.final_video_path
                    )
                    
                    # БЕЗОПАСНЫЙ вызов асинхронной функции из синхронного потока!
                    # Передаем путь к ФИНАЛЬНОМУ файлу со звуком
                    asyncio.run_coroutine_threadsafe(
                        self.send_alert(str(self.final_video_path)), 
                        self.loop
                    )

            prev_gray = gray.copy()

        # Очистка при завершении
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
        self.cap.release()
        logger.info("Детектор движения остановлен.")

    def stop(self):
        """Метод для безопасной остановки цикла"""
        self.is_running = False