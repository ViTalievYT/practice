import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import sys
from contextlib import contextmanager
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np

@contextmanager
def suppress_opencv_output():
    """
    Контекстный менеджер для подавления вывода ошибок OpenCV в консоль.
    Используется для скрытия предупреждений и сообщений об ошибках OpenCV.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)# Открываем специальный файл для "поглощения" вывода
    old_stderr = os.dup(sys.stderr.fileno()) # Сохраняем текущий дескриптор stderr
    os.dup2(devnull, sys.stderr.fileno()) # Перенаправляем stderr в /dev/null
    try:
        yield
    finally:
        os.dup2(old_stderr, sys.stderr.fileno()) # Восстанавливаем оригинальный stderr
        os.close(devnull) # Закрываем /dev/null
        os.close(old_stderr)  # Закрываем сохранённый дескриптор


def ask_integer(title, prompt, minvalue=None, maxvalue=None):
    """
    Запрос целого числа через диалоговое окно с сообщениями об ошибках.
    Возвращает введённое число или None, если пользователь нажал Отмена.
    Параметры minvalue и maxvalue задают границы допустимых значений.
    """
    while True:
        val = simpledialog.askstring(title, prompt)          # Запрашиваем строку у пользователя
        if val is None:                                       # Пользователь нажал Отмена
            return None
        try:
            num = int(val)                                    # Пробуем преобразовать в целое число
            if minvalue is not None and num < minvalue:
                messagebox.showwarning("Ошибка", f"Число должно быть не меньше {minvalue}")
                continue
            if maxvalue is not None and num > maxvalue:
                messagebox.showwarning("Ошибка", f"Число должно быть не больше {maxvalue}")
                continue
            return num
        except ValueError:
            messagebox.showwarning("Ошибка", "Пожалуйста, введите корректное целое число.")


def get_available_camera_indices(max_index=10):
    """
    Проверяет доступность камер с индексами от 0 до max_index-1.
    Возвращает список индексов камер, которые успешно открываются.
    """
    available = []
    for i in range(max_index):
        with suppress_opencv_output():
            cap = cv2.VideoCapture(i)       # Пытаемся открыть камеру с индексом i
            if cap.isOpened():
                ret, _ = cap.read()         # Пробуем прочитать кадр
                if ret:
                    available.append(i)     # Камера доступна
                cap.release()
    return available


class ImageApp:
    """
    Основное приложение с GUI на tkinter для работы с изображениями и камерой.
    Поддерживает загрузку, съемку, отображение цветовых каналов,
    применение негатива, добавление границ, рисование линии и сброс к оригиналу.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Обработка изображений")

        # Переменные для хранения изображений и состояния
        self.image = None          # Текущее изображение (с изменениями)
        self.base_image = None     # Исходное изображение без изменений
        self.tensor_image = None   # Тензор PyTorch для возможной дальнейшей обработки
        self.target_size = (640, 480)  # Стандартный размер изображений
        self.negative_applied = False   # Флаг, применён ли негатив
        self.selected_camera_index = 0  # Индекс выбранной камеры

        self.canvas = tk.Label(self.root)  # Виджет для отображения изображения
        self.canvas.pack()

        self.init_buttons()  # Инициализация кнопок управления

    def init_buttons(self):
        """
        Создаёт и размещает кнопки управления приложением на главном окне.
        """
        btn_frame = tk.Frame(self.root)
        btn_frame.pack()

        # Кнопки для выбора камеры, загрузки изображения, съемки с камеры,
        # показа отдельных цветовых каналов, негатив, граница, линия, сброс
        (tk.Button(btn_frame, text="Выбрать камеру", command=self.select_camera)
         .pack(side=tk.LEFT, padx=2))
        (tk.Button(btn_frame, text="Загрузить изображение", command=self.load_image)
         .pack(side=tk.LEFT, padx=2))
        (tk.Button(btn_frame, text="Сделать снимок", command=self.capture_image)
         .pack(side=tk.LEFT, padx=2))
        (tk.Button(btn_frame, text="Красный канал", command=lambda: self.show_channel(0))
         .pack(side=tk.LEFT, padx=2))
        (tk.Button(btn_frame, text="Зелёный канал", command=lambda: self.show_channel(1))
         .pack(side=tk.LEFT, padx=2))
        (tk.Button(btn_frame, text="Синий канал", command=lambda: self.show_channel(2))
         .pack(side=tk.LEFT, padx=2))
        (tk.Button(btn_frame, text="Негатив", command=self.negative).
         pack(side=tk.LEFT, padx=2))
        (tk.Button(btn_frame, text="Добавить границу", command=self.add_border)
         .pack(side=tk.LEFT, padx=2))
        (tk.Button(btn_frame, text="Нарисовать линию", command=self.draw_line)
         .pack(side=tk.LEFT, padx=2))
        (tk.Button(btn_frame, text="Сбросить к оригиналу", command=self.reset_image)
         .pack(side=tk.LEFT, padx=2))

    def select_camera(self):
        """
        Запускает диалог выбора доступной камеры.
        Позволяет выбрать индекс камеры из списка доступных.
        """
        indices = get_available_camera_indices()
        if not indices:
            messagebox.showerror("Нет камер", "Не найдено доступных камер.")
            return

        index = ask_integer("Выбор камеры", f"Доступные индексы: "
                                            f"{indices}\nВведите номер камеры:", 0, max(indices))
        if index is None:
            return
        if index in indices:
            self.selected_camera_index = index
            messagebox.showinfo("Камера выбрана",
                                f"Выбрана камера с индексом {index}")
        else:
            messagebox.showwarning("Ошибка", "Недопустимый индекс.")

    def load_image(self):
        """
        Открывает диалог выбора файла изображения, загружает его,
        изменяет размер до target_size и отображает.
        """
        try:
            path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg")])
            if not path:
                return
            img_pil = Image.open(path).convert("RGB") # Загружаем изображение и переводим в RGB
            img_pil = img_pil.resize(self.target_size, Image.LANCZOS) # Изменяем размер изображения
            img_rgb = np.array(img_pil) # Преобразуем в numpy-массив

            self.base_image = img_rgb.copy() # Сохраняем исходное изображение
            self.negative_applied = False # Сброс флага негатива
            self.set_image_from_array(img_rgb) # Отображаем изображение
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")

    def capture_image(self):
        """
        Снимает кадр с выбранной камеры, изменяет размер и отображает.
        Обрабатывает ошибки при отсутствии камеры или проблемах с захватом.
        """
        try:
            with suppress_opencv_output():
                cap = cv2.VideoCapture(self.selected_camera_index)
                if not cap.isOpened():
                    messagebox.showerror("Ошибка",
                                         f"Не удалось открыть камеру с индексом "
                                         f"{self.selected_camera_index}")
                    return
                ret, frame = cap.read()
                cap.release()
            if ret:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Конвертируем BGR (OpenCV) в RGB
                img_resized = cv2.resize(img_rgb, self.target_size) # Изменяем размер
                self.base_image = img_resized.copy() # Сохраняем исходное изображение
                self.negative_applied = False # Сброс флага негатива
                self.set_image_from_array(img_resized) # Отображаем изображение
            else:
                messagebox.showwarning("Ошибка", "Не удалось получить изображение с камеры.")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def set_image_from_array(self, arr):
        """
        Устанавливает текущее изображение из numpy-массива.
        Также конвертирует изображение в тензор PyTorch для дальнейшей обработки.
        """
        self.image = arr
        self.tensor_image = torch.from_numpy(arr).float() / 255.0
        self.show_image(arr)

    def show_image(self, arr):
        """
        Отображает numpy-массив изображения в виджете tkinter.Label.
        """
        img = Image.fromarray(arr.astype('uint8'))
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk

    def show_channel(self, channel_idx):
        """
        Отображает один из цветовых каналов (0 - красный, 1 - зеленый, 2 - синий).
        Остальные каналы зануляются.
        """
        if self.image is None:
            messagebox.showwarning("Внимание", "Сначала загрузите или сделайте снимок изображения.")
            return
        zero = np.zeros_like(self.image)
        zero[:, :, channel_idx] = self.image[:, :, channel_idx]  # Копируем только выбранный канал
        img_pil = Image.fromarray(zero.astype('uint8'))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk

    def negative(self):
        """
        Применяет негатив к исходному изображению.
        Предотвращает повторное применение негатива.
        """
        if self.image is None:
            messagebox.showwarning("Внимание", "Сначала загрузите или сделайте снимок изображения.")
            return
        if self.negative_applied:
            messagebox.showinfo("Инфо", "Негатив уже применён.")
            return
        neg = 255 - self.base_image
        self.set_image_from_array(neg)
        self.negative_applied = True

    def add_border(self):
        """
        Добавляет чёрную границу по периметру изображения.
        Ширина границы запрашивается у пользователя.
        """
        if self.image is None:
            messagebox.showwarning("Внимание", "Сначала загрузите или сделайте снимок изображения.")
            return
        size = ask_integer("Граница", "Введите ширину границы (в пикселях):", 1, 200)
        if size is None:
            return
        bordered = cv2.copyMakeBorder(
            self.image, size, size, size, size,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]  # Чёрная граница
        )
        self.set_image_from_array(bordered)

    def draw_line(self):
        """
        Запрашивает координаты начала и конца линии, а также толщину,
        рисует зелёную линию на текущем изображении.
        """
        if self.image is None:
            messagebox.showwarning("Внимание", "Сначала загрузите или сделайте снимок изображения.")
            return

        x1 = ask_integer("Координаты", "x1:", 0, self.image.shape[1])
        if x1 is None:
            return
        y1 = ask_integer("Координаты", "y1:", 0, self.image.shape[0])
        if y1 is None:
            return
        x2 = ask_integer("Координаты", "x2:", 0, self.image.shape[1])
        if x2 is None:
            return
        y2 = ask_integer("Координаты", "y2:", 0, self.image.shape[0])
        if y2 is None:
            return
        thickness = ask_integer("Толщина", "Введите толщину линии:", 1, 100)
        if thickness is None:
            return

        img_copy = self.image.copy()
        cv2.line(img_copy, (x1, y1), (x2, y2), color=(0, 255, 0),
                 thickness=thickness)  # Рисуем зелёную линию
        self.set_image_from_array(img_copy)

    def reset_image(self):
        """
        Сбрасывает текущее изображение к исходному, загруженному или снятому с камеры.
        """
        if self.base_image is None:
            messagebox.showwarning("Внимание", "Оригинальное изображение отсутствует.")
            return
        self.set_image_from_array(self.base_image.copy())
        self.negative_applied = False


if __name__ == "__main__":
    root = tk.Tk()               # Создаём корневое окно
    app = ImageApp(root)         # Инициализируем приложение
    root.mainloop()              # Запускаем главный цикл обработки событий
