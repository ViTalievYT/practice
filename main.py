import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np


def ask_integer_ru(title, prompt, minvalue=None, maxvalue=None):
    """Запрос целого числа с проверкой и русскими сообщениями об ошибках.
    При нажатии Отмена возвращает None сразу."""
    while True:
        val = simpledialog.askstring(title, prompt)
        if val is None:
            return None
        try:
            num = int(val)
            if minvalue is not None and num < minvalue:
                messagebox.showwarning("Ошибка", f"Число должно быть не меньше {minvalue}")
                continue
            if maxvalue is not None and num > maxvalue:
                messagebox.showwarning("Ошибка", f"Число должно быть не больше {maxvalue}")
                continue
            return num
        except ValueError:
            messagebox.showwarning("Ошибка", "Пожалуйста, введите корректное целое число.")


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        self.image = None          # Текущее изображение (с изменениями)
        self.base_image = None     # Исходное изображение без изменений
        self.tensor_image = None
        self.target_size = (640, 480)  # Размер для всех изображений
        self.negative_applied = False   # Флаг применения негатива

        self.canvas = tk.Label(self.root)
        self.canvas.pack()

        self.init_buttons()

    def init_buttons(self):
        btn_frame = tk.Frame(self.root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Загрузить изображение", command=self.load_image).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Сделать снимок", command=self.capture_image).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Красный канал", command=lambda: self.show_channel(0)).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Зелёный канал", command=lambda: self.show_channel(1)).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Синий канал", command=lambda: self.show_channel(2)).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Негатив", command=self.negative).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Добавить границу", command=self.add_border).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Нарисовать линию", command=self.draw_line).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Сбросить к оригиналу", command=self.reset_image).pack(side=tk.LEFT, padx=2)

    def load_image(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
            if not path:
                return
            img_pil = Image.open(path).convert("RGB")
            img_pil = img_pil.resize(self.target_size, Image.LANCZOS)
            img_rgb = np.array(img_pil)

            self.base_image = img_rgb.copy()
            self.negative_applied = False
            self.set_image_from_array(img_rgb)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")

    def capture_image(self):
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, self.target_size)
                self.base_image = img_resized.copy()
                self.negative_applied = False
                self.set_image_from_array(img_resized)
            else:
                messagebox.showwarning("Ошибка", "Не удалось получить изображение с камеры.")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def set_image_from_array(self, arr):
        self.image = arr
        self.tensor_image = torch.from_numpy(arr).float() / 255.0
        self.show_image(arr)

    def show_image(self, arr):
        img = Image.fromarray(arr.astype('uint8'))
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk

    def show_channel(self, channel_idx):
        if self.image is None:
            return
        zero = np.zeros_like(self.image)
        zero[:, :, channel_idx] = self.image[:, :, channel_idx]
        img_pil = Image.fromarray(zero.astype('uint8'))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk

    def negative(self):
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
        if self.image is None:
            messagebox.showwarning("Внимание", "Сначала загрузите или сделайте снимок изображения.")
            return
        size = ask_integer_ru("Граница", "Введите ширину границы (в пикселях):", 1, 200)
        if size is None:
            return
        bordered = cv2.copyMakeBorder(
            self.image, size, size, size, size,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        self.set_image_from_array(bordered)

    def draw_line(self):
        if self.image is None:
            messagebox.showwarning("Внимание", "Сначала загрузите или сделайте снимок изображения.")
            return

        x1 = ask_integer_ru("Координаты", "x1:", 0, self.image.shape[1])
        if x1 is None:
            return
        y1 = ask_integer_ru("Координаты", "y1:", 0, self.image.shape[0])
        if y1 is None:
            return
        x2 = ask_integer_ru("Координаты", "x2:", 0, self.image.shape[1])
        if x2 is None:
            return
        y2 = ask_integer_ru("Координаты", "y2:", 0, self.image.shape[0])
        if y2 is None:
            return
        thickness = ask_integer_ru("Толщина", "Введите толщину линии:", 1, 100)
        if thickness is None:
            return

        img_copy = self.image.copy()
        cv2.line(img_copy, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=thickness)
        self.set_image_from_array(img_copy)

    def reset_image(self):
        if self.base_image is None:
            messagebox.showwarning("Внимание", "Оригинальное изображение отсутствует.")
            return
        self.set_image_from_array(self.base_image.copy())
        self.negative_applied = False


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
