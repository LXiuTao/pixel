"""
像素艺术生成器主程序入口
"""

from gui import PixelArtApp
import tkinter as tk


def main():
    """主函数"""
    root = tk.Tk()
    app = PixelArtApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()