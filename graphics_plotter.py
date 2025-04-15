import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


class GraphicsPlotter:
    def __init__(self, descent):
        self.descent = descent
        bounds = self.descent.get_bounds()
        self.is_1d = len(bounds) == 1

    @staticmethod
    def _setup_plot_style():
        sns.set_theme(style="white", context="talk", palette="deep")
        plt.rcParams.update({
            'figure.facecolor': '#f5f5f5',
            'axes.facecolor': 'white'
        })

    @staticmethod
    def _create_figure(figsize=(12, 9), dpi=1000, projection=None):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d' if projection == '3d' else None)
        return fig, ax

    @staticmethod
    def _finalize_plot(ax, xlabel, ylabel, title, is_1d=False):
        ax.set_xlabel(xlabel, fontsize=16, labelpad=15)
        ax.set_ylabel(ylabel if is_1d else 'y', fontsize=16, labelpad=15)
        ax.set_title(title, fontsize=20, pad=25, fontweight='bold')
        ax.legend(fontsize=14, loc='upper right', frameon=True, edgecolor='black',
                  facecolor='white', framealpha=0.95)
        plt.tight_layout()

    def _plot_levels(self, ax):
        bounds = [np.linspace(start, end, 1000) for start, end in self.descent.get_bounds()]
        if self.is_1d:
            x = bounds[0]
            f_values = self.descent.get_f().call_without_memorization(x)
            ax.plot(x, f_values, color=sns.color_palette("flare")[2], linewidth=2, alpha=0.9, label='Функция')
        else:
            grid = np.meshgrid(*bounds)
            f_grid = self.descent.get_f().call_without_memorization(grid)
            contourf = ax.contourf(*grid, f_grid, levels=50, cmap="flare", alpha=0.9)
            contours = ax.contour(*grid, f_grid, levels=15, colors='black', linewidths=1, alpha=0.8)
            ax.clabel(contours, inline=True, fontsize=12, fmt='%.1f', colors='black')
            cbar = plt.colorbar(contourf, pad=0.05, fraction=0.046, aspect=25)
            cbar.set_label('Значение функции', fontsize=14, labelpad=15)

    def _plot_trajectory(self, ax):
        path = np.array(self.descent.get_path())
        if self.is_1d:
            x_path = path
            y_path = self.descent.get_f().call_without_memorization(x_path)
            ax.plot(x_path, y_path, color='red', linewidth=3, alpha=0.9, label='Путь')
            ax.scatter(x_path[0], y_path[0], color='lime', s=250, edgecolor='black', label='Старт')
            ax.scatter(x_path[-1], y_path[-1], color='red', s=250, edgecolor='black', label='Минимум')
        else:
            ax.plot(path[:, 0], path[:, 1], color='red', linewidth=3, alpha=0.9, label='Путь')
            ax.scatter(path[0, 0], path[0, 1], color='lime', s=250, edgecolor='black', label='Старт')
            ax.scatter(path[-1, 0], path[-1, 1], color='red', s=250, edgecolor='black', label='Минимум')

    def plot(self, title="Оптимизация функции методом градиентного спуска"):
        """
            :param title: заголовок графика
        """
        GraphicsPlotter._setup_plot_style()
        fig, ax = GraphicsPlotter._create_figure()
        self._plot_levels(ax)
        self._plot_trajectory(ax)
        sns.despine(left=False, bottom=False)
        GraphicsPlotter._finalize_plot(ax, 'x', 'f(x)', title, self.is_1d)
        plt.show()

    def plot_3d(self, title="3D Оптимизация градиентным спуском"):
        """
            :param title: заголовок графика
        """
        if self.is_1d:
            raise ValueError("3D-график доступен только для 2D-функций.")
        GraphicsPlotter._setup_plot_style()
        fig, ax = GraphicsPlotter._create_figure(projection='3d')
        bounds = [np.linspace(start, end, 50) for start, end in self.descent.get_bounds()]
        grid = np.meshgrid(*bounds)
        f_grid = self.descent.get_f()(grid)
        surf = ax.plot_surface(*grid, f_grid, cmap="flare", alpha=0.7, edgecolor='none')
        path = np.array(self.descent.get_path())
        ax.plot(path[:, 0], path[:, 1], self.descent.get_f()([path[:, 0], path[:, 1]]),
                color='red', linewidth=3, alpha=1.0, label='Путь')
        ax.scatter(path[0, 0], path[0, 1], self.descent.get_f()([path[0, 0], path[0, 1]]),
                   color='lime', s=250, edgecolor='black', label='Старт')
        ax.scatter(path[-1, 0], path[-1, 1], self.descent.get_f()([path[-1, 0], path[-1, 1]]),
                   color='red', s=250, edgecolor='black', label='Минимум')
        ax.view_init(elev=30, azim=135)
        ax.set_xlabel('x', fontsize=16, labelpad=20)
        ax.set_ylabel('y', fontsize=16, labelpad=20)
        ax.set_zlabel('f(x, y)', fontsize=16, labelpad=20)
        ax.set_title(title, fontsize=20, pad=25, fontweight='bold')
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=25, pad=0.1, fraction=0.046)
        cbar.set_label('Значение функции', fontsize=14, labelpad=15)
        ax.legend(fontsize=14, loc='upper right', frameon=True, edgecolor='black', facecolor='white', framealpha=0.95)
        plt.show()

    def generate_2d_anim(self, title="Анимация градиентного спуска"):
        """
            :param title: заголовок графика
        """
        GraphicsPlotter._setup_plot_style()
        fig, ax = GraphicsPlotter._create_figure()
        self._plot_levels(ax)
        path = np.array(self.descent.get_path())
        if path.ndim == 1 and not self.is_1d:
            raise ValueError("Для 2D анимации путь должен быть двумерным массивом (n, 2)")
        elif path.ndim == 2 and self.is_1d:
            raise ValueError("Для 1D анимации путь должен быть одномерным массивом")
        line, = ax.plot([], [], color='red', linewidth=3, alpha=0.9, label='Путь')
        start_point, = ax.plot([], [], 'o', color='lime', markersize=15, markeredgecolor='black', label='Старт')
        current_point, = ax.plot([], [], 'o', color='red', markersize=15, markeredgecolor='black',
                                 label='Текущая позиция')
        GraphicsPlotter._finalize_plot(ax, 'x', 'f(x)', title, self.is_1d)

        def init():
            line.set_data([], [])
            start_point.set_data([], [])
            current_point.set_data([], [])
            return line, start_point, current_point

        def update(frame):
            if self.is_1d:
                x_path = path[:frame + 1]
                y_path = self.descent.get_f()([x_path])
                if np.isscalar(y_path):
                    y_path = np.array([y_path])
                line.set_data(x_path, y_path)
                start_point.set_data([path[0]], [self.descent.get_f()([path[0]])])
                current_point.set_data([path[frame]], [self.descent.get_f()([path[frame]])])
            else:
                line.set_data(path[:frame + 1, 0], path[:frame + 1, 1])
                start_point.set_data([path[0, 0]], [path[0, 1]])
                current_point.set_data([path[frame, 0]], [path[frame, 1]])
            return line, start_point, current_point

        anim = FuncAnimation(fig, update, init_func=init, frames=len(path),
                             interval=200, blit=False, repeat=True)
        plt.close(fig)
        return anim

    def generate_3d_anim(self, title="Анимация градиентного спуска"):
        """
            :param title: заголовок графика
        """
        if self.is_1d:
            raise ValueError("3D-анимация доступна только для 2D-функций.")
        GraphicsPlotter._setup_plot_style()
        fig, ax = GraphicsPlotter._create_figure(projection='3d')
        bounds = [np.linspace(start, end, 50) for start, end in self.descent.get_bounds()]
        grid = np.meshgrid(*bounds)
        f_grid = self.descent.get_f()(grid)
        surf = ax.plot_surface(*grid, f_grid, cmap="flare", alpha=0.7, edgecolor='none')
        path = np.array(self.descent.get_path())
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError("Для 3D анимации путь должен быть двумерным массивом (n, 2)")
        line, = ax.plot([], [], [], color='red', linewidth=3, alpha=1.0, label='Путь')
        start_point, = ax.plot([], [], [], 'o', color='lime', markersize=15, markeredgecolor='black', label='Старт')
        current_point, = ax.plot([], [], [], 'o', color='red', markersize=15, markeredgecolor='black',
                                 label='Текущая позиция')
        ax.view_init(elev=30, azim=135)
        ax.set_xlabel('x', fontsize=16, labelpad=20)
        ax.set_ylabel('y', fontsize=16, labelpad=20)
        ax.set_zlabel('f(x, y)', fontsize=16, labelpad=20)
        ax.set_title(title, fontsize=20, pad=25, fontweight='bold')
        ax.legend(fontsize=14, loc='upper right', frameon=True, edgecolor='black', facecolor='white', framealpha=0.95)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=25, pad=0.1, fraction=0.046)
        cbar.set_label('Значение функции', fontsize=14, labelpad=15)

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            start_point.set_data([], [])
            start_point.set_3d_properties([])
            current_point.set_data([], [])
            current_point.set_3d_properties([])
            return line, start_point, current_point

        def update(frame):
            line.set_data(path[:frame + 1, 0], path[:frame + 1, 1])
            line.set_3d_properties(self.descent.get_f()([path[:frame + 1, 0], path[:frame + 1, 1]]))
            start_point.set_data([path[0, 0]], [path[0, 1]])
            start_point.set_3d_properties([self.descent.get_f()([path[0, 0], path[0, 1]])])
            current_point.set_data([path[frame, 0]], [path[frame, 1]])
            current_point.set_3d_properties([self.descent.get_f()([path[frame, 0], path[frame, 1]])])
            return line, start_point, current_point

        anim = FuncAnimation(fig, update, init_func=init, frames=len(path),
                             interval=200, blit=False, repeat=True)
        plt.close(fig)
        return anim

    def animate_2d(self):
        anim = self.generate_2d_anim()
        html = HTML(anim.to_jshtml())
        return html

    def animate_3d(self):
        anim = self.generate_3d_anim()
        html = HTML(anim.to_jshtml())
        return html

    def save_animation(self, filename='animation.gif', animation_type='2d', fps=5):
        if animation_type not in ['2d', '3d']:
            raise ValueError("animation_type должен быть '2d' или '3d'")

        if animation_type == '2d':
            anim = self.generate_2d_anim()
        else:
            anim = self.generate_3d_anim()

        anim.save(filename, writer='pillow', fps=fps)
        print(f"Анимация сохранена как {filename}")