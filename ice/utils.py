"""
@Author: Zhang Ruihan
@Date: 2020-01-15 05:50:01
@LastEditors: Thao Le
@Description: file content
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

EPSILON = 1e-8


class ImageUtils:
    def __init__(
        self,
        img_size=(224, 224),
        nchannels=3,
        img_format="channels_first",
        mode=None,
        std=None,
        mean=None,
    ):
        self.img_format = img_format
        self.nchannels = nchannels
        self.img_width = img_size[0]
        self.img_height = img_size[1]
        self.std = std
        self.mean = mean
        self.mode = mode
        # 8 directions: horizontally, vertically and diagnonally
        self.directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    def deprocessing(self, x):
        x = np.array(x)
        X = x.copy()

        if self.img_format == "channels_first":
            if X.ndim == 3:
                X = np.transpose(X, (1, 2, 0))
            else:
                X = np.transpose(X, (0, 2, 3, 1))

        if self.mode is None:
            return X
        elif self.mode == "ham10000":
            mean = self.mean
            std = self.std

        if std is not None:
            X[..., 0] *= std[0]
            X[..., 1] *= std[1]
            X[..., 2] *= std[2]
        X[..., 0] += mean[0]
        X[..., 1] += mean[1]
        X[..., 2] += mean[2]

        if self.mode == "ham10000":
            X *= 255
        return X

    def resize_img(self, array, smooth=False):
        size = array.shape
        if smooth:
            tsize = list(size)
            tsize[1] = self.img_width
            tsize[2] = self.img_height
            res = resize(array, tsize, order=1, mode="reflect", anti_aliasing=False)
        else:
            res = []
            for i in range(size[0]):
                temp = array[i]
                temp = np.repeat(temp, self.img_width // size[1], axis=0)
                temp = np.repeat(temp, self.img_height // size[2], axis=1)
                res.append(temp)
            res = np.array(res)
        return res

    def show_img(
        self,
        X,
        nrows=1,
        ncols=1,
        heatmaps=None,
        useColorBar=True,
        deprocessing=True,
        save_path=None,
        is_show=True,
    ):
        X = np.array(X)
        if not heatmaps is None:
            heatmaps = np.array(heatmaps)
        if len(X.shape) < 4:
            print("Dim should be 4")
            return

        X = np.array(X)
        if deprocessing:
            X = self.deprocessing(X)

        if (not X.min() == 0) or X.max() > 1:
            X = X - X.min()
            X = X / X.max()

        if X.shape[0] == 1:
            X = np.squeeze(X)
            X = np.expand_dims(X, axis=0)
        else:
            X = np.squeeze(X)

        if self.nchannels == 1:
            cmap = "Greys"
        else:
            cmap = "viridis"

        l = nrows * ncols
        plt.figure(figsize=(5 * ncols, 5 * nrows))
        for i in range(l):
            plt.subplot(nrows, ncols, i + 1)
            plt.axis("off")
            img = X[i]
            img = np.clip(img, 0, 1)
            plt.imshow(img, cmap=cmap)
            if not heatmaps is None:
                if not heatmaps[i] is None:
                    heapmap = heatmaps[i]
                    plt.imshow(heapmap, cmap="jet", alpha=0.5, interpolation="bilinear")
                    if useColorBar:
                        plt.colorbar()

        if save_path is not None:
            plt.savefig(save_path)
        if is_show:
            plt.show()

    def img_filter(
        self, x, h, threshold=0.5, background=0.2, smooth=True, minmax=False
    ):
        x = x.copy()
        h = h.copy()

        if minmax:
            h = h - h.min()

        h = h * (h > 0)
        for i in range(h.shape[0]):
            h[i] = h[i] / (h[i].max() + EPSILON)

        h = (h - threshold) * (1 / (1 - threshold))

        # h = h * (h>0)

        h = self.resize_img(h, smooth=smooth)
        h = (h > 0).astype("float") * (1 - background) + background
        h_mask = np.repeat(h, self.nchannels).reshape(list(h.shape) + [-1])
        if self.img_format == "channels_first":
            h_mask = np.transpose(h_mask, (0, 3, 1, 2))  # transpose h to channel first
        x = x * h_mask

        h = h - h.min()
        h = h / (h.max() + EPSILON)

        return x, h

    def DFS(self, grid, rows, cols, cr, cc, visited):
        """
        cr: current row
        cc: current column
        """
        # Stack for DFS
        stack = [(cr, cc)]
        island = []

        while stack:
            x, y = stack.pop()

            if visited[x][y]:
                continue

            visited[x][y] = True
            island.append((x, y))
            for d in self.directions:
                nr = x + d[0]
                nc = y + d[1]
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr][nc] and grid[nr][nc] > 0:
                        stack.append((nr, nc))

        return island

    def find_max_area_contour(self, grid):
        rows = len(grid)
        cols = len(grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        max_area = 0
        max_island = []
        for r in range(rows):
            for c in range(cols):
                if not visited[r][c] and grid[r][c] > 0:
                    island = self.DFS(grid, rows, cols, r, c, visited)
                    if len(island) > max_area:
                        max_island = island
                        max_area = len(island)
        result = np.zeros_like(grid, dtype=int)
        
        # Set 1 for elements in the largest island
        for x, y in max_island:
            result[x, y] = 1
        return result

    def contour_img(self, x, h, dpi=100):
        dpi = float(dpi)
        size = x.shape
        if x.max() > 1:
            x = x / x.max()
        fig = plt.figure(figsize=(size[1] / dpi, size[0] / dpi), dpi=dpi)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        xa = np.linspace(0, size[1] - 1, size[1])
        ya = np.linspace(0, size[0] - 1, size[0])
        X, Y = np.meshgrid(xa, ya)
        if x.shape[-1] == 1:
            x = np.squeeze(x)
            ax.imshow(x, cmap="Greys")
        else:
            ax.imshow(x)
        ax.contour(X, Y, h, colors="r")
        return fig
