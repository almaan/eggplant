import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PIL import Image
from PIL.ImageQt import ImageQt

import numpy as np
import pandas as pd

import os
import os.path as osp
import anndata as ad

from typing import Tuple, Union, Optional


def get_adata_sample_name(adata) -> str:
    return list(adata.uns["spatial"].keys())[0]


def get_image_from_anndata(
    adata: ad.AnnData,
) -> Union[Image.Image, float]:

    spatial = adata.uns["spatial"]
    sample_name = get_adata_sample_name(adata)
    image = spatial[sample_name]["images"]["hires"]
    if image.max() <= 1:
        image = image.astype(float) / float(image.max()) * 255
    image = image.astype("uint8")
    image = Image.fromarray(image).convert("RGBA")

    scalefactor = spatial[sample_name]["scalefactors"]["tissue_hires_scalef"]

    if max(image.size) > 500:
        im_w, im_h = image.size
        im_w = float(im_w)
        im_h = float(im_h)
        sf = 500 / im_w

        resized_image = image.resize((int(im_w * sf), int(im_h * sf)), Image.ANTIALIAS)

        scalefactor *= sf
        scalefactor = 1 / scalefactor
    else:
        resized_image = image

    return resized_image, scalefactor


def read_data(
    path: str,
) -> Tuple[Optional[ad.AnnData], Image.Image, float]:

    image_ext = ["png", "gif", "jpg", "jpeg", "bmp"]

    if any([path.endswith(ext) for ext in image_ext]):
        image = Image.open(path).convert("RGBA")
        im_w, im_h = image.size
        im_w = float(im_w)
        im_h = float(im_h)
        scalefactor = 500 / im_w
        image = image.resize(
            (int(im_w * scalefactor), int(im_h * scalefactor)), Image.ANTIALIAS
        )
        scalefactor = 1 / scalefactor
        adata = None
    elif path.endswith("h5ad"):
        adata = ad.read_h5ad(path)
        image, scalefactor = get_image_from_anndata(adata)
    else:
        print("[ERROR] : Data format not supported. Exiting.")
        sys.exit(-1)

    return adata, image, scalefactor


class MainWindow(QMainWindow):
    def __init__(
        self,
        path: str,
    ):
        super().__init__()

        np.random.seed(1)

        self.path = path
        self.adata, self.im, self.scalefactor = read_data(self.path)

        self.im_w, self.im_h = self.im.size
        self.margin = 100
        self.box_space = 150

        self.h = self.im_h + self.margin * 2 + self.box_space
        self.w = self.im_w + self.margin * 2

        #    self.initUI()
        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        self.setFixedWidth(self.w)
        self.setFixedHeight(self.h)
        self.points = []
        self.colors = []

        self.im = QtGui.QPixmap.fromImage(ImageQt(self.im))

        self.UiComponents()

    def getColor(
        self,
    ):
        rgb = np.random.uniform(0, 255, size=3).astype(int).tolist()
        return QtGui.QColor(*rgb)

    def mousePressEvent(self, QMouseEvent):
        pos = QMouseEvent.pos()
        if (
            pos.y() <= self.im_h + self.margin
            and pos.y() >= self.margin / 2
            and pos.x() <= self.im_w + self.margin + self.margin / 2
            and pos.x() >= self.margin / self.margin
        ):
            self.points.append(pos)
            self.update()

    def paintEvent(self, ev):
        qp = QtGui.QPainter(self)
        rect = QtCore.QRect(self.margin, self.margin, self.im_w, self.im_h)
        pm = qp.drawPixmap(rect, self.im)

        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.setFont(QtGui.QFont("Calibri", 20))
        for k, pos in enumerate(self.points):
            if len(self.colors) > k:
                color = self.colors[k]
            else:
                color = self.getColor()
                self.colors.append(color)
            pen = QtGui.QPen(color, 5)
            brush = QtGui.QBrush(color)
            qp.setPen(pen)
            qp.setBrush(brush)
            qp.drawEllipse(pos, 5, 5)
            txt_x = pos.x() + 15
            txt_y = pos.y() + 15
            qp.drawText(txt_x, txt_y, str(k))

    def UiComponents(self):

        button_margin = 10
        undo_button_h = 30
        undo_button_w = 60
        undo_button = QPushButton("UNDO", self)
        undo_button.setGeometry(
            self.w / 2 - undo_button_w - button_margin / 2,
            self.im_h + self.margin * 2,
            undo_button_w,
            undo_button_h,
        )
        undo_button.clicked.connect(self.undopoints)
        save_button_h = 30
        save_button_w = 60
        save_button = QPushButton("SAVE", self)
        save_button.setGeometry(
            self.w / 2 + button_margin / 2,
            self.im_h + self.margin * 2,
            save_button_w,
            save_button_h,
        )

        save_button.clicked.connect(self.savepoints)

        if self.adata is None:

            tf_y = self.im_h + self.margin * 2 + undo_button_h + 5

            s_tf_h = 30
            s_tf_w = 200

            o_tf_h = 30
            o_tf_w = 200

            tf_margin = 10

            self.s_tf = QLineEdit("Sample Name", self)
            self.s_tf.setGeometry(
                self.w / 2 - tf_margin / 2 - s_tf_w,
                tf_y,
                s_tf_w,
                s_tf_h,
            )

            self.o_tf = QLineEdit("Outdir", self)
            self.o_tf.setGeometry(
                self.w / 2 + tf_margin / 2,
                tf_y,
                o_tf_w,
                o_tf_h,
            )

            self.o_tf.textChanged.connect(self._update_outdir)
            self.s_tf.textChanged.connect(self._update_sample)

    def _update_outdir(
        self,
    ):
        self.outdir = self.o_tf.text()

    def _update_sample(
        self,
    ):
        self.sample = self.s_tf.text()

    def undopoints(
        self,
    ):
        self.points.pop(-1)
        self.update()

    def savepoints(
        self,
    ):

        xs = [(p.x() - self.margin) * self.scalefactor for p in self.points]
        ys = [(p.y() - self.margin) * self.scalefactor for p in self.points]
        index = ["Landmark_{}".format(x) for x in range(len(self.points))]
        df = pd.DataFrame(
            dict(
                x_coord=xs,
                y_coord=ys,
            ),
            index=index,
        )

        screen = QtWidgets.QApplication.primaryScreen()
        screenshot = screen.grabWindow(
            window.winId(),
            x=self.margin,
            y=self.margin,
            width=self.im_w,
            height=self.im_h,
        )

        if self.adata is None:

            if not osp.isdir(self.outdir):
                os.mkdir(self.outdir)

            if self.outdir is None:
                self.outdir = os.getcwd()
            if self.sample is None:
                self.sample = "unknown_sample"

            basename = osp.join(self.outdir, self.sample + "_landmarks")

            df_name = basename + ".tsv"
            im_name = basename + ".png"

            df.to_csv(df_name, sep="\t")

            screenshot.save(im_name, "png")
        else:
            sample_name = get_adata_sample_name(self.adata)
            image = Image.fromqimage(screenshot)
            image = np.asarray(image)
            self.adata.uns["spatial"][sample_name]["images"]["hires_landmarks"] = image
            self.adata.uns["curated_landmarks"] = df.values
            self.adata.write_h5ad(self.path)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainWindow(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
