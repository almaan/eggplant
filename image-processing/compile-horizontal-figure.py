import string
import os.path as osp
import argparse as arp
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def main(args):
    font = ImageFont.truetype("Calibri.ttf", args.font_size)

    if args.padding is None:
        PADDING = dict(
            left=80,
            right=20,
            top=20,
            bottom=20,
        )
    elif len(args.padding) == 2:
        PADDING = dict(
            left=args.padding[0],
            right=args.padding[0],
            top=args.padding[1],
            bottom=args.padding[1],
        )
    elif len(args.padding) == 4:
        PADDING = dict(
            left=args.padding[0],
            right=args.padding[2],
            top=args.padding[1],
            bottom=args.padding[3],
        )
    else:
        raise ValueError("Wrong size of padding")

    if args.text_margin is None:
        TEXT_MARGIN = dict(top=15, left=10)
    elif len(args.text_margin) == 1:
        TEXT_MARGIN = dict(top=args.text_margin[0], left=args.text_margin[0])
    elif len(args.text_margin) == 2:
        TEXT_MARGIN = dict(top=args.text_margin[0], left=args.text_margin[1])
    else:
        raise ValueError("Wrong size of text margin")

    BORDERSIZE = args.border_size

    # DEFAULTS
    n_subplots = len(args.sub_figures)
    letters = string.ascii_uppercase[:n_subplots]
    print(letters)
    sub_figs_pths = {ll: p for ll, p in zip(letters, args.sub_figures)}

    std_borders = (BORDERSIZE, BORDERSIZE, BORDERSIZE, 0)
    bot_borders = (BORDERSIZE, BORDERSIZE, BORDERSIZE, BORDERSIZE)
    sub_border_size = {k: std_borders for k in sub_figs_pths.keys()}
    last_key = letters[-1]
    sub_border_size[last_key] = bot_borders

    W_FULL = args.full_width
    W_ADJ = W_FULL - PADDING["left"] - PADDING["right"] - 2 * BORDERSIZE

    sub_figs = {k: Image.open(v).convert("RGBA") for k, v in sub_figs_pths.items()}
    imgs = dict()
    total_h = 0

    for sub, img in sub_figs.items():
        w, h = img.size
        if w > W_ADJ:
            sf = float(W_ADJ) / float(w)
            w = int(round(w * sf))
            h = int(round(h * sf))
            img = img.resize((w, h), Image.ANTIALIAS)

        bg_h = h + PADDING["top"] + PADDING["bottom"]
        bg = Image.new("RGB", (W_FULL - 2 * BORDERSIZE, bg_h), color="white")

        box = (
            int(W_ADJ / 2 - w / 2 + PADDING["left"]),
            int(h / 2 - h / 2 + PADDING["top"]),
        )
        bg.paste(img, box, img)
        img = bg
        draw = ImageDraw.Draw(img)
        draw.text(
            (TEXT_MARGIN["left"], TEXT_MARGIN["top"]), sub.upper(), (0, 0, 0), font=font
        )

        border_w = img.size[0] + sub_border_size[sub][0] + sub_border_size[sub][2]
        border_h = img.size[1] + sub_border_size[sub][1] + sub_border_size[sub][3]
        bg = Image.new("RGB", (border_w, border_h), color="black")

        bg.paste(img, (sub_border_size[sub][0], sub_border_size[sub][1]))
        del img
        total_h += bg.size[1]
        imgs[sub] = bg

    full_img = Image.new("RGB", (W_FULL, total_h))

    box_h = 0
    for key in letters:
        _, img_h = imgs[key].size
        full_img.paste(imgs[key], (0, box_h))
        box_h += img_h

    if args.out_dir is None:
        from os import getcwd

        args.out_dir = getcwd()

    full_img.save(
        osp.join(args.out_dir, args.figure_name + ".png"),
    )


if __name__ == "__main__":
    prs = arp.ArgumentParser()
    aa = prs.add_argument

    aa(
        "--sub_figures",
        "-sf",
        nargs="+",
        help="paths to subfigures. Needs to be in sequential order",
        required=True,
    )

    aa(
        "--padding",
        "-pd",
        type=int,
        nargs="+",
        default=None,
    )
    aa(
        "--text_margin",
        "-tm",
        type=float,
        nargs="+",
    )
    aa(
        "--full_width",
        type=int,
        default=1000,
    )

    aa(
        "--border_size",
        "-b",
        type=float,
        default=4,
    )
    aa(
        "-out_dir",
        "-od",
        help="output directory",
        default=None,
    )

    aa(
        "--figure_name",
        "-fg",
        help="name of figure",
        required=True,
    )

    aa("--font_size", "-fs,", type=float, default=60, help="font size of panel labels")

    args = prs.parse_args()

    main(args)
