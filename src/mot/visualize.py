import gif
import matplotlib.pyplot as plt
from cycler import cycler as cy
from tqdm.auto import tqdm
from collections import defaultdict


colors = [
    "aliceblue",
    "antiquewhite",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanchedalmond",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgray",
    "darkgreen",
    "darkgrey",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "darkslategrey",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dimgray",
    "dimgrey",
    "dodgerblue",
    "firebrick",
    "floralwhite",
    "forestgreen",
    "fuchsia",
    "gainsboro",
    "ghostwhite",
    "gold",
    "goldenrod",
    "gray",
    "green",
    "greenyellow",
    "grey",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgreen",
    "lightgrey",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "lightskyblue",
    "lightslategray",
    "lightslategrey",
    "lightsteelblue",
    "lightyellow",
    "lime",
    "limegreen",
    "linen",
    "magenta",
    "maroon",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "navy",
    "oldlace",
    "olive",
    "olivedrab",
    "orange",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "purple",
    "rebeccapurple",
    "red",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "silver",
    "skyblue",
    "slateblue",
    "slategray",
    "slategrey",
    "snow",
    "springgreen",
    "steelblue",
    "tan",
    "teal",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "whitesmoke",
    "yellow",
    "yellowgreen",
]


def plot_sequence(tracks, dataset, first_n_frames=None, dst_path=None):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
        first_n_frames (int): eval only first N frames from all sequence
    """
    # infinite color loop
    cyl = cy("ec", colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))  # skipcq: PTC-W0063

    for i, v in enumerate(dataset):
        img = v["img"].mul(255).permute(1, 2, 0).byte().numpy()
        width, height, _ = img.shape

        dpi = 96
        fig, ax = plt.subplots(1, dpi=dpi)
        fig.set_size_inches(width / dpi, height / dpi)
        ax.set_axis_off()
        ax.imshow(img)

        for j, t in tracks.items():
            if i in t.keys():
                t_i = t[i]
                ax.add_patch(
                    plt.Rectangle(
                        (t_i[0], t_i[1]),
                        t_i[2] - t_i[0],
                        t_i[3] - t_i[1],
                        fill=False,
                        linewidth=1.0,
                        **styles[j],
                    )
                )

                ax.annotate(
                    j,
                    (
                        t_i[0] + (t_i[2] - t_i[0]) / 2.0,
                        t_i[1] + (t_i[3] - t_i[1]) / 2.0,
                    ),
                    color=styles[j]["ec"],
                    weight="bold",
                    fontsize=6,
                    ha="center",
                    va="center",
                )

        plt.axis("off")
        plt.show()
        if dst_path:
            plt.savefig(dst_path)

        if first_n_frames is not None and first_n_frames - 1 == i:
            break


@gif.frame
def plot_single_tracked_frame(img, img_idx, tracks):
    cyl = cy("ec", colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))  # skipcq: PTC-W0063

    dpi = 300
    _, ax = plt.subplots(1, dpi=dpi)
    ax.set_axis_off()
    ax.imshow(img)

    for j, t in tracks.items():
        if img_idx in t.keys():
            t_i = t[img_idx]
            ax.add_patch(
                plt.Rectangle(
                    (t_i[0], t_i[1]),
                    t_i[2] - t_i[0],
                    t_i[3] - t_i[1],
                    fill=False,
                    linewidth=1.0,
                    **styles[j],
                )
            )

            ax.annotate(
                j,
                (
                    t_i[0] + (t_i[2] - t_i[0]) / 2.0,
                    t_i[1] + (t_i[3] - t_i[1]) / 2.0,
                ),
                color=styles[j]["ec"],
                weight="bold",
                fontsize=6,
                ha="center",
                va="center",
            )


def collect_frames_for_gif(sequence, tracker_seq_res, first_n_frames=None):
    frames = []
    total_length = first_n_frames if first_n_frames is not None else len(sequence)
    for img_idx, data in tqdm(enumerate(sequence), total=total_length):
        img = data["img"].mul(255).permute(1, 2, 0).byte().numpy()

        frame = plot_single_tracked_frame(img, img_idx, tracker_seq_res)
        frames.append(frame)
        if first_n_frames is not None and first_n_frames - 1 == img_idx:
            break
    return frames
