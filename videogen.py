import cv2
import numpy as np
import time


def create_blank_image(width, height, color=(255, 255, 255)):
    """Creates a blank image with the specified color and size"""
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = tuple(reversed(color))

    return image


def plot_object(frame, pos, object_color, object_type, size):
    if object_type == "ball":
        cv2.circle(frame, pos, size, object_color, thickness=-1)
    elif object_type == "rect":
        if isinstance(size, tuple):
            w, h = size
        else:
            w = size
            h = size
        cv2.rectangle(
            frame,
            (int(pos[0] - w / 2), int(pos[0] + w / 2)),
            (int(pos[1] - h / 2), int(pos[1] + h / 2)),
        )
    return frame


def add_bakcround_noise(frame, type):
    if type == "gaussian":
        gauss = np.random.normal(0, 1, frame.size)
        gauss = gauss.reshape(frame.shape[0], frame.shape[1], frame.shape[2]).astype(
            "uint8"
        )
        return frame + frame * gauss
    elif type == "s&p":
        row, col, ch = frame.shape
        s_vs_p = options["svp"]
        amount = options["amount"]
        out = frame

        num_salt = np.ceil(amount * frame.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in frame.shape]
        out[coords] = 1

        num_pepper = np.ceil(amount * frame.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in frame.shape]
        out[coords] = 0
        return out
    # elif type == "poisson":
    #    vals = len(np.unique(frame))
    #    vals = 2 ** np.ceil(np.log2(vals))
    #    frame = np.random.poisson(frame * vals) / float(vals)
    #    return frame
    # elif type == "speckle":
    #    row, col, ch = frame.shape
    #    gauss = np.random.randn(row, col, ch)
    #    gauss = gauss.reshape(row, col, ch)
    #    frame = frame + frame * gauss
    #    return frame
    return frame


class VideoCreator:
    def __init__(self, width, height, x_func, y_func, step, end):
        self.width = width
        self.height = height
        self.x_func = x_func
        self.y_func = y_func
        self.t = np.linspace(0, end, int(end / step))

    def export(self, filename, options, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, fps, (self.width, self.height))
        for t in range(len(self.t)):
            frame = create_blank_image(
                self.width, self.height, options["background-color"]
            )
            frame = plot_object(
                frame,
                (self.x_func(t), self.y_func(t)),
                options["object-color"],
                options["object-type"],
                options["object-size"],
            )
            if "noise-type" in options:
                frame = add_bakcround_noise(frame, options["noise-type"])

            writer.write(frame)
            print(str(round(t / len(self.t) * 100)) + "/100%", end="\r")
        print("Successfully created video: " + filename, end="\r")


options = {
    "background-color": (255, 255, 255),  # tuple (required)
    "noise-type": "gaussian",  # string (optional)
    "object-type": "ball",  # string (required)
    "object-size": 10,  # int (required)
    "object-color": (255, 0, 0),  # Tuple (required)
    "svp": 0.5,  # Only for salt and pepper
    "amount": 0.05,  # only for salt and pepper
}


## DEFINE TRAJECTORY FUNCTIONS
def f_x(t):
    return int(1920 / 2 + 800 * np.sin(t / 1000 * 2 * np.pi))


def f_y(t):
    return int(1080 / 2 + 500 * np.sin(t / 1000 * 4 * np.pi))


## DEFINED TRAJECTORY FUNCTIONS
videogen = VideoCreator(1920, 1080, f_x, f_y, 1, 500)
videogen.export("output.mp4", options)

