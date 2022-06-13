import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



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
            (int(pos[0] - w / 2), int(pos[1] - h / 2)),
            (int(pos[0] + w / 2), int(pos[1] + h / 2)),
            object_color,
            thickness=-1
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
    def __init__(self, width, height, x_func, dx_func, ddx_func, y_func, dy_func, ddy_func, step, end):
        self.width = width
        self.height = height

        self.x_func = x_func
        self.dx_func=dx_func
        self.ddx_func=ddx_func

        self.y_func = y_func
        self.dy_func=dy_func
        self.ddy_func=ddy_func

        self.t = np.linspace(0, end, int(end / step))

    def export(self, filename, options, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename+".mp4", fourcc, fps, (self.width, self.height))
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
        x=self.x_func(self.t)
        dx=self.dx_func(self.t)
        ddx=self.ddx_func(self.t)

        y=self.y_func(self.t)
        dy=self.dy_func(self.t)
        ddy=self.ddy_func(self.t)
        time=self.t/fps
        data=np.zeros((len(time),7))
        data[:,0]=time
        
        data[:,1]=x
        data[:,2]=dx
        data[:,3]=ddx
        
        data[:,4]=y
        data[:,5]=dy
        data[:,6]=ddy

        df=pd.DataFrame(data=data, columns=["time [s]", "x [pix]", "dx [pix/s]", "ddx [pix/^2]", "y [pix]", "dy [pix/s]", "ddy [pix/^2]"])
        df.to_csv(filename+"_data.csv")

        plt.plot(x,y)
        plt.show()


options = {
    "background-color": (255, 255, 255),  # tuple (required)
     #"noise-type": "none",  # string (optional)
    "object-type": "rect",  # string (required)
    "object-size": 40,  # int (required)
    "object-color": (255, 0, 0),  # Tuple (required)
    "svp": 0.5,  # Only for salt and pepper
    "amount": 0.05,  # only for salt and pepper
}


## DEFINE TRAJECTORY FUNCTIONS
def f_x(t):
    return np.floor(3840 / 2 + 1800 * np.sin(t / 1200 * 2 * np.pi))

def df_x(t):
    return 3*np.pi*np.cos(t*np.pi/600)

def ddf_x(t):
    return -1/200*(np.pi**2)*np.sin(t*np.pi/600)

def f_y(t):
    return np.floor(2160 / 2 + 1000 * np.sin(t / 1200 * 12 * np.pi))

def df_y(t):
    return 10*np.pi*np.cos(t*np.pi/100)

def ddf_y(t):
    return -1/10*(np.pi**2)*np.sin(t*np.pi/100)


## DEFINED TRAJECTORY FUNCTIONS
videogen = VideoCreator(3840, 2160, f_x, df_y, ddf_x, f_y, df_y, ddf_y, 1, 1200)
videogen.export("sample_video_4k", options, fps=60)


