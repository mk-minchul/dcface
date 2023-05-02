import numpy as np
UINT8_MAX = 2. ** 8. - 1.
UINT16_MAX = 2. ** 16. - 1.
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageDraw, ImageFont
import cv2

def convert_image_type(image, dtype=np.float32):
    if image.dtype == np.uint8:

        if dtype == np.float32:
            image = image.astype(np.float32)
            image /= UINT8_MAX
            return image
        elif dtype == np.uint8:
            return image
        else:
            raise TypeError('numpy.float32 or numpy.uint8 supported as a target dtype')

    elif image.dtype == np.uint16:

        if dtype == np.float32:
            image = image.astype(np.float32)
            image /= UINT16_MAX
            return image
        elif dtype == np.uint8:
            image = image.astype(np.float32)
            image *= UINT8_MAX / UINT16_MAX
            image = image.astype(np.uint8)
            return image
        elif dtype == np.uint16:
            return image
        else:
            raise TypeError('numpy.float32 or numpy.uint8 or numpy.uint16 supported as a target dtype')

    elif image.dtype == np.float32:
        assert image.max() <= 1
        assert image.min() >= 0
        if dtype == np.float32:
            return image
        elif dtype == np.uint8:
            image *= UINT8_MAX
            image = image.astype(np.uint8)
            return image
        elif dtype == np.uint16:
            image *= UINT16_MAX
            image = image.astype(np.uint16)
            return image

    else:
        raise TypeError('numpy.uint8 or numpy.uint16 or np.float32 supported as an input dtype')


def stack_images(images, num_cols, num_rows, pershape=(112,112)):
    stack = []
    for rownum in range(num_rows):
        row = []
        for colnum in range(num_cols):
            idx = rownum * num_cols + colnum
            if idx > len(images)-1:
                img_resized = np.zeros((pershape[0], pershape[1], 3))
            else:
                if isinstance(images[idx], str):
                    img = cv2.imread(images[idx])
                    img_resized = cv2.resize(img, dsize=pershape)
                else:
                    img_resized = cv2.resize(images[idx], dsize=pershape)
            row.append(img_resized)
        row = np.concatenate(row, axis=1)
        stack.append(row)
    stack = np.concatenate(stack, axis=0)
    return stack

def prepare_text_img(text, height=300, width=30, fontsize=16, textcolor='C1', fontweight='normal', bg_color='white'):
    text_kwargs = dict(ha='center', va='center', fontsize=fontsize, color=textcolor, fontweight=fontweight)
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(width*px, height*px), facecolor=bg_color)
    plt.text(0.5, 0.5, text, **text_kwargs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_facecolor(bg_color)
    array = get_img_from_fig(fig)
    plt.clf()
    array = cv2.resize(array, (width, height))
    return array

def prepare_text_img_V2(text, height=300, width=30, fontsize=16, textcolor='black', bg_color='white'):
    wh = (width, height)
    txt = Image.new("RGB", wh, color=bg_color)
    draw = ImageDraw.Draw(txt)
    font = ImageFont.truetype('data/DejaVuSans.ttf', size=fontsize)
    text_w, text_h = draw.textsize(text, font=font)
    draw.text(((wh[0]-text_w)/2,(wh[1]-text_h)/2), text, fill=textcolor, font=font)
    return np.array(txt)[:,:,::-1]


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def tensor_to_numpy(tensor):
    arr = tensor.numpy().transpose(1,2,0)
    return (arr * 0.5 + 0.5) * 255

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def temp_plot(image, is_numpy=False, path='/mckim/temp/temp.png'):
    if not is_numpy:
        image = tensor_to_numpy(image.float().detach().cpu())
    cv2.imwrite(path, image)