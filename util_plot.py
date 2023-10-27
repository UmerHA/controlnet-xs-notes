from PIL import Image, ImageOps, ImageDraw
from tqdm.notebook import tqdm
from functools import partial

def save_latents(i,t,lat,lats):
    if i==0 and len(lat)>0: lats = []
    lats.append((i,t,lat))

def lat2img(lat, resize_to=None, output_type='pil'):
    with torch.no_grad():
        ims = cnxs_pipe.vae.decode(lat / cnxs_pipe.vae.config.scaling_factor, return_dict=False)[0]
        ims = cnxs_pipe.image_processor.postprocess(ims, output_type=output_type)
        if resize_to is not None:
            if output_type=='pil': ims = [im.resize(resize_to) for im in ims]
            else: print(f'Not resizing as output_type = {output_type} requested')
    return ims

def only_lat(o): return o[-1] if isinstance(o,tuple) else o
def lats2imgs(lats, resize_to=None, output_type='pil',pbar=True):
    if pbar: lats = tqdm(lats)
    ims = [lat2img(only_lat(lat), resize_to, output_type) for lat in lats]
    if output_type=='pt': ims = [im.cpu() for im in ims]
    return ims

real_idx = None
def plot_latents_to_pil_grid(lats, every=5, cols=7, im_size=(300, 300), pbar=True, border=2, return_ims=True, output_type='pil'):
    global real_idx
    
    real_idx = partial(lambda o,every,total: min(total-1,every*o), every=every, total=len(lats))
    
    titles = [f'Image {i}' for i, _, _ in lats if i % every == 0 or i == len(lats)-1]
    lats = [lat for i, _, lat in lats if i % every == 0 or i == len(lats)-1]
    if pbar: lats = tqdm(lats)
    ims = [lat2img(lat, resize_to=im_size, output_type=output_type)[0] for lat in lats]
    ims_bordered = [ImageOps.expand(im, border=2, fill='black') for im in ims]
    im_size = (im_size[0]+border, im_size[1]+border)

    rows = len(ims) // cols
    if rows * cols < len(ims): rows += 1

    grid_image = Image.new('RGB', (cols * im_size[0], rows * im_size[1]), color='grey')
    # draw diagonal white lines
    draw = ImageDraw.Draw(grid_image)
    for xy in range(0,2*max(cols * im_size[0], rows * im_size[1])+1,100):
        draw.line([(xy, 0), (0, xy)], fill="white", width=1)
    
    for i, img in enumerate(ims_bordered):
        x_offset = (i % cols) * im_size[0]
        y_offset = (i // cols) * im_size[1]
        grid_image.paste(img, (x_offset, y_offset))

    if return_ims: return grid_image, ims
    else: return grid_image