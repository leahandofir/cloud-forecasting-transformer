## Inference

The full argument list for *ims_inference.py* is as follows: 

**`--ckpt (required)`**: the checkpoint of the model we want to inference with. 

**`--data-dir (required)`**: the path where the images are at (in the format %Y%m%d%H%M). the image files has to be in PNG format.

**`--start-time (required)`**: the time of the first frame in the input (in the format %Y%m%d%H%M).

**`--output-dir`** (default="./"): the path where the inference will be saved at.

**`--img-format`** (default="png"): the format of the input images.

**`--fs`**: the font size in the visualization of the output.

**`--figsize`**: the figure size of the visualization of the output.

**`--plot-stride`**: the plot stride in the visualization of the output (meaning 'how many frames we skip when displaying the output'

**`--cmap`**: a pyplot-supported color map for the visualization of the output.

**`--left`**: set where to start cropping the image from the left. If not set, taken from checkpoint.

**`--top`**: set where to start cropping the image from the top. If not set, taken from checkpoint.

**`--width`**: set the width of the cropped image. If not set, taken from checkpoint.

**`--height`**: set the height of the cropped image. If not set, taken from checkpoint.