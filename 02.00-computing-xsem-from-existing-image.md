##### **1. Title & Purpose**

```md

# XSEM from Image

This document explains how to extract XSEM (semantic encoding) from an image using the DiffAE model.

```


```md

## Extracting XSEM from an Image

  

The following code loads an image, processes it, and extracts the semantic encoding:

  

```python

from PIL import Image

import torch

from torchvision.transforms import functional as VF

from templates import ffhq256_autoenc, LitModel

  

device = 'cuda'

conf = ffhq256_autoenc()

model = LitModel(conf)


  

# Load Image

img = Image.open('example.jpg').resize((256, 256)).convert('RGB')

  

# Convert to Tensor

x = VF.to_tensor(img).unsqueeze(0).to(device)

  

# Encode

xsem = model.encode(x)

```

  

##### **4. Expected Output**

```md

## Expected Output

- The variable `xsem` now contains the extracted features of the image.

- These features can be used for image manipulation, reconstruction, or generation.

```

  

---

  


