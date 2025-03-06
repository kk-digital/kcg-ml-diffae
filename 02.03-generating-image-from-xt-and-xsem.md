##### **1. Title & Purpose**

```md

# xt-and-xsem from Image

This document explains how to generate an image from `xt` and `xsem`.

```


```md

##  generate an image from `xt` and `xsem`

  


  

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

  


xsem = model.encode(x)
xt = model.encode_stochastic(x, cond, T=250)
xt_and_xsem = model.render(xt, xsem, T=20)

```

  

##### **4. Expected Output**

```md

## Expected Output

- xsem provides global structure, while xt refines details.


```

  

---

  

