# webml

- neural network with wasm for image classification. 
- classify the image detection where you can define the boundary boxes. 

Conceptualization:
- RGB based single neural classification method i implemented in Burn
  for classification of the image coming from the diseased and non-diseased
  datasets.
- I thought of this as i have previously implemented in XgBoost and classification
  outperform the neural where we dont have the bounding box detection defined in YOLO, COCO.
- The idea is that we take the image which is clean and read the RGB coordinates
  and convert them into a tensor
- The same we collect the diseased ones as the tensor and then implement
  the single neural classification model on the same.
- Complete conceptualization to coding four hours offline.

Tomorrow
- Adding the web interface complete in Axum. 

```
cargo build
```

Gaurav Sablok \
codeprog@icloud.com
