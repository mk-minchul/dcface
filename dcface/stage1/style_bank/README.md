
# Style Bank

Style Images have to be aligned like the ID images. We provide few aligned images in
```
dcface/stage1/style_bank/style_images/man
dcface/stage1/style_bank/style_images/woman
```
We differentiate the genders as the synthesis works better with same gender between ID and Style images.

## Custom Style Images

We provide the code to align the images.

1. Place images with a face in a directory of your choice. `ex: dcface/stage1/style_bank/style_images/raw`
2. Run the following command

```
cd dcface/stage1/align_faces/
python main.py --root ../style_bank/style_images/raw \
               --save_root ../style_bank/style_images/raw_aligned
```

This will save the aligned images in `dcface/stage1/style_bank/style_images/raw_aligned`