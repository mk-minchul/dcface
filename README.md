

# DCFace: Synthetic Face Generation with Dual Condition Diffusion Model
Official repository for the paper
DCFace: Synthetic Face Generation with Dual Condition Diffusion Model (CVPR 2023).

- Main paper: [main.pdf](assets/main.pdf)
- Supplementary: [supp.pdf](assets/supp.pdf)

### Pipepline
![Demo](assets/pipeline2.gif)
### ID Consistent Generation
![Demo](assets/pipeline.gif)


### Installation

```
pip install -r requirements.txt
```

Training Code, and Dataset are coming soon. This is the image synthesis code.

### Image generation
- Download the pretrained weights from the [link](https://drive.google.com/drive/folders/1ePqFN2eDo0l31aQOkW_U5Mcrlcl83j19?usp=share_link)
- Place the `pretrained_models` directory under `dcface` (same level as `src`)
- Run
```
cd dcface/src
python synthesis.py --id_images_root sample_images/id_images/sample_57.png --style_images_root sample_images/style_images/woman
```