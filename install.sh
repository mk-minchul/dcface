
pip install -r requirements.txt
mkdir -p dcface/pretrained_models
cd dcface/pretrained_models
wget -O adaface_ir50_casia.ckpt https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1AmaMTvfHq25obqb2i7MsbftJdAuvbN19
wget -O adaface_ir50_webface4m.ckpt https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1HdW-F1GxJv0MVBUIVpE6HAZ3S9SLsytL
wget -O center_ir_50_adaface_casia_faces_webface_112x112.pth https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1JsxekmFk-81JL9uqGR9ZUUqepo1QB53G
wget -O center_ir_101_adaface_webface4m_faces_webface_112x112.pth https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1a6eGAl5B2hbYLdyNHD8mFhGU0YRC_bd6
wget -O dcface_3x3.ckpt https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1yGniRYWz4Evr66LRgDxjwvN4gmnK5Ohi
wget -O dcface_5x5.ckpt https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1GJktCfKgO7prNPdeKlIXQ2CrmBddFp5Z
wget -O ffhq_10m_ft.pt https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1rSxpTkzO_pj_HsIfYmA6pkjXbZvHoKdm
wget -O ffhq_10m.pt https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1rlpA4uB5GLQfUTpDVJSv22VyvHlt22xk

