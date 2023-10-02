# Facial landmark detection

1. Clonare il progetto con il comando:
  ```bash
  git clone https://github.com/christopherburatti/CV-DeepLearning
  ```


2. Scaricare il file <a href="https://drive.google.com/file/d/109q61tkFa5X5lh_pjx3VPZVrJwR7C99z/view?usp=sharing">model_best_FINAL_3.pth</a> e inserirlo nella cartella di default di progetto.

3. Installare le librerie di progetto con il comando:
  ```bash
  pip install -r requirements.txt
  ```

4. Per testare il programma con la demo, lanciare il comando
  ```bash
  python tools/demo.py --cfg hrnetv2_w18_imagenet_pretrained.pth --model model_best_FINAL_3.pth
  ```

## Nota
Per il corretto funzionamento del codice è necessario inserire una cartella "data", contenente una sotto-cartella "images", contenente tutte le immagini del dataset, e i file .csv, ognuno relativo ai valori di ground truth dei landmark facciali delle immagini del dataset utilizzate rispettivamente in fase di training, validation e testing.
