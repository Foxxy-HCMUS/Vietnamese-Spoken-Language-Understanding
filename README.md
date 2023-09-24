# Vietnamese-Spoken-Language-Understanding
This repository is based on the SOICT 2023 contest's Spoken Language Understanding track. The goal of this track is to use audio for intent detection and slot tagging.

## Data
[https://drive.google.com/drive/folders/1FqCmmSjMMgkYjANXY7FD6tzqsfDwZJrY](https://drive.google.com/drive/folders/1FqCmmSjMMgkYjANXY7FD6tzqsfDwZJrY)

## Pre-trained Model
[https://huggingface.co/nguyenvulebinh/wav2vec2-large-vi](https://huggingface.co/nguyenvulebinh/wav2vec2-large-vi)

## Processed Data and Model
- Due to the limited resources for training, I use Kaggle's free accelerator. As a result, the processed data and additional models are located on the Kaggle platform.
  - [https://www.kaggle.com/code/noobhocai/soict2023-slu-preprocess-and-augmentation-stage-1](https://www.kaggle.com/code/noobhocai/soict2023-slu-preprocess-and-augmentation-stage-1)
  - [https://www.kaggle.com/code/noobhocai/soict2023-slu-wav2vec2-training-stage-1](https://www.kaggle.com/code/noobhocai/soict2023-slu-wav2vec2-training-stage-1)
  - [https://www.kaggle.com/code/noobhocai/soict-2023-wav2vec2-n-gram-inference-stage-1](https://www.kaggle.com/code/noobhocai/soict-2023-wav2vec2-n-gram-inference-stage-1)
  - [https://www.kaggle.com/code/huynhtuannam/soict2023-slu-training-stage-2-v2](https://www.kaggle.com/code/huynhtuannam/soict2023-slu-training-stage-2-v2)
  - [https://www.kaggle.com/code/noobhocai/soict2023-slu-inference-stage-2](https://www.kaggle.com/code/noobhocai/soict2023-slu-inference-stage-2)
- Finetuned `wav2vec2` model: [https://huggingface.co/foxxy-hm/wav2vec2-base-finetune-vi-v6](https://huggingface.co/foxxy-hm/wav2vec2-base-finetune-vi-v6)

## Solution
- First, I tried adding noise and adjusting the volume to expose the model to different types of data. This helped to improve predictions and make the model more versatile.

- Next, I fine-tuned the pre-trained `wav2vec2` model to get even better accuracy.

- I also created a language model using `ngrams` to correct any mistakes made by the automatic speech recognition system. The accuracy of the model was significantly improved by this.

- Finally, I used `A Bi-model based RNN Semantic Frame Parsing Model` for intent detection and slot filling. This model uses shared states to enhance task assistance and improve input interpretation accuracy.

## Conclusion
- As I mentioned earlier, my limited resources prevented the model from fully converging.
- In addition, maybe I'll use other models to correct grammatical errors for better accuracy.



  
