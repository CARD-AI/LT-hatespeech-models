# Exploring Hate Speech Detection Models for Lithuanian Language (Accepted at ACL WOAH 2025)

Authors: Justina Mandravickaitė, Eglė Rimkienė, Mindaugas Petkevičius, Milita Songailaitė, Eimantas Zaranka, Tomas Krilavičius

Accepted and presented and the Workshop on Online Abuse and Harms (WOAH) at ACL 2025.

## Abstract

Online hate speech poses a significant challenge, as it can incite violence and contribute to social polarization. This study evaluates traditional machine learning, deep learning and large language models (LLMs) for Lithuanian hate speech detection, addressing class imbalance issue via data augmentation and resampling techniques.
Our dataset included 27,358 user-generated comments, annotated into Neutral language (56%), Offensive language (29%) and Hate speech (15%). We trained BiLSTM, LSTM, CNN, SVM, and Random Forest models and fine-tuned Multilingual BERT, LitLat BERT, Electra, RWKV, ChatGPT, LT-Llama-2, and Gemma-2 models. Additionally, we pre-trained Electra for Lithuanian. Models were evaluated using accuracy and weighted F1-score.
On the imbalanced dataset, LitLat BERT (0.76 accuracy) and Electra (0.73 accuracy) performed best. Over-sampling further boosted accuracy, with Multilingual BERT (0.85) and LitLat BERT (0.84) outperforming other models. Over-sampling combined with augmentation provided the best overall results.
Under-sampling led to performance declines and was less effective.
Finally, fine-tuning LLMs significantly improved their accuracy which highlighted the importance of fine-tuning for more specialized NLP tasks.

**Please cite our paper in any published work that uses our results**

```bibtex
@inproceedings{mandravickaite2025lthatespeech,
  title={Exploring Hate Speech Detection Models for Lithuanian Language},
  author={Mandravickaite, Justina and Rimkiene, Egle and Petkevicius, Mindaugas and Songailaite, Milita and Zaranka, Eimantas and Krilavicius, Tomas},
  booktitle={Proceedings ...},
  volume={},
  number={},
  pages={},
  year={2025}
}
```

## Folders description

```

./src                 --> Contains the scripts for all hates speec detection modes.
./models              --> This folder will be created automatically, model checkpoints will be saved there.
./utils     	      --> Additional scripts for data preparation.	

```

## Running instructions

Install the required libraries using the command:
```
pip install -r requirements
```

To run the models, follow the instructions provided in each model's folder. The training data for these models is currently unavailable, as it has not yet been anonymized. If you'd like access to the already anonymized portion of the dataset, feel free to reach out — we’ll be happy to share it with you. Once the full dataset has been anonymized, it will be made publicly available on Hugging Face (a link will be added to this repository).

## TODO
- [ ] Add citation information
- [ ] Add instructions for running Llama, RWKV and other deep learning models
- [ ] Add ChatGPT testing scripts
- [ ] Add a link to the dataset once it's released
- [ ] Add a link to the published paper
- [ ] Upload our models to [transformers community](https://huggingface.co/models) to make them public
