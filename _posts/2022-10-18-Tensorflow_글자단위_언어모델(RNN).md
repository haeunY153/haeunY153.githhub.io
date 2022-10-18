---
layout: single
title: "[DL] Seq2Seq_RNN model"
categories: Text_Analysis
tags: [NLP, tensorflow, DL]
toc: True
author_profile: False
sidebar:
    nav: "docs"
---



## Seq2seq 시퀀스-투-시퀀스 모델링 작업
- 문자 수준의 텍스트 생성 RNN -LSTM 모델 이용
- '머신러닝 교과서 개정3판 p705' 참고

### 1. 텍스트 데이터셋 확보



```python
import numpy as np

# 데이터셋: https://www.gutenberg.org/files/11027/11027.txt (그림형제 요정 동화)
# 텍스트 읽고 전처리하기
with open("/content/drive/MyDrive/11027.txt", "r", encoding="utf-8") as f:
  text = f.read()

text[:400]
```




    "The Project Gutenberg eBook, Grimm's Fairy Stories, by Jacob Grimm and\nWilhelm Grimm, Illustrated by John B Gruelle and R. Emmett Owen\n\n\nThis eBook is for the use of anyone anywhere at no cost and with\nalmost no restrictions whatsoever.  You may copy it, give it away or\nre-use it under the terms of the Project Gutenberg License included\nwith this eBook or online at www.gutenberg.org\n\n\n\n\n\nTitle: Gr"




```python
start_idx = text.find('THE GOOSE-GIRL')
end_idx = text.find('END OF THE PROJECT GUTENBERG')
text = text[start_idx:end_idx]
char_set = set(text)
print('전체 길이:', len(text))
print('고유 문자:', len(char_set))
```

    전체 길이: 266073
    고유 문자: 70
    


```python
text[:14]
```




    'THE GOOSE-GIRL'




```python
# 정수 인코딩 & 넘파일 배열을 활용한 역매핑

chars_sorted = sorted(char_set)
char2int = {ch:i for i, ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)

text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32)
print('인코딩 된 텍스트 크기:', text_encoded.shape)
```

    인코딩 된 텍스트 크기: (266073,)
    


```python
print(text[:14], ' ====인코딩===> ', text_encoded[:14])
print(text_encoded[14:46], ' ===디코딩===> ', ''.join(char_array[text_encoded][14:46]))
```

    THE GOOSE-GIRL  ====인코딩===>  [34 22 19  1 21 29 29 33 19  9 21 23 32 26]
    [ 0  0 34 22 19  1 26 23 34 34 26 19  1 16 32 29 34 22 19 32  1 15 28 18
      1 33 23 33 34 19 32  0]  ===디코딩===>  
    
    THE LITTLE BROTHER AND SISTER
    
    

### 2. 텐서플로우로 데이터셋 만들기


```python
import tensorflow as tf

# 텍스트 순서대로 인코딩 저장
ds_text_encoded = tf.data.Dataset.from_tensor_slices(text_encoded)

for ex in ds_text_encoded.take(5):
  print('{} -> {}'.format(ex.numpy(), char_array[ex.numpy()]))
```

    34 -> T
    22 -> H
    19 -> E
    1 ->  
    21 -> G
    


```python
# bath() 이용하여 텍스트 조각 만들기

seq_length = 50 # 변경 가능
chunk_size = seq_length + 1
ds_chunks = ds_text_encoded.batch(chunk_size, drop_remainder=True)

# x, y를 나누기 위한 함수 정의

def split_input_target(chunk):
  input_seq = chunk[:-1]
  target_seq = chunk[1:]
  return input_seq, target_seq

ds_sequences = ds_chunks.map(split_input_target) # 모든 조각에 적용하기

# 데이터셋에서 샘플 확인

for example in ds_sequences.take(2):
  print('입력 x: ', repr(''.join(char_array[example[0].numpy()])))
  print('타깃 y: ', repr(''.join(char_array[example[1].numpy()])))
  print()

```

    입력 x:  'THE GOOSE-GIRL\n\nTHE LITTLE BROTHER AND SISTER\n\nHAN'
    타깃 y:  'HE GOOSE-GIRL\n\nTHE LITTLE BROTHER AND SISTER\n\nHANS'
    
    입력 x:  'EL AND GRETHEL\n\nOH, IF I COULD BUT SHIVER!\n\nDUMMLI'
    타깃 y:  'L AND GRETHEL\n\nOH, IF I COULD BUT SHIVER!\n\nDUMMLIN'
    
    


```python
# 미니 배치로 나누기 (여러 개의 훈련 샘플을 갖고 있음)

BATCH_SIZE = 64
BUFFER_SIZE = 10000
ds = ds_sequences.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

### 3. RNN 모델 만들기


```python
# 모델 정의 (함수 사용)

def build_model(vocab_size, embedding_dim, rnn_units):
  model= tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim),
      tf.keras.layers.LSTM(
          rnn_units,
          return_sequences = True),
          tf.keras.layers.Dense(vocab_size)])
  
  return model

# 매개변수 설정

charset_size = len(char_array)
embedding_dim= 256
rnn_units = 512

tf.random.set_seed(1)
model = build_model(
    vocab_size = charset_size,
    embedding_dim = embedding_dim,
    rnn_units = rnn_units
)

model.summary()

# LSTM 출력크기 (None, None, 256) 랭크 3 -> 이유: LSTM 만들때 return_sequences=True로 지정했기 때문
# 완전 연결층 (Dense)이 LSTM 출력을 받아 출력 시퀀스의 각 원소마다 로짓을 계산
# 순서대로 배치 차원, 출력 시퀀스 길이, 은닉 유닛 개수
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, None, 256)         17920     
                                                                     
     lstm (LSTM)                 (None, None, 512)         1574912   
                                                                     
     dense (Dense)               (None, None, 70)          35910     
                                                                     
    =================================================================
    Total params: 1,628,742
    Trainable params: 1,628,742
    Non-trainable params: 0
    _________________________________________________________________
    

### 모델 학습


```python
model.compile(
    optimizer='Adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )
)

# from_logits=True: 새로운 텍스트를 생성하기 위해 모델 예측값에서 샘플링 할 수 있도록 로짓 출력값이 필요
# activation=None

model.fit(ds, epochs=20)
```

    Epoch 1/20
    82/82 [==============================] - 8s 15ms/step - loss: 2.9678
    Epoch 2/20
    82/82 [==============================] - 1s 14ms/step - loss: 2.2996
    Epoch 3/20
    82/82 [==============================] - 1s 14ms/step - loss: 2.0773
    Epoch 4/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.9225
    Epoch 5/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.8030
    Epoch 6/20
    82/82 [==============================] - 1s 13ms/step - loss: 1.7069
    Epoch 7/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.6286
    Epoch 8/20
    82/82 [==============================] - 1s 13ms/step - loss: 1.5626
    Epoch 9/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.5076
    Epoch 10/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.4590
    Epoch 11/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.4158
    Epoch 12/20
    82/82 [==============================] - 1s 13ms/step - loss: 1.3789
    Epoch 13/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.3429
    Epoch 14/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.3118
    Epoch 15/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.2823
    Epoch 16/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.2536
    Epoch 17/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.2270
    Epoch 18/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.2000
    Epoch 19/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.1744
    Epoch 20/20
    82/82 [==============================] - 1s 14ms/step - loss: 1.1505
    




    <keras.callbacks.History at 0x7f94602a24d0>



### 평가: 새로운 텍스트 생성하기


```python
def sample(model, starting_str,
           len_generated_text=500,
           max_input_length=50,
           scale_factor=1.0):
  encoded_input = [char2int[s] for s in starting_str]
  encoded_input = tf.reshape(encoded_input, (1, -1))

  generated_str = starting_str

  model.reset_states

  for i in range(len_generated_text):
    logits = model(encoded_input)
    logits = tf.squeeze(logits, 0)

    scaled_logits = logits * scale_factor
    new_char_indx = tf.random.categorical(
        scaled_logits, num_samples=1)

    new_char_indx = tf.squeeze(new_char_indx)[-1].numpy()

    generated_str += str(char_array[new_char_indx])

    new_char_indx = tf.expand_dims([new_char_indx], 0)
    encoded_input = tf.concat([encoded_input, new_char_indx], axis=1)
    encdoed_input = encoded_input[:, -max_input_length:]

    
  return generated_str

tf.random.set_seed(1)
print(sample(model, starting_str='Grimm Brothers'))

```

    Grimm Brothers behild, and then came
    towards; for
    asked her boud
    and one sat discovered to open, and kepping toward a paugh stranger pior, and bofred at him
    round them, sent a bit blow. I shat her sut off this younger man understuning that it grinding before,
    they went all to forest! I will not get deward" saking of a fine, but he took her and feathers, but they called out works into the lazeest meant she had mother,
    when they deppinayed nothing in the old of hunger, but only heard her,
    another
    our cheeses up
    

**샘플 예측 가능성 (Randomness) 조절**
- 생성된 텍스트가 훈련 테스트에서 학습한 패턴을 따르게 할지,
- 랜덤하게 생성할지 조절하기 위해 RNN이 계산한 로짓을 tf.random.categorical()
 - 샘플링 함수로 전달 하기 전 *ɑ* 알파로 스케일 조정가능
 - 온도의 역수ɑ: 높을수록 무작위성 커지고, 낮을 수록 예측가능


```python
# 예시, ɑ < 1로 로짓 스케일을 조정하면 softmax 함수가 계산할 확률은
# 다음과 코드처럼 더 균일해짐

logits = np.array([1.0, 1.0, 3.0])
print('스케일 조정 전의 확률: ', tf.math.softmax(logits).numpy())
print('0.5배 조정 후 확률: ', tf.math.softmax(0.5*logits).numpy())
print('0.1배 조정 후 확률: ', tf.math.softmax(0.1*logits).numpy())

```

    스케일 조정 전의 확률:  [0.10650698 0.10650698 0.78698604]
    0.5배 조정 후 확률:  [0.21194156 0.21194156 0.57611688]
    0.1배 조정 후 확률:  [0.31042377 0.31042377 0.37915245]
    

- ɑ=0.1로 스케일을 조정하면 거의 균등한 확률을 얻음
- 균등한 분포일수록 더 랜덤하게 샘플링


```python
# 텍스트 생성에 적용

tf.random.set_seed(1)
print(sample(model, starting_str='THE GOOSE-GIRL',
             scale_factor=2.0))

```

    THE GOOSE-GIRL
    
    
    Once upon a time conding to go to the beautiful clothes, and it heard they were not be into the house. She stood a fine disappeared, and the bridd bring and went to the faithful John, and they went out of the castle, and was a great feather. The King all was from the heak of the words and heard themself began to the true trees and went to the forest to cellect for them. The terthing took their dear bride, and cried out, "Who you will learn have no one mount to cold her, and asked them as so m
    


```python
tf.random.set_seed(1)
print(sample(model, starting_str='THE GOOSE-GIRL',
             scale_factor=0.5))

# ɑ=0.5로 온도를 높이면 더 랜덤한 텍스트 생성되는 것을 확인가능
```

    THE GOOSE-GIRLT
    
    OAlo: Tase, Snoccencurmah
    Tumf axarness,
    you yausE"; but Hankit?"
    Then then, stxnuck suct*man wihe after
    youghad you; orewhort
    of: Keptinrabr, wed iptabse!" "Mying Wom."
     Then her lutgesmening.
    "Younrymanlubred, upon a: lyiZ hargly
    Rashee,".
    
    So Once more t off.
    Sow! We knew I knepld"; finh!
      THee; arful Firg ame.
    It squck hexe; I wil, boAl
    Cared dim nom ye;
    bebir,
    ant ilow!e's mace, she haismyt,
    "Caw!
    take Pell!,")ay, and tyen is qulenir own, Ofewht." HorWhe's dwilld,
    Oho! Ropou."
    
    Thenself
    
