---
layout: single
title: "2022.09_네이버 영화 리뷰 감성분석"
categories: Text_Analysis
tags: [Wordcloud, Text_Mining, python]
toc: True
author_profile: False
sidebar:
    nav: "docs"

# 검색 안 되게 하려면 search: false로 하면 됨
---

<h5>영화 리뷰 scrapping & 시각화, 긍부정 점수로 트렌드 파악하기</h5>

```python
# matplotlib font 깨짐 현상 방지, 한글 폰트 설정
from matplotlib import font_manager

!apt install fonts-nanum

for font in font_manager.fontManager.ttflist:
    if 'Nanum' in font.name:
        print(font.name, font.fname)

import matplotlib.font_manager as fm

# 설치된 폰트 출력
font_list = [font.name for font in fm.fontManager.ttflist]
font_list

plt.rcParams['font.family'] = 'NanumGothic'
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    fonts-nanum is already the newest version (20170925-1).
    The following package was automatically installed and is no longer required:
      libnvidia-common-460
    Use 'apt autoremove' to remove it.
    0 upgraded, 0 newly installed, 0 to remove and 20 not upgraded.
    

### 1.  자연어 처리 한국어 자연어처리 패키지 konlpy install하기


```python
!pip install konlpy
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: konlpy in /usr/local/lib/python3.7/dist-packages (0.6.0)
    Requirement already satisfied: numpy>=1.6 in /usr/local/lib/python3.7/dist-packages (from konlpy) (1.21.6)
    Requirement already satisfied: JPype1>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from konlpy) (1.4.0)
    Requirement already satisfied: lxml>=4.1.0 in /usr/local/lib/python3.7/dist-packages (from konlpy) (4.9.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from JPype1>=0.7.0->konlpy) (4.1.1)
    

### 2. 과제
- 1) 네이버 영화 리뷰 페이지에서 영화 리뷰 수집
- 2) 영화 이름 조회 시 리뷰 긍정 부정 점수, 평균 점수 출력
- 3) 영화 별 워드클라우드 만들기

#### 2-1) 네이버 영화 리뷰 페이지에서 영화 리뷰 수집


```python
# 1) 네이버 영화 리뷰 수집

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from konlpy.tag import Okt # 한국어 자연어처리 태그 모듈
import matplotlib.pyplot as plt
from multiprocessing import Pool # 멀티프로세싱 작업풀
import multiprocessing, datetime

def collect_review(page_num):
  url = f"https://movie.naver.com/movie/point/af/list.naver?&page={page_num}"
  df = pd.DataFrame(columns=["No", "Title", "Score", "Review"])
  table = pd.read_html(url)

  for i in range(10): # 1 페이지 당 리뷰 라인 10줄
    try:
      name = table[0]["감상평"][i].split("  별점 - 총 10점 중")[0]
      data = table[0]["감상평"][i].split("  별점 - 총 10점 중")[1][:-3].split("  ")
      index = table[0]["번호"][i]
      if len(data) == 2: # 점수, 감상평 둘 다 남긴 후기일 때
        df = df.append({"No":int(index), "Title":str(name), "Score": int(data[0]), "Review": str(data[1])}, ignore_index = True) # 기존 인덱스 무시 새로 세팅

    except:
      pass

  return df

print("몇 페이지까지 스크래핑 하실건가요?")


num_input = int(input())
df_list = []
for i in range(num_input):
  df_list.append(collect_review(i))

pool = Pool(processes = 10)
result = pool.map(collect_review, range(0, 100)) # 여기서 map함수 역할 (apply 함수와 같음) => map(함수, 적용범위)
csv_name = f"{datetime.datetime}.csv" # 저장 시간을 이름으로 csv 저장
pd.concat(result, axis = 0).to_csv(csv_name) # pandas concat함수: append rows of DataFrames

```

    몇 페이지까지 스크래핑 하실건가요?
    20
    


```python
# csv 맞게 저장 되었는지 확인한 뒤 파일 명 확인해서 불러오기 -- > 저장 안해도 무방
df = pd.read_csv("/content/2022-09-10 08:07:29.928385.csv")
```


```python
df
```





  <div id="df-0861b114-4b45-4215-8a92-ba7dd2797136">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>No</th>
      <th>Title</th>
      <th>Score</th>
      <th>Review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>18431105</td>
      <td>헌트</td>
      <td>8</td>
      <td>정우성 죽기전 마지막 대사랑 영화 마지막 총소리 이해가 안됨. 평점은 8점 재밌엇음</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>18431104</td>
      <td>공조2: 인터내셔날</td>
      <td>10</td>
      <td>외계인 비상선언보다는 훨씬 재밌었음</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>18431103</td>
      <td>공조2: 인터내셔날</td>
      <td>10</td>
      <td>지루할틈도 없었고 범죄도시2 이후에 간만에 또 잼나게봄. 개인적으로 1보다 2가 더...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>18431102</td>
      <td>육사오(6/45)</td>
      <td>10</td>
      <td>아 진짜 처음부터 끝까지 그냥 계속 웃음ㅋㅋㅋ</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>18431101</td>
      <td>매트릭스</td>
      <td>9</td>
      <td>감독의 상상력과 그것을 영화로 탁월하게 표현함이 감탄입니다. 디지털과 기계화 시대 ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>930</th>
      <td>5</td>
      <td>18430089</td>
      <td>육사오(6/45)</td>
      <td>10</td>
      <td>볼만 했음 ㅎㅎㅎㅎ가볍게 웃으며 볼 수 있음</td>
    </tr>
    <tr>
      <th>931</th>
      <td>6</td>
      <td>18430088</td>
      <td>공조2: 인터내셔날</td>
      <td>10</td>
      <td>평점이 왜 이렇게 낮아요? 기대하고 보면 별로겠지만, 기대없이 보면 요즘 한국 영화...</td>
    </tr>
    <tr>
      <th>932</th>
      <td>7</td>
      <td>18430087</td>
      <td>육사오(6/45)</td>
      <td>10</td>
      <td>배우들 연기도 잘하고 정말 개웃기다 ㅋㅋ</td>
    </tr>
    <tr>
      <th>933</th>
      <td>8</td>
      <td>18430086</td>
      <td>육사오(6/45)</td>
      <td>10</td>
      <td>정말 재미있어요 정말 강추 합니다 정말 추천 합니다 이영화 안보면 정말 후회 합니다...</td>
    </tr>
    <tr>
      <th>934</th>
      <td>9</td>
      <td>18430085</td>
      <td>비상선언</td>
      <td>10</td>
      <td>무능했던 정부에 화났던 영화</td>
    </tr>
  </tbody>
</table>
<p>935 rows × 5 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0861b114-4b45-4215-8a92-ba7dd2797136')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0861b114-4b45-4215-8a92-ba7dd2797136 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0861b114-4b45-4215-8a92-ba7dd2797136');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# 데이터프레임에서 필요 없는 Column 삭제

df.drop("Unnamed: 0", axis = 1, inplace=True)
df
```





  <div id="df-e7703103-9697-45cd-b13c-2ae26d6192d7">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No</th>
      <th>Title</th>
      <th>Score</th>
      <th>Review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18431105</td>
      <td>헌트</td>
      <td>8</td>
      <td>정우성 죽기전 마지막 대사랑 영화 마지막 총소리 이해가 안됨. 평점은 8점 재밌엇음</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18431104</td>
      <td>공조2: 인터내셔날</td>
      <td>10</td>
      <td>외계인 비상선언보다는 훨씬 재밌었음</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18431103</td>
      <td>공조2: 인터내셔날</td>
      <td>10</td>
      <td>지루할틈도 없었고 범죄도시2 이후에 간만에 또 잼나게봄. 개인적으로 1보다 2가 더...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18431102</td>
      <td>육사오(6/45)</td>
      <td>10</td>
      <td>아 진짜 처음부터 끝까지 그냥 계속 웃음ㅋㅋㅋ</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18431101</td>
      <td>매트릭스</td>
      <td>9</td>
      <td>감독의 상상력과 그것을 영화로 탁월하게 표현함이 감탄입니다. 디지털과 기계화 시대 ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>930</th>
      <td>18430089</td>
      <td>육사오(6/45)</td>
      <td>10</td>
      <td>볼만 했음 ㅎㅎㅎㅎ가볍게 웃으며 볼 수 있음</td>
    </tr>
    <tr>
      <th>931</th>
      <td>18430088</td>
      <td>공조2: 인터내셔날</td>
      <td>10</td>
      <td>평점이 왜 이렇게 낮아요? 기대하고 보면 별로겠지만, 기대없이 보면 요즘 한국 영화...</td>
    </tr>
    <tr>
      <th>932</th>
      <td>18430087</td>
      <td>육사오(6/45)</td>
      <td>10</td>
      <td>배우들 연기도 잘하고 정말 개웃기다 ㅋㅋ</td>
    </tr>
    <tr>
      <th>933</th>
      <td>18430086</td>
      <td>육사오(6/45)</td>
      <td>10</td>
      <td>정말 재미있어요 정말 강추 합니다 정말 추천 합니다 이영화 안보면 정말 후회 합니다...</td>
    </tr>
    <tr>
      <th>934</th>
      <td>18430085</td>
      <td>비상선언</td>
      <td>10</td>
      <td>무능했던 정부에 화났던 영화</td>
    </tr>
  </tbody>
</table>
<p>935 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e7703103-9697-45cd-b13c-2ae26d6192d7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-e7703103-9697-45cd-b13c-2ae26d6192d7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e7703103-9697-45cd-b13c-2ae26d6192d7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# csv import 해서 처리할 때
# def data_preprocessing(csv_path):
#   data_set = pd.read_csv(csv_path)
#   data_set = data_set.dropna()
#   train = pd.DataFrame()
#   train["title"] = data_set["Title"]
#   train["review"] = data_set["Review"]
#   train["label"] = data_set["Score"]
#   return train

# Train data set
data_set = df
data_set = data_set.dropna()
train = pd.DataFrame()
train["title"] = data_set["Title"]
train["review"] = data_set["Review"]
train["score"] = data_set["Score"]

# 평점 7점 이상이면 높음 (1), 미만 낮음 (0)
def score_standard(x):
  if x >= 7:
    return 1
  else:
    return 0
```


```python
# 정의한 함수로 apply로 데이터 적용

train["rate_label"] = train["score"].apply(score_standard)
```


```python
train
```





  <div id="df-bbe510ea-e074-451c-bd93-0a2b7eec882c">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>review</th>
      <th>score</th>
      <th>rate_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>헌트</td>
      <td>정우성 죽기전 마지막 대사랑 영화 마지막 총소리 이해가 안됨. 평점은 8점 재밌엇음</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>공조2: 인터내셔날</td>
      <td>외계인 비상선언보다는 훨씬 재밌었음</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>공조2: 인터내셔날</td>
      <td>지루할틈도 없었고 범죄도시2 이후에 간만에 또 잼나게봄. 개인적으로 1보다 2가 더...</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>육사오(6/45)</td>
      <td>아 진짜 처음부터 끝까지 그냥 계속 웃음ㅋㅋㅋ</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>매트릭스</td>
      <td>감독의 상상력과 그것을 영화로 탁월하게 표현함이 감탄입니다. 디지털과 기계화 시대 ...</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>930</th>
      <td>육사오(6/45)</td>
      <td>볼만 했음 ㅎㅎㅎㅎ가볍게 웃으며 볼 수 있음</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>931</th>
      <td>공조2: 인터내셔날</td>
      <td>평점이 왜 이렇게 낮아요? 기대하고 보면 별로겠지만, 기대없이 보면 요즘 한국 영화...</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>932</th>
      <td>육사오(6/45)</td>
      <td>배우들 연기도 잘하고 정말 개웃기다 ㅋㅋ</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>933</th>
      <td>육사오(6/45)</td>
      <td>정말 재미있어요 정말 강추 합니다 정말 추천 합니다 이영화 안보면 정말 후회 합니다...</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>934</th>
      <td>비상선언</td>
      <td>무능했던 정부에 화났던 영화</td>
      <td>10</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>935 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bbe510ea-e074-451c-bd93-0a2b7eec882c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-bbe510ea-e074-451c-bd93-0a2b7eec882c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bbe510ea-e074-451c-bd93-0a2b7eec882c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
train.groupby('title').size() # 특정 컬럼으로 groupby 해서 row count 알고 싶을 때

# 동일한 코드
# test = train.groupby(["title"]).sum()
# print(test)
```




    title
    2012               1
    500일의 썸머           2
    A-특공대              1
    Io                 1
    거짓말                2
                      ..
    헤어질 결심             4
    화니 페이스             2
    환생                 1
    황야의 무법자            1
    흥부: 글로 세상을 바꾼 자    1
    Length: 171, dtype: int64




```python
# 영화 리스트 
movie_list = train['title'].unique().tolist()
movie_list
```




    ['헌트',
     '공조2: 인터내셔날',
     '육사오(6/45)',
     '매트릭스',
     '기억의 밤',
     '서울대작전',
     '비상선언',
     '흥부: 글로 세상을 바꾼 자',
     '외계+인 1부',
     '놉',
     '인생은 뷰티풀: 비타돌체',
     '광대들: 풍문조작단',
     '블랙폰',
     '토르: 러브 앤 썬더',
     '한산: 용의 출현',
     '탑건: 매버릭',
     '헤어질 결심',
     '온리 더 브레이브',
     '터미널 스피드',
     '잉여들의 히치하이킹',
     '뜨거운 피',
     '알라딘',
     '한여름밤의 재즈',
     '모럴센스',
     '데이 시프트',
     '뒤틀린 집',
     '러브 인 더 빌라',
     '그레이 맨',
     '마이펫의 이중생활',
     '그것: 두 번째 이야기',
     '악마를 보았다',
     '황야의 무법자',
     '겟 아웃',
     '엄마까투리',
     '소리도 없이',
     '연애 빠진 로맨스',
     '허드 앤 씬',
     '관상',
     '니 부모 얼굴이 보고 싶다',
     '마녀(魔女) Part2. The Other One',
     '어쩌다 공주, 닭냥이 왕자를 부탁해!',
     '엑시트',
     '오직 사랑뿐',
     '아이 인 더 스카이',
     '카터',
     '새콤달콤',
     '정직한 후보',
     '공조',
     '마고리엄의 장난감 백화점',
     '쿵푸 요가',
     '500일의 썸머',
     '설리: 허드슨강의 기적',
     '라라랜드',
     '노스맨',
     '거짓말',
     '언어의 정원',
     '엄마',
     '테넷',
     '불릿 트레인',
     '삼진그룹 영어토익반',
     '신비한 동물들과 덤블도어의 비밀',
     '명량',
     '시맨틱 에러: 더 무비',
     '보이콰이어',
     '첫눈이 사라졌다',
     '베놈 2: 렛 데어 비 카니지',
     '빌리 홀리데이',
     '도쿄 리벤저스',
     '경관의 피',
     '조제',
     '로미오와 줄리엣',
     '뮤트',
     '스텔라',
     '봄날',
     '민스미트 작전',
     '뱅크시',
     '말모이',
     '캐리비안의 해적 - 블랙 펄의 저주',
     '노인을 위한 나라는 없다',
     '너의 이름은.',
     '안녕하세요',
     '보스 베이비 2',
     '무수단',
     '냉정과 열정 사이',
     '파이어스타터',
     '스파이더맨: 노 웨이 홈',
     '옐로우버드',
     '귀멸의 칼날: 아사쿠사 편',
     '국제수사',
     '태극기 휘날리며',
     '듄',
     '해적: 도깨비 깃발',
     '어웨이크',
     '씽2게더',
     '그대를 사랑합니다',
     '캐롤',
     '분노의 질주: 더 얼티메이트',
     '석양의 무법자',
     '화니 페이스',
     '그것',
     '소림축구',
     '피구의 제왕',
     '익스팅션 - 종의 구원자',
     '야쿠자와 가족',
     '사랑할 땐 누구나 최악이 된다',
     '방울토마토',
     '이터널 선샤인',
     '런',
     '뷰티 인사이드',
     '말아톤',
     '얼라이드',
     '뽀로로 극장판 드래곤캐슬 대모험',
     '크루엘라',
     '썬다운',
     '모가디슈',
     '풀타임',
     '범죄도시2',
     '죽은 시인의 사회',
     '제 7기사단',
     '닥터 스트레인지: 대혼돈의 멀티버스',
     '미나리',
     '스타 이즈 본',
     '쥬라기 월드: 도미니언',
     '다 잘된 거야',
     '고양이를 부탁해',
     '환생',
     '변호인',
     '탑건',
     '서유기 2 - 선리기연',
     '유열의 음악앨범',
     '사랑하니까, 괜찮아',
     '투사부일체',
     '카시오페아',
     '피는 물보다 진하다',
     '제인 도',
     '캐리비안의 해적: 죽은 자는 말이 없다',
     '길버트 그레이프',
     '마크맨',
     '러브 앳',
     '어디선가 누군가에 무슨 일이 생기면 틀림없이 나타난다 홍반장',
     'A-특공대',
     '슈렉',
     '저니스 엔드',
     '스티브 잡스',
     '극장판 안녕 자두야',
     '인어 공주',
     '큰엄마의 미친봉고',
     '더 킬러: 죽어도 되는 아이',
     '성적표의 김민영',
     '몬스터',
     '나는 내일, 어제의 너와 만난다',
     'Io',
     '모스트 원티드 맨',
     '버즈 라이트이어',
     '배트맨 대 슈퍼맨: 저스티스의 시작',
     '블레이즈',
     '2012',
     '부적: 남의 운을 빼앗는 자',
     '킹메이커',
     '마녀',
     '스포트라이트',
     '업',
     '봉자',
     '월터의 상상은 현실이 된다',
     '보통날',
     '뽀로로 극장판 공룡섬 대모험',
     '더 렛지',
     '눈의 여왕4',
     '극장판 바다 탐험대 옥토넛 : 해저동굴 대탈출',
     '남영동1985',
     '보이스']




```python
round(train[train["title"] == '공조2: 인터내셔날']["rate_label"].value_counts()/len(train[train["title"] == '공조2: 인터내셔날']) * 100, 1)
```




    1    84.9
    0    15.1
    Name: rate_label, dtype: float64




```python
train[train["title"] == '공조2: 인터내셔날']["rate_label"]
```




    1      1
    2      1
    5      1
    9      0
    11     1
          ..
    924    1
    925    1
    926    1
    927    1
    931    1
    Name: rate_label, Length: 350, dtype: int64




```python
font_list
```




    ['STIXSizeOneSym',
     'STIXSizeTwoSym',
     'STIXSizeOneSym',
     'DejaVu Sans',
     'STIXSizeFourSym',
     'STIXNonUnicode',
     'STIXSizeTwoSym',
     'STIXGeneral',
     'DejaVu Sans Mono',
     'DejaVu Sans Mono',
     'DejaVu Sans',
     'DejaVu Serif',
     'cmsy10',
     'STIXGeneral',
     'DejaVu Sans Mono',
     'DejaVu Sans Mono',
     'cmtt10',
     'cmr10',
     'DejaVu Sans',
     'STIXSizeFiveSym',
     'cmmi10',
     'DejaVu Sans Display',
     'cmex10',
     'DejaVu Serif Display',
     'cmb10',
     'STIXNonUnicode',
     'STIXSizeThreeSym',
     'STIXNonUnicode',
     'DejaVu Serif',
     'STIXGeneral',
     'DejaVu Serif',
     'STIXNonUnicode',
     'DejaVu Sans',
     'cmss10',
     'STIXGeneral',
     'DejaVu Serif',
     'STIXSizeThreeSym',
     'STIXSizeFourSym',
     'Liberation Mono',
     'Liberation Sans',
     'Liberation Serif',
     'Liberation Sans Narrow',
     'Liberation Serif',
     'Liberation Sans Narrow',
     'Liberation Sans',
     'Liberation Serif',
     'Liberation Mono',
     'Humor Sans',
     'Liberation Sans Narrow',
     'Liberation Sans Narrow',
     'Liberation Mono',
     'Liberation Mono',
     'Liberation Sans',
     'Liberation Sans',
     'Liberation Serif']




```python
plt.rcParams['font.family'] = 'NanumGothic'
```

#### 2-2) 영화 이름 조회 시 리뷰 긍정 부정 점수, 평균 점수 출력


```python
name = input("평점을 알고 싶은 영화 제목을 입력하세요:  ") # 영화 이름 full 입력

def movie_review(name):
   review_cnt = train.groupby('title').size()[name]
   train_pn = round(train[train["title"] == name]["rate_label"].value_counts()/review_cnt * 100, 1)
   train_p = train_pn[1]
   train_n = train_pn[0]
   review_mean = round(train[train["title"] == name]["score"].mean(), 0)

   print('--------------------------------------------------------------------')
   return print(f"{name}의 리뷰 평점은 평균 {review_mean}점\n전체 리뷰 {review_cnt}개에서 긍정 비율 {train_p}% \n부정 비율 {train_n}%")

movie_review(name)
```

    평점을 알고 싶은 영화 제목을 입력하세요:  놉
    --------------------------------------------------------------------
    놉의 리뷰 평점은 평균 6.0점
    전체 리뷰 11개에서 긍정 비율 36.4% 
    부정 비율 63.6%
    


```python
# 특정 영화의 리뷰 긍부정 비율 확인

movies = ['공조2: 인터내셔날', '탑건: 매버릭', '육사오(6/45)', '놉', '헌트']

positives = []
negatives = []
for movie in movies:
  if movie in movie_list:
    movie_review(movie)
    
    review_cnt = train.groupby('title').size()[movie]
    train_pn = round(train[train["title"] == movie]["rate_label"].value_counts()/review_cnt * 100, 1)
    train_p = train_pn[1]
    positives.append(train_p)
    train_n = train_pn[0]
    negatives.append(train_n)
  
  else:
    print("영화 이름을 다시 확인하세요.")
  
print('End')


# 특정 영화 긍부정 비율 통계
index = movies

result = pd.DataFrame({'Positive': positives,
                   'Negative': negatives}, index=index)
result.sort_values('Positive', ascending=False)
```

    --------------------------------------------------------------------
    공조2: 인터내셔날의 리뷰 평점은 평균 9.0점
    전체 리뷰 350개에서 긍정 비율 84.9% 
    부정 비율 15.1%
    --------------------------------------------------------------------
    탑건: 매버릭의 리뷰 평점은 평균 9.0점
    전체 리뷰 23개에서 긍정 비율 91.3% 
    부정 비율 8.7%
    --------------------------------------------------------------------
    육사오(6/45)의 리뷰 평점은 평균 9.0점
    전체 리뷰 83개에서 긍정 비율 88.0% 
    부정 비율 12.0%
    --------------------------------------------------------------------
    놉의 리뷰 평점은 평균 6.0점
    전체 리뷰 11개에서 긍정 비율 36.4% 
    부정 비율 63.6%
    --------------------------------------------------------------------
    헌트의 리뷰 평점은 평균 8.0점
    전체 리뷰 26개에서 긍정 비율 84.6% 
    부정 비율 15.4%
    End
    





  <div id="df-d5086eb4-c2f0-48ac-948d-432a3ef7c06a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Positive</th>
      <th>Negative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>탑건: 매버릭</th>
      <td>91.3</td>
      <td>8.7</td>
    </tr>
    <tr>
      <th>육사오(6/45)</th>
      <td>88.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>공조2: 인터내셔날</th>
      <td>84.9</td>
      <td>15.1</td>
    </tr>
    <tr>
      <th>헌트</th>
      <td>84.6</td>
      <td>15.4</td>
    </tr>
    <tr>
      <th>놉</th>
      <td>36.4</td>
      <td>63.6</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d5086eb4-c2f0-48ac-948d-432a3ef7c06a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d5086eb4-c2f0-48ac-948d-432a3ef7c06a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d5086eb4-c2f0-48ac-948d-432a3ef7c06a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




#### 2-3) 워드 클라우드 만들기


```python
from wordcloud import WordCloud
from konlpy.tag import Twitter
from konlpy.tag import Kkma
from konlpy.tag import Okt
from collections import Counter
import pandas as pd
from tqdm import tqdm_notebook # for문 돌릴 때 진행상황을 %게이지로 알려줌
```


```python
import re

# 특수문자 제거, 형태소 분리 함수
def preprocessing(review, okt, remove_stopwords = False, stop_words = []):
    review_text = re.sub('[^가-힣ㄱ-ㅎ+/ㅏ-ㅣ]','', review) # 한글과 띄어쓰기를 제외한 모든 부분을 제거
    
    # okt.morphs 를 통해 형태소 분석을 해주고 불용어가 없으면 리스트에 추가 후 반환
    review_text = [word for word in okt.morphs(review_text, stem=True) if word not in stop_words]

    if remove_stopwords: # 디폴트 false니까 사용하려면 true로 바꾸기
      clean_reviews = [token for token in review_text if not token in stop_words]

    return clean_reviews

# okt 객체 꼭 지정
okt = Okt()
stop_words = ["는","은","이","가","라고","게","도","에", "등", "한", "되", "수", "있", "의", "랑","의","던","만",",", '없다', '영화', '인'
              ,"을","를","들","고","다", "지", "!", ".", "~", "ㅋㅋ", "ㅎ", ",", "보다", "하다", "좀", "있다", "되다", "그", "이", "저", "중", "과", "같다", "번", "로"]


# 워드클라우드 생성
for movie in tqdm_notebook(movies):
  print('영화 이름:', movie)
  reviews = train.loc[train['title'] == movie]['review']

  tokenized = []
  for review in reviews:
      if type(review) == str:
        token = preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words)
        tokenized.append(token[:])

      else:
        continue

  tokenized = sum(tokenized, [])
  tmp = []
  for s in tokenized:
    for token, tag in okt.pos(s.replace(' ', '')):
        if tag == 'Noun' or tag == 'Adjective':
          tmp.append(token)

  tokenized_s = ' '.join(tmp)

  # 생성되는 워드 클라우드 이미지 확인
  review_cloud = WordCloud(font_path='/content/drive/MyDrive/NanumBarunGothic.ttf', background_color='white', max_words=50, width=1000, height=500).generate(tokenized_s)
  plt.figure(figsize=(10, 8))
  plt.axis('off')
  plt.imshow(review_cloud)
  plt.show()

  print()
```

    영화 이름: 공조2: 인터내셔날
    


    
![image](/images/2022-09-10/output_25_3.png)
    


    
    영화 이름: 탑건: 매버릭
    


    
![image](/images/2022-09-10/output_25_5.png)
    


    
    영화 이름: 육사오(6/45)
    


    
![image](/images/2022-09-10/output_25_7.png)
    


    
    영화 이름: 놉
    


    
![image](/images/2022-09-10/output_25_9.png)
    


    
    영화 이름: 헌트
    


    
![image](/images/2022-09-10/output_25_11.png)
    


    
    


```python

```
