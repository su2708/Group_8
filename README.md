### 영화 리뷰 감성 분석

<aside>
💡  **주제**
영화 리뷰 데이터를 사용하여 긍정적/부정적 감정을 분류하는 모델을 만드는 프로젝트

</aside>

<aside>
🎯 **목표**
Netflix의 영화 리뷰 데이터를 사용하여, 리뷰의 평점을 예측해보고, 긍정과 부정 감정을 분류해보는 것이 목표입니다. 과제는 순서대로 풀어주세요.

</aside>

### **필수 과제:**

1. **데이터셋 불러오기**
    
    [이 링크](https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated)에서 데이터셋을 다운로드한 후에 데이터셋을 불러오고, 불러온 데이터프레임의 상단 5개의 데이터와 하단 5개의 데이터, 컬럼과 shape를 불러오는 코드를 작성해주세요.
    <details>
    <summary>**참고**</summary>
    <div markdown="1">
        
    ```python
    import pandas as pd
    
    df = pd.read_csv("다운로드 받은 csv 파일.csv")  # 파일 불러오기
    ```
    
    아래와 같은 형태의 결과가 출력되어야 합니다.
    
    ```
    Shape of the dataset: (113068, 8)
    Columns in the dataset: Index(['reviewId', 'userName', 'content', 'score', 'thumbsUpCount',
           'reviewCreatedVersion', 'at', 'appVersion'],
          dtype='object')
    ```
        
    </div>
    </details>
   
        
2. **데이터 전처리**
    
    텍스트 데이터에는 불용어(쓸모없는 단어, 구두점 등)가 많습니다. 해당 부분을 없애주는 처리가 필요합니다. 텍스트 데이터에 대한 전처리를 해주세요.

    <details>
    <summary>**참고**</summary>
    <div markdown="1">
        
    ```python
    # 전처리 함수
    def preprocess_text(text):
        if isinstance(text, float):
            return ""
        text = text.lower()  # 대문자를 소문자로
        text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
        text = re.sub(r'\d+', '', text)  # 숫자 제거
        text = text.strip()  # 띄어쓰기 제외하고 빈 칸 제거
        return text
    ```
        
    </div>
    </details>
    
        
3. **feature 분석 (EDA)**
    
    데이터를 잘 불러오셨다면 해당 데이터의 feature를 찾아야 합니다. 해당 넷플릭스의 데이터에는 **리뷰가 1점부터 5점**까지 있습니다. **해당 데이터의 분포를 그래프**로 그려주세요. 

    <details>
    <summary>**참고**</summary>
    <div markdown="1">
        
    ```python
    import seaborn as sns  # 그래프를 그리기 위한 seaborn 라이브러리 임포트 (없으면 설치 바랍니다)
    import matplotlib.pyplot as plt  # 그래프 표시를 위한 pyplot 임포트
    
    sns.barplot(x=리뷰컬럼, y=리뷰갯수)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Distribution of Scores')
    plt.show()
    ```
        
    </div>
    </details>

        
4. **리뷰 예측 모델 학습시키기 (LSTM)**
    
    이제 어떤 리뷰를 쓰면 점수가 어떻게 나올지에 대해서 예측을 해보고 싶습니다. 로지스틱 회귀 등을 사용하여, 리뷰에 대한 점수 예측을 진행해보세요

    <details>
    <summary>**참고 예시**</summary>
    <div markdown="1">
        
    ```python
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    
    # 데이터셋 클래스 정의
    class ReviewDataset(Dataset):
        def __init__(self, reviews, ratings, text_pipeline, label_pipeline):
            self.reviews = reviews
            self.ratings = ratings
            self.text_pipeline = text_pipeline
            self.label_pipeline = label_pipeline
    
        def __len__(self):
            return len(self.reviews)
    
        def __getitem__(self, idx):
            review = self.text_pipeline(self.reviews[idx])
            rating = self.label_pipeline(self.ratings[idx])
            return torch.tensor(review), torch.tensor(rating)
    
    # 데이터셋 정의
    train_dataset = ReviewDataset(train_reviews, train_ratings, text_pipeline, label_pipeline)
    test_dataset = ReviewDataset(test_reviews, test_ratings, text_pipeline, label_pipeline)
    
    # 데이터 로더 정의
    BATCH_SIZE = 64
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # LSTM 모델 정의
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
            super(LSTMModel, self).__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
    
        def forward(self, text):
            embedded = self.embedding(text)
            output, (hidden, cell) = self.lstm(embedded.unsqueeze(0))
            return self.fc(hidden[-1])
    
    # 하이퍼파라미터 정의
    VOCAB_SIZE = len(vocab)
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    OUTPUT_DIM = len(set(ratings))  # 예측할 점수 개수
    
    # 모델 초기화
    model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 모델 학습은 직접 작성해보세요!!!
    
    # 예측 함수(예시)
    def predict_review(model, review):
        model.eval()
        with torch.no_grad():
            tensor_review = torch.tensor(text_pipeline(review))
            output = model(tensor_review)
            prediction = output.argmax(1).item()
            return label_encoder.inverse_transform([prediction])[0]
    
    # 새로운 리뷰에 대한 예측
    new_review = "This app is great but has some bugs."
    predicted_score = predict_review(model, new_review)
    print(f'Predicted Score: {predicted_score}')
    ```
        
    </div>
    </details>
   

### 추가 선택 과제 (영화 리뷰 감성 분석)

1. **NLP 이용해보기**
    
    본격적으로 NLP를 이용해보는 시간이 필요합니다. 라이브러리들(nltk, gensim, textblob)을 설치하고 리뷰들의 감성을 분류해보세요. **참고자료를 꼭 읽어주세요.**

    <details>
    <summary>**(필수) 참고**</summary>
    <div markdown="1">
        
    아나콘다 프롬프트에서 라이브러리를 먼저 설치하셔야 합니다. **(Anaconda3 사용 필수!!!)**
        
    ```bash
    pip install nltk textblob gensim
    ```
    
    ```python
    # 텍스트 전처리와 자연어 처리를 위한 라이브러리
    import nltk
    from textblob import TextBlob
    
    # 토픽 모델링을 위한 라이브러리
    import gensim
    from gensim import corpora
    from gensim.utils import simple_preprocess
    
    # 감성 분석을 위한 함수
    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity
    
    df['sentiment'] = ...  # (DIY) apply를 사용하여 감성 분석을 해보세요. 필수 텍스트가 전처리되어있어야 합니다.
    # df에 sentiment 값을 적용을 먼저 하시고, 아래와 같이 긍정과 부정을 분류하세요.
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))
    df[['content_c', 'score', 'sentiment_label']]
    ```
    
    **아래**와 같이 결과가 나와야 합니다
    
    | content_c | score | sentiment_label |
    | --- | --- | --- |
    | netfix canada forced my wife into a screen tha... | 1 | neutral |
    | i use this app until it asks if im still there... | 2 | negative |
    | boycott netflix from bharat | 1 | neutral |
    | little good movies and a lot of wonderful tv s... | 5 | positive |
    | new to this but so far smooth sailingapp is ea... | 5 | positive |
    | ... | ... | … |
    | i really like it there are so many movies and ... | 5 | positive |
    | i love netflix i always enjoy my time using it | 5 | positive |
    | sound quality is very slow of movies | 1 | neutral |
    | rate is very expensive bcos we see netflix sun... | 1 | negative |
    | this app is awesome for english movies series ... | 4 | positive |
        
    </div>
    </details>
    
2. **긍정 / 부정 리뷰의 워드 클라우드 그려보기**
    
    여러분은 워드 클라우드를 그려서, 어떤 리뷰의 내용이 제일 많은지에 대해서 쉽게 확인해보고 싶습니다. wordcloud 라이브러리를 설치하고, **긍정과 부정의 워드 클라우드**를 그려보세요.
    
    - **워드 클라우드란?**  문서의 키워드, 개념 등을 직관적으로 파악할 수 있도록 핵심 단어를 시각화하는 기법. 예를 들면 많이 언급될수록 단어를 크게 표현해 한눈에 들어올 수 있게 하는 기법 등이 있습니다.
    <details>
    <summary>**(필수) 참고**</summary>
    <div markdown="1">
        
     ```python
    from wordcloud import WordCloud, STOPWORDS

    # (선택) 불용어를 먼저 제거해주세요.
    stopwords = set(STOPWORDS)
    stopwords.update(['netflix', 'movie', 'show', 'time', 'app', 'series', 'phone'])  # 리뷰에서 필요없는 단어는 여기 안에 추가하셔도 좋습니다.

    # 부정적인 리뷰만 먼저 모아본 다음, 아래처럼 wordcloud를 그려보세요
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(negative_reviews)

    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negative Reviews Word Cloud')
    plt.show()
    ```
        
    </div>
    </details>
    
