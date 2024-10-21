### 타이타닉 생존자 예측

<aside>
💡 **주제**

타이타닉 탑승객 데이터셋을 활용해 생존자를 예측하는 모델을 만드는 프로젝트

</aside>

<aside>
🎯 **목표**
다양한 모델을 통해서 생존자를 예측하는 모델을 학습시키는 것이 목표입니다. 
과제는 순서대로 풀어주세요.

</aside>

### **필수 과제:**

1. **데이터셋 불러오기** 
    
    `seaborn` 라이브러리에 있는 titanic 데이터를 불러옵니다.
    
    - 참고
        
        ```jsx
        import seaborn as sns
        
        titanic = sns.load_dataset('titanic')
        ```
        
2. **feature 분석**
    
    데이터를 잘 불러오셨다면 해당 데이터의 feature를 파악해야합니다. 데이터의 feature를 파악하기 위해 아래의 다양한 feature 분석을 수행해주세요. 
    
    2-1. `head` 함수를 이용해 데이터 프레임의 첫 5행을 출력하여 어떤 feature들이 있는지 확인해주세요. 
    
    - 참고
        
        ```jsx
        titanic.head()
        ```
        
    
     2-2. `describe` 함수를 통해서 기본적인 통계를 확인해주세요. 
    
    - 참고
        
        ```jsx
        titanic.describe()
        ```
        
    
    2-3. `describe` 함수를 통해 확인할 수 있는 `count`, `std`, `min`, `25%`, `50%`, `70%`, `max` 가 각각 무슨 뜻인지 주석 혹은 markdown 블록으로 간단히 설명해주세요. 
    
    - 참고
        
        ```python
        # count: count는 ㅇㅇㅇ한 값으로 ㅇㅇㅇㅇ함을 보여줍니다. ㅇㅇㅇ을 통해 구할 수 있습니다.
        # std: ...
        ```
        
    
    2-4. `isnull()` 함수와 `sum()`  함수를 이용해 각 열의 결측치 갯수를 확인해주세요.
    
    - 참고
        
        ```python
        print(titanic.isnull().sum())
        ```
        
3. **feature engineering**
    
    feature engineering은 모델의 성능을 향상시키기 위해 중요한 단계입니다. 2번 feature 분석에서 얻은 데이터에 대한 이해를 바탕으로 아래의 feature engineering을 수행해주세요. 
    
    **3-1. 결측치 처리**
    
    Age(나이)의 결측치는 중앙값으로, Embarked(승선 항구)의 결측치는 최빈값으로 대체해주세요. 모두 대체한 후에, 대체 결과를 `isnull()` 함수와 `sum()`  함수를 이용해서 확인해주세요. 
    
    - 참고
        
        ```python
        titanic['age'].fillna(titanic['age'].median(), inplace=True)
        titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)
        
        print(titanic['age'].isnull().sum())
        print(titanic['embarked'].isnull().sum())
        ```
        
    
    **3-2. 수치형으로 인코딩**
    
    Sex(성별)를 남자는 0, 여자는 1로 변환해주세요. alive(생존여부)를 True는 1, False는 0으로 변환해주세요. Embarked(승선 항구)는 ‘C’는 0으로, Q는 1으로, ‘S’는 2로 변환해주세요. 모두 변환한 후에, 변환 결과를 `head` 함수를 이용해 확인해주세요. 
    
    - 참고
        
        ```python
        titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
        titanic['alive'] = titanic['alive'].map({'no': 1, 'yes': 0})
        titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2,})
        
        print(titanic['sex'].head())
        print(titanic['alive'].head())
        print(titanic['embarked'].head())
        ```
        
    
    **3-3. 새로운 feature 생성**
    
    SibSip(타이타닉호에 동승한 자매 및 배우자의 수), Parch(타이타닉호에 동승한 부모 및 자식의 수)를 통해서 family_size(가족크기)를 생성해주세요. 새로운 feature를 `head` 함수를 이용해 확인해주세요. 
    
    - 참고
        
        ```python
        titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
        
        print(titanic['family_size'].head())
        ```
        
4. **모델 학습시키기 (Logistic Regression, Random Forest, XGBoost)**
    
    4-1. 모델 학습 준비 
    
    이제 모델을 학습시키기 위한 데이터를 준비하겠습니다. 학습에 필요한 feature은 'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', ‘family_size’ 입니다. feature과 target을 분리해주세요.  그 다음 데이터 스케일링을 진행해주세요. 
    
    - 참고
        
        ```python
        titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]
        X = titanic.drop('survived', axis=1) # feature
        y = titanic['survived'] # target
        ```
        
    
    이제 Logistic Regression, Random Forest, XGBoost를 통해서 생존자를 예측하는 모델을 학습하세요. 학습이 끝난 뒤 Logistic Regression과 Random Forest는 모델 accuracy를 통해, XGBoost는 mean squared error를 통해 test data를 예측하세요. 
    
    4-2. Logistic Regression
    
    - 참고
        
        ```python
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 데이터 스케일링
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 모델 생성 및 학습
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 평가
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        ```
        
    
    4-3. Random Forest
    
    - 참고
        
        ```python
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 데이터 스케일링
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 모델 생성 및 학습
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 평가
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        ```
        
    
    4-4. XGBoost
    
    - 참고