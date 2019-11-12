# 07. 머신 러닝(Machine Learning) 개요
#### 참고 : https://wikidocs.net/21669

1. 선형 회귀
    - 엄청난 데이터량을 한꺼번에 계산할 수 없기 때문에 경사하강법과 같은 방법을 사용
        1. 초기 w를 정한다.
        2. w에 대해 계산가능한 최대한의 데이터를 이용해 less를 구한다.
        3. less에 대해 이동할 방향을 결정하고 학습률과 w와 less의 기울기 값에 따 이동 => 0에 가까워질수록 기울기 값이 낮아짐 -> 조금씩 이동, 반대의 경우엔 크게 이동
        4. 0에 최대한 가까워지도록 반복
    
    - 활성함수 : 선형, 시그모이드 함수 등

2. 로지스틱 회귀
    - 분류 문제에 선형회귀가 적합하지 않은 이유
        1. y가 1과 0만 존재하는 S자 형태의 선이 나와 직선으로 표현하기 힘들다
        2. 실제값(y)이 0과 1 값 만을 가지므로 예측값이 0과 1사이의 값밖에 가지지않는데, 선형회귀는 음의 무한대와 양의 무한대 같은 값을 가질 수 있다. 
    => 따라서, 선형함수가 아닌 시그모이드 함수를 써야 함

3. 소프트맥스 회귀
    - 단순하게 인덱스를 부여하는 정수 인코딩이 아닌 원-핫 벡터를 사용하는 이유 : 
        - 레이블이 가까울 수록 연관성이 있다고 보이게 됨  (계산 시 가까운 레이블 끼리의 오차가 먼쪽의 오차보다 작음)