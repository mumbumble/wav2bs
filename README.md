# wav2bs


https://github.com/user-attachments/assets/40e34921-5e98-4c57-94d3-d6f191c3f815


성우의 연기 음성을 기반으로 ARKit 기반 Blendshape Facial Animation을 생성하는 End-to-End 딥러닝 모델입니다. 음성의 스타일, 감정, 화자 특성을 반영한 고품질 립싱크를 목표로 합니다.

## 🧠 개요

- 성우 연기 음성 기반의 자연스럽고 감정 표현이 가능한 립싱크 생성
- wav2vec 2.0 + Transformer 기반 Diffusion Model을 결합
- 긴 오디오 시퀀스에서도 일반화 가능하도록 ALiBi 및 PPE 기법 도입

## 🧩 모델 구성
![{7F699E96-8632-4C5E-B037-38E89ED0AA2B}](https://github.com/user-attachments/assets/ec04dd0f-2408-4a8b-b0ac-cbc7b661c2ca)

1. **Audio Encoder**
   - wav2vec 2.0 사용
   - 사전학습된 모델을 파인튜닝하여 소규모 데이터로도 고성능 기대
   - 음성 내 화자의 특성까지 인코딩 가능

2. **Motion Decoder**
   - Transformer 기반 Diffusion Denoiser 구조
   - 감정 표현 및 비언어적 소리에 강한 표현력
   - ALiBi(시간편향 Attention), PPE(주기적 위치 인코딩) 적용

3. **Style ID**
   - 동일한 입력 음성에 대해 다른 스타일(캐릭터)의 립싱크 생성 가능

## 🧪 학습 정보

- 총 약 400분 데이터 (여성 63분 + 남성 124분 + 음성 변환 포함)
- 2500 epoch 학습
- Loss 구성: Reconstruction Loss + Velocity Loss

## ⚡ 성능

- 20초 오디오 기준 추론 시간: 약 3.9초 (A2F 대비 10배 이상 빠름)
- BS weight 출력으로 추가 solving 불필요
- 연기 톤, 감정 표현, 비언어적 소리(비명, 웃음) 표현 가능

## 🚧 한계점

- 데이터 다양성 부족 시 표현력 제한
- 다수 화자 데이터 적용 시 스타일 유지 성능 추가 검증 필요

## 📌 참고 문헌

- DiffSpeaker: Speech-Driven 3D Facial Animation with Diffusion Transformer  
  https://arxiv.org/pdf/2402.05712  
- FaceFormer: Speech-Driven 3D Facial Animation with Transformers  
  https://evelynfan.github.io/audio2face/
