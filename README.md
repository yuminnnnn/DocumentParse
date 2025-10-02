# Docling 기반 문서 파싱 및 벤치마킹 시스템
<br />
<br />  

## Tech Stack

<img src="https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/jupyter-F37626?style=flat&logo=jupyter&logoColor=white"/>

<br />

## Description
이 프로젝트는 **docling** 라이브러리를 활용하여 문서를 효율적으로 파싱하고, 여러 Vision-Laguage Model(Gemma, Llama, Qwen 등) 기반 파서의 성능을 벤치마킹 및 평가하기 위한 프레임워크입니다.
문서에서 텍스트, 레아이웃, 수학수식, 테이블 등을 정확하게 추출하고 분석하는 데 중점을 둡니다.
이미지 기반 문서 처리를 위한 OCR 기능을 지원하여 고품질의 파싱 결과를 얻을 수 있도록 돕습니다.

<br />

## Features
- Docling 라이브러리 통합: 강력한 문서 처리 및 변환 기능을 제공하는 docling 라이브러리를 핵심으로 사용합니다.
- OCR 기능: 이미지 내 텍스트를 인식하여 추출하고 파싱 과정에 통합하여 처리합니다. 
- 종합적인 벤치마킹: 추춛된 문서의 레이아웃, 정확도, 수학 수식 정화도, 테이블 구조 보존, 텍스트 추출 품질 등 다각적인 평가 지표를 통해 파서의 성능을 정량적으로 분석합니다.

<br />
  
## Installation

### Python 환경 설정
Python 3.11 버전이 필요합니다.

### 의존성 설치
이 프로젝트는 uv를 사용하여 의존성을 관리합니다.
uv가 설치되어 있지 않다면 먼저 설치합니다.
```
pip install uv
```
프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 모든 의존성을 설치합니다.
```
uv sync
```
docling 라이브러리는 project.toml에 Git소스로 명시되어 있으므로 uv sync 명령어가 자동으로 처리합니다.

<br />

## Usage

### 1. 문서 파싱 실행

doc_parser/main.py 스크립트를 사용하여 문서를 파싱할 수 있습니다.
```
# 예시
python doc_parser/main.py doc_parser/data/<문서이름>.pdf
```

### 1-1. Jupyter Notebook을 통합 실행

runner/ 디렉토리에는 파싱 및 벤치마킹 과정을 시연하는 Jupyter Notebook 파일이 있습니다.

- **runner/docling_parser_no_ocr.ipynb**: OCR 없이 문서를 파싱하는 예시
- **runner/docling_parser_with_ocr.ipynb**: OCR을 포함하여 문서를 파싱하는 예시 

Jupyter Notebook을 실행하여 이 파일들을 열고 지시에 따라 실행할 수 있습니다.

### 2. 벤치마킹 실행

md-bench/evaluate.py 스크립트를 사용하여 파싱된 결과와 골드 표준 데이터를 비교하여 벤치마킹을 수행합니다.
```
# 예시
python md-bench/evaluate.py --gold data/gold/<파일이름>.md --pred data/pred/<파일이름>.md
```

<br />

## Project Structure

```
├── data/
│   ├── gold/                  
│   │   └── ...
│   ├── pred/                  
│       ├── pred_gemma/
│       │   └── ...
│       └── pred_qwen/
│           └── ...
├── doc_parser/
│   ├── data/                  
│   │   └── ...
│   ├── output/                
│   │   ├── final_gemma/
│   │   └── final_qwen/
│   ├── src/                  
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── docling_setup.py
│   │   └── processing.py
│   └── main.py                
├── docling/                  
│   └── docling/
├── md-bench/
│   ├── src/                   
│   │   ├── layout_evaluation.py
│   │   ├── math_evaluation.py
│   │   ├── table_evaluation.py
│   │   └── text_evaluation.py
│   └── evaluate.py           
├── runner/
│   ├── docling_parser_no_ocr.ipynb  
│   └── docling_parser_with_ocr.ipynb 
├── pyproject.toml            
├── uv.lock                    
└── .gitignore                

```

<br />

## Benchmarking and Evaluation

**md-bench/scr** 디렉토리에는 다음과 같은 평가 스크립트가 포함되어 있습니다.

- layout_evaluation.py: 문서 구조 보존력 평가
- math_evaluation.py: 수학 수식의 추출 정확도 평가
- table_evaluation.py: 테이블 구조 및 내용 추출 정확도 평가
- text_evaluation.py: 문서 전체 텍스트 추출 정확도 평가


이 스크립트들을 통해 각 파서의 강점과 약점을 파악하고 문서 파싱 성능을 종합적으로 개선할 수 있습니다.








