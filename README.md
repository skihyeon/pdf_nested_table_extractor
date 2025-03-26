# PDF Nested Table Extractor

PDF 문서에서 중첩된 테이블과 병합된 셀을 포함한 복잡한 테이블 구조를 추출하기 위한 파이썬 라이브러리입니다.

## 특징

- 복잡한 PDF 테이블 구조 감지
- 병합된 셀 인식
- 테이블 데이터를 Pandas DataFrame으로 변환
- 선택적 시각화 기능
- 테이블을 HTML이나 Excel로 내보내기 가능
- 사용자 정의 가능한 매개변수

## 설치

```bash
pip install pdf-nested-table-extractor
```

## 기본 사용법

```python
from pdf_nested_table_extractor import extract_table
import pandas as pd

# PDF 파일에서 테이블 추출
tables, dataframes = extract_table(
    file_path="your_document.pdf",
    p_num=None,  # 특정 페이지만 처리하려면 페이지 번호 지정
    save_visualization=False,  # 시각화 결과 저장 여부
    visualization_dir="./vis"  # 시각화 저장 디렉토리
)

# 추출된 테이블 확인
print(f"추출된 테이블 수: {len(tables)}")

# DataFrame으로 처리
for i, df in enumerate(dataframes):
    print(f"\n테이블 {i+1}:")
    print(df)
    
    # DataFrame을 Excel 파일로 저장
    df.to_excel(f"table_{i+1}.xlsx", index=False)
```

## 고급 사용법 (클래스 기반)

```python
from pdf_nested_table_extractor import PDFTableExtractor

# 사용자 정의 매개변수로 추출기 초기화
extractor = PDFTableExtractor(
    tolerance=3.0,             # 값 그룹화 시 허용 오차
    row_tolerance=1.5,         # 행 그룹화 허용 오차
    col_tolerance=5.0,         # 열 그룹화 허용 오차
    row_threshold=15,          # 행 너비 필터링 임계값
    cell_merge_tolerance=10.0  # 셀 병합 감지 허용 오차
)

# 시각화 설정
extractor.set_visualization(True, "./output_vis")

# 테이블 추출
tables, dfs = extractor.extract_table_from_file("your_document.pdf")

# 통계 정보 확인
stats = extractor.get_table_statistics()
print(f"추출된 테이블 개수: {stats['count']}")
for table_info in stats['tables']:
    print(f"테이블 {table_info['index']}: {table_info['rows']}행 x {table_info['columns']}열, 병합셀 {table_info['merged_cells']}개")

# HTML로 출력
html_table = extractor.get_table_as_html(0)  # 첫 번째 테이블

# 모든 테이블을 Excel 파일로 저장
extractor.save_tables_to_excel("output_tables.xlsx")
```

## 매개변수 설명

### 기본 매개변수

- `file_path`: PDF 파일 경로
- `p_num`: 특정 페이지 번호 (None이면 모든 페이지 처리)
- `save_visualization`: 시각화 이미지 저장 여부
- `visualization_dir`: 시각화 이미지 저장 디렉토리

### PDFTableExtractor 클래스 매개변수

- `tolerance`: 값 그룹화 시 허용 오차 (기본값: 2.0)
- `row_tolerance`: 행 그룹화 허용 오차 (기본값: 1.0)
- `col_tolerance`: 열 그룹화 허용 오차 (기본값: 5.0)
- `row_threshold`: 행 너비 필터링 임계값 (기본값: 10)
- `cell_merge_tolerance`: 셀 병합 감지 허용 오차 (기본값: 10.0)

## 반환 값

`extract_table` 함수와 `extract_table_from_file` 메서드는 두 가지 값을 반환합니다:

1. `tables`: 테이블 위치, 셀 좌표, 텍스트 등 상세 정보를 포함한 리스트
2. `dataframes`: 추출된 각 테이블의 Pandas DataFrame 리스트

## 시각화 예시

시각화 옵션을 활성화하면 다음과 같은 이미지가 생성됩니다:

- 원본 페이지 이미지
- 감지된 테이블 이미지 (병합셀은 파란색, 일반 셀은 빨간색으로 표시)
- 열 경계선(녹색)과 감지된 세로선(파란색)을 보여주는 디버깅 이미지

## 알고리즘 동작 원리

1. 페이지에서 의미 있는 가로선 추출
2. 근접한 y좌표 기준으로 가로선 그룹화
3. 각 행의 너비 계산
4. 시작 위치와 너비가 유사한 행을 그룹화하여 테이블 후보 찾기
5. 테이블 행 필터링
6. 열 경계선 위치 감지
7. 세로선으로 열 위치 조정
8. 셀 병합 판단 및 처리
9. 셀 내용 추출
10. 결과 저장 및 시각화

## 요구사항

- Python 3.10 이상
- pdfplumber 0.7.0 이상
- numpy 1.20.0 이상
- pandas 1.3.0 이상

## 라이선스

MIT 