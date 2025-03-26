"""
기본 테이블 추출 예제

기본 함수형 인터페이스를 사용하여 PDF에서 테이블을 추출하는 예제입니다.
"""

from pdf_nested_table_extractor import extract_table
import pandas as pd
import os

def main():
    # 현재 디렉토리 확인
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # PDF 파일 경로 지정 (상대 경로)
    file_path = os.path.join(os.path.dirname(script_dir), "연제구.pdf")
    
    print(f"PDF 파일 처리 중: {file_path}")
    
    # 출력 디렉토리 생성
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 테이블 추출 (시각화 옵션 활성화)
    tables, dataframes = extract_table(
        file_path=file_path,
        save_visualization=True,
        visualization_dir=os.path.join(output_dir, "visualization")
    )
    
    # 추출된 테이블 정보 출력
    print(f"추출된 테이블 수: {len(tables)}")
    
    # 각 테이블의 DataFrame 출력
    for i, df in enumerate(dataframes):
        print(f"\n테이블 {i+1}:")
        print(df.head())  # 데이터가 많을 경우 앞부분만 출력
        
        # DataFrame을 Excel 파일로 저장
        excel_path = os.path.join(output_dir, f"table_{i+1}.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"테이블 {i+1} Excel로 저장됨: {excel_path}")

if __name__ == "__main__":
    main() 