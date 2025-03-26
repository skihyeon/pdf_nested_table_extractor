"""
고급 테이블 추출 예제

PDFTableExtractor 클래스를 사용한 고급 기능 활용 예제입니다.
- 사용자 정의 매개변수
- 테이블 통계 정보
- HTML 및 Excel 내보내기
"""

import os
import sys

# 패키지 경로 설정 (개발 환경에서만 필요)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_nested_table_extractor import PDFTableExtractor

def main():
    # 현재 디렉토리 확인
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # PDF 파일 경로 지정 (상대 경로)
    file_path = os.path.join(os.path.dirname(script_dir), "연제구.pdf")
    
    print(f"PDF 파일 처리 중: {file_path}")
    
    # 출력 디렉토리 생성
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # PDFTableExtractor 인스턴스 생성 (사용자 정의 매개변수)
    extractor = PDFTableExtractor(
        tolerance=3.0,              # 값 그룹화 시 허용 오차 증가
        row_tolerance=1.5,          # 행 그룹화 허용 오차 증가
        row_threshold=15,           # 행 너비 필터링 임계값 증가
        cell_merge_tolerance=12.0   # 셀 병합 감지 허용 오차 증가
    )
    
    # 시각화 설정
    extractor.set_visualization(
        save_visualization=True,
        visualization_dir=os.path.join(output_dir, "advanced_visualization")
    )
    
    # 테이블 추출
    tables, dataframes = extractor.extract_table_from_file(file_path)
    
    # 테이블 통계 정보 출력
    stats = extractor.get_table_statistics()
    print(f"\n--- 테이블 통계 정보 ---")
    print(f"추출된 테이블 수: {stats['count']}")
    
    for table_info in stats['tables']:
        print(f"\n테이블 {table_info['index']+1} (페이지 {table_info['page']}):")
        print(f"  - 행 수: {table_info['rows']}")
        print(f"  - 열 수: {table_info['columns']}")
        print(f"  - 병합된 셀 수: {table_info['merged_cells']}")
    
    # 첫 번째 테이블을 HTML로 저장
    if dataframes:
        html_content = extractor.get_table_as_html(0)
        html_path = os.path.join(output_dir, "first_table.html")
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>PDF에서 추출한 테이블</title>
                <style>
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <h1>PDF에서 추출한 테이블</h1>
                {html_content}
            </body>
            </html>
            """)
        
        print(f"\nHTML 파일 저장됨: {html_path}")
    
    # 모든 테이블을 하나의 Excel 파일로 저장
    excel_path = os.path.join(output_dir, "all_tables.xlsx")
    if extractor.save_tables_to_excel(excel_path):
        print(f"Excel 파일 저장됨: {excel_path}")
    else:
        print("Excel 파일 저장 실패")

if __name__ == "__main__":
    main() 