import pdfplumber
import numpy as np
import os
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

# 로깅 설정
logger = logging.getLogger(__name__)

class PDFTableExtractor:
    """PDF 문서에서 테이블을 추출하는 클래스"""
    
    def __init__(self, 
                 tolerance: float = 2.0, 
                 row_tolerance: float = 1.0,
                 col_tolerance: float = 5.0,
                 row_threshold: int = 10,
                 cell_merge_tolerance: float = 10.0):
        """
        PDF 테이블 추출기 초기화
        
        Args:
            tolerance: 값 그룹화 시 허용 오차
            row_tolerance: 행 그룹화 허용 오차
            col_tolerance: 열 그룹화 허용 오차
            row_threshold: 행 너비 필터링 임계값
            cell_merge_tolerance: 셀 병합 감지 허용 오차
        """
        self.tolerance = tolerance
        self.row_tolerance = row_tolerance
        self.col_tolerance = col_tolerance
        self.row_threshold = row_threshold
        self.cell_merge_tolerance = cell_merge_tolerance
        
        # 시각화 설정
        self.save_visualization = False
        self.visualization_dir = "./vis"
        
        # 결과 저장 
        self.tables = []
        self.dataframes = []
        
    def set_visualization(self, save_visualization: bool, visualization_dir: str = "./vis"):
        """시각화 설정 지정
        
        Args:
            save_visualization: 시각화 저장 여부
            visualization_dir: 시각화 저장 디렉토리
        """
        self.save_visualization = save_visualization
        self.visualization_dir = visualization_dir
        
        if save_visualization:
            os.makedirs(visualization_dir, exist_ok=True)
            
    def group_values(self, values: List[float], tolerance: Optional[float] = None) -> List[List[float]]:
        """값들을 지정된 허용 오차 내에서 그룹화합니다."""
        if not values:
            return []
        
        if tolerance is None:
            tolerance = self.tolerance

        groups = []
        current_group = [values[0]]
        for x in values[1:]:
            if abs(x - current_group[-1]) <= tolerance:
                current_group.append(x)
            else:
                groups.append(current_group)
                current_group = [x]
        if current_group:
            groups.append(current_group)
        return groups

    def extract_horizontal_lines(self, page, page_height):
        """페이지에서 의미 있는 가로 선을 추출합니다."""
        want_lines = []
        
        for edge in page.horizontal_edges:
            if edge["x1"] < 0 or edge["x0"] < 0:
                continue

            y = (edge["top"] + edge["bottom"]) / 2

            # 페이지 상하단 가장자리는 제외
            margin = 0.01
            if y < page_height * margin or y > page_height * (1 - margin):
                continue
                
            want_lines.append({'y': y, 'x0': edge["x0"], 'x1': edge["x1"], 'edge': edge})
        
        # y 좌표 기준으로 정렬
        want_lines.sort(key=lambda x: x['y'])
        return want_lines

    def group_horizontal_lines(self, want_lines):
        """가로 선을 근접한 y좌표 기준으로 그룹화합니다."""
        rows = {}
        
        for line in want_lines:
            midline = round(line['y'], 0)
            added = False
            for key in rows:
                if abs(midline - key) <= self.row_tolerance:
                    rows[key].append(line)
                    added = True
                    break
            if not added:
                rows[midline] = [line]
        
        return rows

    def get_row_widths(self, rows):
        """각 행의 너비를 계산합니다."""
        row_widths = {}
        for row_y, lines in rows.items():
            min_x0 = min(line['x0'] for line in lines)
            max_x1 = max(line['x1'] for line in lines)
            row_widths[row_y] = max_x1 - min_x0
        return row_widths

    def find_table_row_groups(self, rows, row_widths):
        """테이블을 형성하는 행 그룹을 찾습니다."""
        # 모든 행을 하나의 그룹으로 시작
        table_row_groups = []
        table_row_group = list(rows.keys())
        table_row_groups.append(table_row_group)
        
        # 행 시작 위치에 따라 그룹화
        row_start_x = {}
        for row_y, lines in rows.items():
            row_start_x[row_y] = min(line['x0'] for line in lines)
        
        # 시작 위치와 너비가 유사한 행들 클러스터링
        sorted_rows = sorted(row_widths.items(), key=lambda x: x[0])
        x0_clusters = {}
        for row, width in sorted_rows:
            x0 = row_start_x[row]
            matched = False
            for center in list(x0_clusters.keys()):
                if abs(x0-center) <= 2.0:
                    if any(abs(width-w) <=2.0 for _, w in x0_clusters[center]):
                        x0_clusters[center].append((row, width))
                        matched = True
                        break
            if not matched:
                x0_clusters[x0] = [(row, width)]
        
        # 클러스터에서 테이블 후보 행 그룹 추출
        row_groups = []
        for cluster in x0_clusters.values():
            if len(cluster) > 1:
                row_groups.append(cluster)
        
        # 테이블 행 그룹 추가
        for group in row_groups:
            table_rows = sorted([row for row, width in group])
            table_row_groups.append(table_rows)
        
        return table_row_groups

    def filter_table_rows(self, table_row_groups, row_widths):
        """너비가 유사한 행들만 포함하도록 테이블 행을 필터링합니다."""
        filtered_table_rows = []
        
        for table_row in table_row_groups:
            max_width = max(row_widths[r] for r in table_row)
            filtered_row = []
            for r in table_row:
                if abs(row_widths[r] - max_width) <= self.row_threshold:
                    filtered_row.append(r)
            
            if len(filtered_row) > 1:
                filtered_table_rows.append(filtered_row)
        
        return filtered_table_rows

    def find_column_lines(self, rows, table_rows):
        """테이블의 열 경계선 위치를 찾습니다."""
        x_coords_per_row = {}
        for row_y in table_rows:
            x_coords = []
            for edge in rows[row_y]:
                x_coords.append(edge['x0'])
                x_coords.append(edge['x1'])
            x_coords.sort()
            x_coords_per_row[row_y] = x_coords
        
        # 각 행의 열 중심점과 그룹 수 계산
        row_centers = {}
        group_counts = {}
        for row_y, x_coords in x_coords_per_row.items():
            groups = self.group_values(x_coords, self.col_tolerance)
            group_counts[row_y] = len(groups)
            centers = [sum(g)/len(g) for g in groups]
            row_centers[row_y] = centers
        
        # 가장 빈번한 열 수 찾기
        freq_dict = {}
        for count in group_counts.values():
            freq_dict[count] = freq_dict.get(count, 0) + 1
        
        column_lines_len = 0
        max_freq = 0
        for k, v in freq_dict.items():
            if k == 2:  # 열이 2개인 경우는 무시
                continue
            if v > max_freq:
                max_freq = v
                column_lines_len = k
        
        # 열 경계선 위치 계산
        column_lines = [row_centers[row_y] for row_y in row_centers 
                       if group_counts[row_y] == column_lines_len]
        
        if not column_lines:
            return None
        
        # 열 위치의 중앙값 계산
        column_lines = np.array(column_lines)
        column_lines = np.median(column_lines, axis=0).tolist()
        
        return column_lines

    def adjust_column_lines_with_vertical_edges(self, page, column_lines, table_rows):
        """세로 테두리를 이용해 열 경계선 위치를 조정합니다."""
        # 세로선 추출
        vertical_edges_x = []
        for edge in page.vertical_edges:
            if any(top <= edge["top"] <= bottom for top, bottom in zip(table_rows[:-1], table_rows[1:])):
                x = (edge["x0"] + edge["x1"]) / 2
                vertical_edges_x.append(x)
        
        # 열 위치 조정
        if vertical_edges_x:
            for i in range(len(column_lines)):
                closest_edge = min(vertical_edges_x, key=lambda x: abs(x - column_lines[i]))
                # 가까운 세로선이 있으면 위치 조정
                if abs(closest_edge - column_lines[i]) < 20:
                    column_lines[i] = closest_edge
        
        return column_lines

    def get_vertical_edges_in_table(self, page, table_rows):
        """테이블 영역 내의 세로선 정보를 추출합니다."""
        vertical_edges_list = []
        for edge in page.vertical_edges:
            x_mid = (edge["x0"] + edge["x1"]) / 2
            y_mid = (edge["top"] + edge["bottom"]) / 2
            
            # 테이블 영역 내의 세로선만 저장
            if table_rows[0] <= y_mid <= table_rows[-1]:
                height = edge["bottom"] - edge["top"]
                vertical_edges_list.append({
                    'x': x_mid,
                    'y': y_mid,
                    'height': height,
                    'top': edge["top"],
                    'bottom': edge["bottom"]
                })
        
        return vertical_edges_list

    def get_vertical_edges_in_row(self, vertical_edges_list, top, bottom, row_height):
        """현재 행 영역 내의 세로선 위치를 추출합니다."""
        row_v_edges = []
        for edge in vertical_edges_list:
            # 세로선이 행과 충분히 겹치는지 확인
            overlap = min(edge['bottom'], bottom) - max(edge['top'], top)
            if overlap > row_height * 0.3:  # 행 높이의 30% 이상 겹치면 포함
                row_v_edges.append(edge['x'])
        
        return row_v_edges

    def process_merged_cells(self, column_lines, row_v_edges):
        """병합된 셀을 감지하고 처리합니다."""
        cells = []
        j = 0
        
        while j < len(column_lines) - 1:
            start_j = j  # 병합 시작 열 인덱스
            x0 = column_lines[j]
            
            # 다음 열과 병합 가능한지 확인
            while j < len(column_lines) - 2:  # 마지막 열 직전까지 확인
                next_x0 = column_lines[j + 1]
                
                # 두 열 경계선 사이에 세로선이 있는지 확인
                has_vertical_line = False
                
                for v_x in row_v_edges:
                    # 1. 세로선이 열 경계선 근처인지 확인
                    is_near_column_line = abs(v_x - next_x0) <= self.cell_merge_tolerance
                    
                    # 2. 세로선이 현재 열과 다음 열 사이에 있는지 확인
                    is_between_columns = (x0 + self.cell_merge_tolerance < v_x < next_x0 - self.cell_merge_tolerance)
                    
                    # 둘 중 하나라도 해당되면 셀을 분리해야 함
                    if is_near_column_line or is_between_columns:
                        has_vertical_line = True
                        break
                
                if has_vertical_line:
                    # 세로선이 있으면 병합 중단
                    break
                else:
                    # 세로선이 없으면 다음 열도 병합
                    j += 1
            
            # 나머지 코드는 동일
            x1 = column_lines[j + 1]
            is_merged = (j > start_j)
            
            cells.append({
                'x0': x0,
                'x1': x1,
                'merged': is_merged,
                'colspan': (j - start_j + 1) if is_merged else 1,
                'col_start': start_j,
                'col_end': j
            })
            
            j += 1  # 다음 열로 이동
        
        return cells

    def extract_cell_content(self, page, cell, top, bottom):
        """셀 영역에서 텍스트를 추출합니다."""
        x0, x1 = cell['x0'], cell['x1']
        text = ""
        
        try:
            # 약간의 여백을 두고 추출 시도
            crop_cell = page.crop((x0 + 5, top, x1 - 5, bottom))
            text = crop_cell.extract_text() or ""
        except Exception as e:
            logger.debug(f"셀 내용 추출 중 오류(여백 있음): {e}")
            try:
                # 여백 없이 다시 시도
                crop_cell = page.crop((x0, top, x1, bottom))
                text = crop_cell.extract_text() or ""
            except Exception as e:
                logger.debug(f"셀 내용 추출 중 오류(여백 없음): {e}")
        
        return text.strip()
    
    def extract_table_from_file(self, file_path: str, p_num: Optional[int] = None) -> Tuple[List, List]:
        """
        PDF 파일에서 테이블을 추출합니다.
        
        Args:
            file_path: PDF 파일 경로
            p_num: 특정 페이지 번호. None이면 모든 페이지 처리
            
        Returns:
            (테이블 정보 리스트, Pandas DataFrame 리스트)
        """
        # 결과 초기화
        self.tables = []
        self.dataframes = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                self._process_pdf(pdf, p_num)
        except Exception as e:
            logger.error(f"PDF 처리 중 오류 발생: {e}")
            
        return self.tables, self.dataframes
        
    def _process_pdf(self, pdf, p_num=None):
        """PDF 객체 처리"""
        for i, page in enumerate(pdf.pages):
            page_number = i + 1
            
            if p_num and page_number != p_num:
                continue
                
            page_width, page_height = page.width, page.height
            
            # 시각화 이미지 생성
            img = None
            if self.save_visualization:
                img = page.to_image(resolution=72)
                save_name = f"{self.visualization_dir}/crop_big_table_{page_number}.png"
                img.save(save_name, format="PNG")
            
            # 1. 가로선 추출
            want_lines = self.extract_horizontal_lines(page, page_height)
            if not want_lines:
                logger.info(f"페이지 {page_number}에서 가로선을 찾지 못했습니다.")
                continue
                
            # 2. 가로선 그룹화
            rows = self.group_horizontal_lines(want_lines)
            
            # 3. 행 너비 계산
            row_widths = self.get_row_widths(rows)
            
            # 4. 테이블 행 그룹 찾기
            table_row_groups = self.find_table_row_groups(rows, row_widths)
            
            # 5. 테이블 행 필터링
            filtered_table_rows = self.filter_table_rows(table_row_groups, row_widths)
            
            # 6. 각 테이블 처리
            for table_idx, table_rows in enumerate(filtered_table_rows):
                copy_img = None
                if self.save_visualization and img:
                    copy_img = img.copy()
                
                # 7. 열 경계선 찾기
                column_lines = self.find_column_lines(rows, table_rows)
                if not column_lines:
                    logger.info(f"테이블 {table_idx}에서 열 경계선을 찾지 못했습니다.")
                    continue
                    
                # 8. 세로선으로 열 위치 조정
                column_lines = self.adjust_column_lines_with_vertical_edges(page, column_lines, table_rows)
                
                # 9. 테이블 내 세로선 정보 추출
                vertical_edges_list = self.get_vertical_edges_in_table(page, table_rows)
                
                table_cells = []
                table_data = []
                
                try:
                    # 10. 각 행 처리
                    for i in range(len(table_rows) - 1):
                        row_cells = []
                        row_data = []
                        top, bottom = table_rows[i], table_rows[i + 1]
                        row_height = bottom - top
                        
                        # 11. 현재 행의 세로선 추출
                        row_v_edges = self.get_vertical_edges_in_row(
                            vertical_edges_list, top, bottom, row_height
                        )
                        
                        # 12. 병합 셀 처리
                        processed_cells = self.process_merged_cells(column_lines, row_v_edges)
                        
                        # 13. 셀 내용 추출 및 시각화
                        for cell in processed_cells:
                            x0, x1 = cell['x0'], cell['x1']
                            is_merged = cell['merged']
                            
                            # 시각화: 병합 셀은 파란색, 일반 셀은 빨간색
                            if self.save_visualization and copy_img:
                                copy_img.draw_rect(
                                    (x0, top, x1, bottom),
                                    stroke_width=1,
                                    stroke='blue' if is_merged else 'red'
                                )
                            
                            # 셀 내용 추출
                            cell_text = self.extract_cell_content(page, cell, top, bottom)
                            row_data.append(cell_text)
                            
                            # 셀 정보 저장
                            cell_info = {
                                "x0": x0, "x1": x1,
                                "top": top, "bottom": bottom,
                                "text": cell_text
                            }
                            
                            # 병합된 셀인 경우 추가 정보 저장
                            if is_merged:
                                cell_info["merged"] = True
                                cell_info["colspan"] = cell['colspan']
                            
                            row_cells.append(cell_info)
                        
                        table_cells.append(row_cells)
                        table_data.append(row_data)
                    
                    # 테이블 셀 정보 저장
                    table_info = {
                        'page': page_number,
                        'table': table_cells,
                        'table_bbox': (column_lines[0], table_rows[0], column_lines[-1], table_rows[-1]),
                        'column_count': len(column_lines) - 1,
                        'row_count': len(table_rows) - 1
                    }
                    self.tables.append(table_info)
                    
                    # DataFrame 생성 및 저장
                    df = pd.DataFrame(table_data)
                    if len(df) > 0 and len(df.columns) > 0:
                        try:
                            # 첫 번째 행을 헤더로 사용
                            df.columns = df.iloc[0]
                            df = df.iloc[1:]
                            df = df.reset_index(drop=True)
                            
                            # NaN 값 처리
                            df = df.replace('', np.nan)
                            
                            # DataFrame에 메타데이터 추가
                            df.attrs['page_number'] = page_number
                            df.attrs['table_index'] = table_idx
                            df.attrs['bbox'] = table_info['table_bbox']
                        except Exception as e:
                            logger.warning(f"DataFrame 처리 중 오류: {e}")
                    
                    self.dataframes.append(df)
                    
                    # 테이블의 열 경계선과 세로선 시각화
                    if self.save_visualization and img:
                        debug_img = img.copy()

                        # 열 경계선 그리기 (녹색)
                        for x in column_lines:
                            debug_img.draw_line(
                                [(x, table_rows[0]), (x, table_rows[-1])],
                                stroke='green',
                                stroke_width=2
                            )

                        # 감지된 세로선 그리기 (파란색)
                        for edge in vertical_edges_list:
                            debug_img.draw_line(
                                [(edge['x'], edge['top']), (edge['x'], edge['bottom'])],
                                stroke='blue',
                                stroke_width=1
                            )

                        debug_img.save(f"{self.visualization_dir}/debug_lines_{page_number}_{table_idx}.png", format="PNG")
                    
                except Exception as e:
                    logger.error(f"테이블 {table_idx} 처리 중 오류: {e}")
                    continue
                
                # 테이블 시각화 저장
                if self.save_visualization and copy_img:
                    save_name = f"{self.visualization_dir}/crop_big_table_{page_number}_{table_idx}.png"
                    copy_img.save(save_name, format="PNG")
    
    def get_table_as_html(self, table_index=0):
        """특정 테이블을 HTML 형식으로 반환합니다."""
        if table_index >= len(self.dataframes):
            return "<p>테이블을 찾을 수 없습니다.</p>"
            
        df = self.dataframes[table_index]
        return df.to_html(index=False, na_rep='')
    
    def save_tables_to_excel(self, output_path, sheet_names=None):
        """모든 테이블을 Excel 파일로 저장합니다."""
        if not self.dataframes:
            logger.warning("저장할 테이블이 없습니다.")
            return False
            
        try:
            with pd.ExcelWriter(output_path) as writer:
                for i, df in enumerate(self.dataframes):
                    sheet_name = f"Table_{i+1}" if sheet_names is None else sheet_names[i]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            return True
        except Exception as e:
            logger.error(f"Excel 저장 중 오류: {e}")
            return False
            
    def get_table_statistics(self):
        """추출된 테이블에 대한 통계 정보를 반환합니다."""
        if not self.tables:
            return {"count": 0}
            
        stats = {
            "count": len(self.tables),
            "tables": []
        }
        
        for i, table in enumerate(self.tables):
            table_stats = {
                "index": i,
                "page": table.get('page', 0),
                "rows": table.get('row_count', 0),
                "columns": table.get('column_count', 0),
                "merged_cells": sum(1 for row in table.get('table', []) 
                                  for cell in row if cell.get('merged', False))
            }
            stats["tables"].append(table_stats)
            
        return stats

# 기존 함수 인터페이스 유지를 위한 래퍼 함수
def extract_table(file_path: str, 
                 p_num: Optional[int] = None, 
                 save_visualization: bool = False, 
                 visualization_dir: str = "./vis",
                 **kwargs) -> Tuple[List, List]:
    """
    PDF 파일에서 테이블을 추출합니다. (함수형 인터페이스)
    
    Args:
        file_path: PDF 파일 경로
        p_num: 특정 페이지 번호. None이면 모든 페이지 처리
        save_visualization: 시각화 이미지를 저장할지 여부
        visualization_dir: 시각화 이미지를 저장할 디렉토리
        **kwargs: PDFTableExtractor에 전달할 추가 매개변수
        
    Returns:
        (테이블 정보 리스트, Pandas DataFrame 리스트)
    """
    extractor = PDFTableExtractor(**kwargs)
    extractor.set_visualization(save_visualization, visualization_dir)
    return extractor.extract_table_from_file(file_path, p_num) 