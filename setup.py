from setuptools import setup, find_packages

setup(
    name='pdf_nested_table_extractor',
    version='0.0.1',
    description='PDF Nested Table Extractor - 병합된 셀을 포함한 복잡한 PDF 테이블 추출 라이브러리',
    author='ghseo',
    author_email='ghseo@edentns.com',
    url='https://github.com/skihyeon/pdf_nested_table_extractor',
    install_requires=[
        'pdfplumber>=0.7.0',
        'numpy>=1.20.0',
        'pandas>=1.3.0'
    ],
    packages=['pdf_nested_table_extractor'],
    keywords=['pdf_nested_table_extractor', 'pdf2table', 'pdf dataframe', 'python', 'pypi', 'table extraction'],
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Text Processing :: Markup',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
    ],
)