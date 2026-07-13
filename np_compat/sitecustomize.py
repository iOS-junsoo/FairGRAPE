# imgaug 0.4.0 호환: numpy 1.24+에서 제거된 구식 alias 복원
# PYTHONPATH에 이 디렉터리를 넣은 실행에서만 적용된다.
try:
    import numpy as _np
    for _name, _alias in {'bool': bool, 'int': int, 'float': float,
                          'complex': complex, 'object': object, 'unicode': str}.items():
        if not hasattr(_np, _name):
            setattr(_np, _name, _alias)
except Exception:
    pass
