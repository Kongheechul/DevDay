#!/bin/zsh
set -eu
cd "$(dirname "$0")"
python3 -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10 이상이 필요합니다"'
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
if [[ ! -f settings.py ]]; then
  cp settings.example.py settings.py
fi
mkdir -p input
echo '설치 완료. settings.py에 영상 경로, API 키, 촬영 FPS, 모델을 설정하세요.'
echo '그다음 run_analysis.command를 실행하세요.' 
