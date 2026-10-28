#!/bin/zsh
cd -- "${0:A:h}" || exit 1
python3 bootstrap.py "$@"
result=$?
if [[ -t 0 ]]; then
  read -r '?按回车关闭窗口。'
fi
exit $result
