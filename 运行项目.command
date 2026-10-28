#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if ! command -v python3 >/dev/null; then
  print '未找到 Python 3，请先安装 Python 3.12。'
  read -r '?按回车关闭窗口。'
  exit 1
fi
mkdir -p logs || exit 1
setopt PIPE_FAIL
log_file="logs/run-$(date +%Y%m%d-%H%M%S)-$$.log"
python3 bootstrap.py 2>&1 | tee "$log_file"
if (( $? != 0 )); then
  print "环境配置失败，日志：$log_file"
  read -r '?按回车关闭窗口。'
  exit 1
fi
.venv/bin/python main.py "$@" 2>&1 | tee -a "$log_file"
result=$?
if (( result == 0 )); then
  split_name=train
  args=("$@")
  for (( i=1; i <= $#args; i++ )); do
    if [[ "${args[$i]}" == --split && $i -lt $#args ]]; then
      split_name="${args[$((i+1))]}"
    elif [[ "${args[$i]}" == --split=* ]]; then
      split_name="${args[$i]#--split=}"
    fi
  done
  print "运行成功，日志：$log_file"
  if [[ -t 0 ]]; then
    open "outputs/$split_name"
  fi
else
  print "运行失败，请查看日志：$log_file"
fi
if [[ -t 0 ]]; then
  read -r '?按回车关闭窗口。'
fi
exit $result
