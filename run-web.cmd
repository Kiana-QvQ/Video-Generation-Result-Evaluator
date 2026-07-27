@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "CACHE=%ROOT%model_cache"

if not exist "%PYTHON%" (
  echo Project environment is missing. Run setup.ps1 first.
  exit /b 1
)

set "PYTHONNOUSERSITE=1"
set "TORCH_HOME=%CACHE%"
set "HF_HOME=%CACHE%\huggingface"
set "HF_HUB_CACHE=%CACHE%\huggingface\hub"
set "HF_DATASETS_CACHE=%CACHE%\huggingface\datasets"
set "TRANSFORMERS_CACHE=%CACHE%\huggingface\transformers"
set "TORCH_EXTENSIONS_DIR=%CACHE%\torch_extensions"
set "MPLCONFIGDIR=%CACHE%\matplotlib"
set "GRADIO_TEMP_DIR=%ROOT%outputs\gradio_temp"
set "EVALUATOR_FACE_DEVICE=cpu"
set "EVALUATOR_IQA_DEVICE=cpu"
set "EVALUATOR_SEMANTIC_DEVICE=cpu"

pushd "%ROOT%"
"%PYTHON%" -m uvicorn web_app:app --host 127.0.0.1 --port 7860
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
