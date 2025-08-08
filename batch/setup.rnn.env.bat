@echo off
echo Criando ambiente virtual com Python 3.11...
"C:\Program Files\Python311\python.exe" -m venv venv_rnn

echo Ativando ambiente virtual...
call venv_rnn\Scripts\activate.bat

echo Atualizando pip...
python -m pip install --upgrade pip

echo Instalando bibliotecas necessárias...
pip install tensorflow scikit-learn imbalanced-learn pandas matplotlib

echo Ambiente configurado com sucesso!
echo Para ativar manualmente no futuro, use:
echo     call venv_rnn\Scripts\activate.bat
pause
