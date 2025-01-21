:: Create Enviroment
python -m venv venv
call venv\Scripts\activate

:: Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

:: Execute program
python -m src.main
