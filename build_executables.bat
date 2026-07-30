@echo off
echo ========================================================
echo   FAB-FL GUI PyInstaller Executable Builder
echo ========================================================
echo.
echo Step 1: Installing PyInstaller...
pip install pyinstaller

echo.
echo Step 2: Building Server GUI (ServerApp.exe)...
cd FAB-FL-Software
pyinstaller --onefile --windowed --paths .. --name ServerApp server_app.py

echo.
echo Step 3: Building Client GUI (ClientApp.exe)...
pyinstaller --onefile --windowed --paths .. --name ClientApp client_app.py
cd ..

echo.
echo ========================================================
echo Build Complete! 
echo You can find your .exe files inside the 'FAB-FL-Software\dist' folder.
echo ========================================================
pause
