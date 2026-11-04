@echo off
REM Start Ask Shorty pointed at the FULL 5,979-video corpus.
REM All searches will use the big database and Chroma index.
REM %USERPROFILE% is your user folder (C:\Users\number2 on Shaun's PC), so the paths are unchanged.

set ASK_SHORTY_DB_PATH=%USERPROFILE%/Desktop/youtube-history-viewer-copy/data/transcripts.db
set ASK_SHORTY_CHROMA_PATH=%USERPROFILE%/Desktop/youtube-history-viewer-copy/data/transcript_chroma_new

echo.
echo  Ask Shorty - FULL CORPUS MODE
echo  DB:     %ASK_SHORTY_DB_PATH%
echo  Chroma: %ASK_SHORTY_CHROMA_PATH%
echo.

python ask_shorty_app.py
