
@echo off
for %%x in (
        "mediumdepth"
        "mediumfisheye"
        "resinc"
        "resdec"
       ) do (
         echo Running validation for "%%x"
         C:\Users\Meriel\Documents\GitHub\supervised-universal-transformation\venv-sutransf\Scripts\python.exe C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation/simulation/take-a-lap-test-baseline4-95-withconfig.py --effect %%x

       )
