import subprocess

program_list = [
    'run_test0.py',
    #'run_test1.py',
    #'run_test2.py',
    'run_test3.py',
]

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)