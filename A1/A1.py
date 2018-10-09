import Exercise1 
import Exercise2

promptString = """
Welcome to Boshen Cui (20613736)'s submission for CS480 A1.
To run a particular question, simply type in the question id and hit enter.
Instructions for running each question, and where output will be located can also be found in the writeup pdf (A1_writeup.pdf).
Valid question ids are:
E1Q1
E1Q2
E1Q4
E2Q1
E2Q2
E2Q3
E2Q4
E2Q5
all
exit
Please ensure that you are running on a Python 2.7ish interpreter, and that you have numpy, matplotlib, and sklearn installed.
"""

cmd = ""

while cmd != "exit":
	cmd = raw_input(promptString)
	if cmd == "E1Q1" or cmd == "all":
		Exercise1.Exercise1P1()
		print("E1Q1 Done!")
	if cmd == "E1Q2" or cmd == "all":
		Exercise1.Exercise1P2()
		print("E1Q2 Done!")
	if cmd == "E1Q4" or cmd == "all":
		Exercise1.Exercise1P4()
		print("E1Q4 Done!")

	if cmd == "E2Q1" or cmd == "all":
		Exercise2.Exercise2P1()
		print("E2Q1 Done!")
	if cmd == "E2Q2" or cmd == "all":
		Exercise2.Exercise2P2()
		print("E2Q2 Done!")
	if cmd == "E2Q3" or cmd == "all":
		Exercise2.Exercise2P3()
		print("E2Q3 Done!")
	if cmd == "E2Q4" or cmd == "all":
		Exercise2.Exercise2P4()
		print("E2Q4 Done!")
	if cmd == "E2Q5" or cmd == "all":
		Exercise2.Exercise2P5()
		print("E2Q5 Done!")