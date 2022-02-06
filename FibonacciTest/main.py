# This is a Fibonacci Test Project for Dr. Nguyen.

class FibonacciTest:

    file1 = open("Output.txt", "w")

    while True:
        try:
            n = int(input("Please enter the number of Fibonacci numbers you want to see: "))
            break
        except:
            print("Invalid input! Next time, enter an integer!")

    n1 = 0
    n2 = 1
    count = 0

    if n <= 0:
        file1.write("You get nothing! Next time, enter an integer greater than 0!")
    elif n == 1:
        file1.write("The first number of the Fibonacci Sequence is: ")
        file1.write("\n" + str(n1))
    else:
        file1.write("The Fibonacci Sequence up to " + str(n) + ":")
        while count < n:
            print(str(n1))
            file1.write("\n" + str(n1))
            n3 = n1 + n2
            n1 = n2
            n2 = n3
            count += 1

