# Python program to display all the prime numbers within an interval

# change the values of lower and upper for a different result
lower = int(input("write a number"))
upper = int(input("write a number"))

numbers = []
# uncomment the following lines to take input from the user
#lower = int(input("Enter lower range: "))
#upper = int(input("Enter upper range: "))

print("Prime numbers between",lower,"and",upper,"are:")

for num in range(lower,upper + 1):
   # prime numbers are greater than 1
   if num > 1:
       for i in range(2,num):
           if (num % i) == 0:
               break
       else:
           numbers.append(num)
           #print("numbers:",num)

print("amount:",len(numbers))
