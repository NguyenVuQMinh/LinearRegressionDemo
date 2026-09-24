def gradient(x):
    return 2*x-4

def cost(x):
    return (x)**2 - 4*x + 5

def GD(x0, learning_rate, iterations):
    x = [x0]
    for i in range(iterations):
        x_new = x[-1] - learning_rate * gradient(x[-1])
        x.append(x_new)
        print("Iteration %d: x = %.4f, cost = %.4f" % (i+1, x_new, cost(x_new)))
    return x

###
iterations = 10
learning_rate = 0.2

x0 = 5
x1 = -5
x = GD(x0, learning_rate, iterations)
print("Final result: x = %.4f, cost = %.4f" % (x[-1], cost(x[-1])))
x = GD(x1, learning_rate, iterations)
print("Final result: x = %.4f, cost = %.4f" % (x[-1], cost(x[-1]))) 