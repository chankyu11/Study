weight = 0.5
input = 0.5
goal_prediction = 0.8

lr = 0.01

for iteration in range(111):
    prediction = input * weight
    # prediction는 0.5 * 0.5
    error = (prediction - goal_prediction) **2
    # error는 (0.25 - 0.8) ** 2 = 0.3025
    print("Error: " + str(error) + "\tPrediction: " + str(prediction))

    up_prediction = input * (weight + lr)
    # up_prediction는 0.5 * (0.5 + 0.001) = 0.2505
    up_error = (goal_prediction - up_prediction) ** 2
    # up_error는 (0.8 - 0.2505) ** 2  = 0.30195025

    down_prediction = input * (weight - lr)
    # down_prediction = 0.5 * (0.5 - 0.001) = 0.2495
    down_error = (goal_prediction - down_prediction) ** 2
    # down_error = (0.8 - 0.2495) ** 2  = 0.30305025
    print("weight1:", weight)
    print("횟수:", iteration)
    if(down_error < up_error) :
        weight = weight - lr
    if(down_error > up_error) :
        weight = weight + lr
    print("weight2:", weight)
