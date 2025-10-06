/*

    DISCLAMER: None of this applies to Genetic Algorithms or anything else, this is Backpropagation Neural Networks

    The most basic form of a neural network starts here with this formula:

                            y = (m * x) + b

    Lets change the variables into something that makes more sense for the terms of machine learning

    y = z = output neuron <- (neuron is a very bad analogy but this is what the community uses)
    x = i = input neuron  <- (these are just numbers honestly you will see)
    m = w = weight
    b = b = bias

    z = (w * i) + b

    I want to make this guess if the next number in a series will go up or down.
    I want to preface that what im making here will not do this well if at all.
    This is purely an example of a 2-dimensional Neural Network.(w,b)

    To start we need to randomly generate weights and biases for the program.
    This is the birthing proccess for a neural net and the reason that most people dog on them.

    I will try to make this follow along-able.(i dont english)

    We start by choosing a number that meets this criteria for the weight: 0 < w < 1
        I will pick w = 0.5.
    Next we do the same for the bias: 0 < b < 1
        I will pick b = 0.12

    My network currently looks like this.
    z = (0.5 * i) + 0.12
    This is what people often call the 'brain' of a neural net.
    This may not look complicated now but it will be soon.

    Now lets define a data set. This is the 'environment' that the neural net is in.
    Im going to randomly generate numbers 1-20 in a list of 20 numbers.
    [8, 12, 9, 6, 14, 17, 7, 6, 18, 20, 8, 20, 6, 17, 4, 11, 15, 9, 13, 5]
    Feel free to use my data set.
    If you make your own it can be any size with numbers of any range.
    Just make sure the next number in the set changes, we dont exactly have the complexity in this
        network to handle more than 2 choices.

    Now you may be asking.
    How will we know if the neural net predicts the next number to be higher or lower?
    Right now if were to run it for the first peice of data we would get this:
        z = (0.5 * i) + 0.12
        i = 8
        z = (0.5 * 8) + 0.12
        z = 4 + 0.12
        z = 4.12
        This doesn't say up or down?

    This is where activation functions come in.
    For picking a binary choice of 1 or 0 we use the Sigmoid activation function.
        Sigmoid:
            E: Euler's number (aprox. 2.718)
            a: Activated Output
            a = 1 / (1 + E^-z)
    This is designed to give us a number closer to 1 as o reaches +Inf and a number closer to
        0 as o reaches -Inf.

    Using this for our z = 4.12:
        a = 1 / (1 + E^-z)
        a = 1 / (1 + 2.718^-4.12)
        a = 1 / (1 + 0.016)
        a = 1 / 1.016
        a = 0.98
    So this 0.98 denotes our Neural Nets choice.
    For this example lets say 1 = up and 0 = down.
    In this case because 0.98 is closer to 1 than 0 we will say the Neural Net chooses up.
    Let's check our data set to see what the real answer was.

    The second item in our data set is 12 which is higher than 8 (the first item in the data set).
    Meaning the correct answer was 1.
        Now that we now what the correct answer is we need to rename some variables
        Real Answer = z
        Guess = z_hat
    So now we employ a Loss function to see how well the Networks guess was.
    For binary type outputs we use the BCEL function.
        BCEL: Binary Cross-Entropy Loss
        L: Loss
        L = z * log(z_hat) + (1 - z) * log(1 - z_hat)
        L = 1 * log(0.98) + (1-1) * log(1-0.98)
        L = 2 * -0.009 + 0 * log(0.02)
        L = -0.017 + 0
        L = -0.017
    This tells us how wrong the Neural Net was.

    Now we need to use this loss number to train the neural net with.
    For this we use Optimization Functions. (this process is referred to as 'learning' a lot)
    The most basic for is Gradient Descent.
    We are going to use a version of gradient decent that optimizes every data point instead of
        every Epoch (full run of a data set).

    Stochastic Gradient Descent
        p: A given Parameter in the system(w or b)
        alpha: This is the learning rate. This is a hyperparameter. It's generally set to 0.001
            This is also known as the 'step-size'
        g: Gradient of the slope of the loss function given the coordinates L(w, b, i) - modifying w and b so when i is entered L is always as low as possible
            The technical mumbo-jumbo is:
                The gradient is a partial derivative of the loss function with respect to each
                parameter.
        p(new) = p(current) - (alpha * g)
        p(t+1) = p(t) - (0.001 * g)

    We will need to calculate the gradient of for each parameter.
        This determines how much a small change in each parameter will affect loss.
        This is represented as a vector that points to the steepest increase in loss.

    For this example we would use the derivative of the BCEL function.
        derivative BCEL g = -z / z_hat + (1 - z) / (1 - z_hat)
    We will also need the derivative of the Sigmoid function to continue backwords through the
        function.
        derivative Sigmoid = z_hat * (1 - z_hat)
    The partial derivative of our function
        z_hat = (w * i) + b
        with respect to w is
            w
        with respect to b is
            1

    Now lets do the make to solve for g_w and g_b.
        g_w: Partial Gradient of the Loss function with respect to w
        g_b: Partial Gradient of the Loss function with resepct to b

        g_w = -z / z_hat + (1 - z) / (1 - z_hat)
        g_b = -z / z_hat + (1 - z) / (1 - z_hat)

        g_w = -z / (z_hat * (1 - z_hat)) + (1 - z) / (1 - (z_hat * (1 - z_hat)))

        g_w = -z / (w * (1 - w))z_hat + (1 - z) / (1 - (w * (1 - w)))z_hat


        g_w = -1 / (.5 * (1 - .5)) + (1 - 1) / (1 - (.5 * (1 - .5)))
        g_w = -1 / (.5 * .5 ) + 0 / (1 - (.5 * .5 ))
        g_w = -1 / .25 + 0 / (1 - .25 )
        g_w = -4 + 0 / .75
        g_w = -4 + 0
        g_w = -4

        1
        g_b = -z / z_hat + (1 - z) / (1 - z_hat)
        g_b = -1 / 0.98 + (1 -1) / (1 - 0.98)
        g_b = -1.020 + 0 / 0.02
        g_b = -1.02 + 0
        g_b = -1.02

   Now we can use these gradients of our parameters in our optimizer function to update the parameters.
        For w:
            w(t+1) = w(t) - (0.001 * g_w)
            w(t+1) = .5 - (0.001 * -4)
            w(t+1) = .5 - (-0.004)
            w(t+1) = .5 + 0.004
            w(t+1) = .504
        For b:
            b(t+1) = b(t) - (0.001 * g_b)
            b(t+1) = .12 - (0.001 * -1.02)
            b(t+1) =  .12 - (-0.001)
            b(t+1) = .12 + .001
            b(t+1) = .121
    This now gives us this updated Neural Net:
        z = (.504 * i) + .121
    Now we would continue this for every data point in the data set.
    This will slowly try to converge onto a function that for any given input it can guess if the
        next number will be higher and lower.(again it will never be able to do this because our
        neural net is too small, but this is the concept)
    Now...because i hate myself this is this "AI" training on the data set and i will see how well
        it does at the end.
    [8., 12., 9., 6., 14., 17., 7., 6., 18., 20., 8., 20., 6., 17., 4., 11, 15, 9, 13, 5]

    w = 0.504
    b = 0.121
    input = 12
    z = (0.504 * 12) + 0.121
    z = 6.169
    guess = 0.998 UP
    answer = 0 DOWN
    Loss = -2.69
    Average Decaying Loss(ADL) = b(ADL(t) + (1-b)(Loss)
    ADL = -0.284

    w = 0.503
    b = 0.120
    input = 9
    z = (0.503 * 9) + 0.120
    z = 4.647
    guess = 0.990 UP
    answer = 0 DOWN
    Loss = -2
    ADL = -0.456

    w = 0.502
    b = 0.020
    input = 6
    z = (0.502 * 6) + 0.020
    z = 3.032
    guess = 0.954 UP
    answer = 1 UP
    Loss = -0.020
    ADL = -0.412

    w = 0.506
    b = 0.021
    input = 14
    z = (0.506 * 14) + 0.021
    z = 7.105
    guess = 0.999 UP
    answer = 1 UP
    Loss = -0.001
    ADL -0.371

    w = 0.510
    b = 0.022
    input 17
    z = (0.510 * 17) + 0.022
    z = 8.692
    guess = 0.9999 UP
    answer = 0 DOWN
    Loss = -4
    ADL = -0.734

    w = 0.509
    b = -10.022
    input = 7
    z = (0.509 * 7) - 10.022
    z = -6.459
    guess = 0.002 DOWN
    answer = 0 DOWN
    Loss = -0.001
    ADL = -0.661

    w = 0.508
    b = -10.021
    input = 6
    z = (0.508 * 6) - 10.021
    z = -6.973
    guess = 0.001 DOWN
    answer = 1 UP
    Loss = -3
    ADL = -0.895

    w = 0.512
    b = -9.021
    input = 18
    z = (0.512 * 18) - 9.021
    z = 0.195 <-notice how this is close to 0,
    this is equivalent to the model being unsure of the answer in this instance
    guess = 0.549 UP (barely)
    answer = 1 UP
    Loss = -0.260
    ADL = -0.832

    w = 0.516
    b = -9.018
    input = 20
    z = (0.516 * 20) - 9.018
    z = 1.302
    guess = 0.786 UP
    answer = 0 DOWN
    Loss = -0.670
    ADL = -0.816

    w = 0.515
    b = -9.013
    input = 8
    z = (0.515 * 8) - 9.013
    z = -4.893
    guess = 0.007 DOWN
    answer = 1 UP
    Loss = -2.155
    ADL = -0.947

    w = 0.519
    b = -8.510
    input = 20
    z = (0.519 * 20) - 8.51
    z = 1.87
    guess = 0.867 UP
    answer = 0 DOWN
    Loss = -0.876
    ADL = -0.940

    w = 0.518
    b = -8.502
    input = 6
    z = (0.518 * 6) - 8.502
    z = -5.394
    guess = 0.005 DOWN
    answer = 1 UP
    Loss = -2.301
    ADL = -1.076

    w = 0.522
    b = -8.302
    input = 17
    z = (0.522 * 17) - 8.302
    z = 0.572
    guess = 0.639 UP
    amswer = 0 DOWN
    Loss = -0.442
    ADL = -1.012

    w = 0.521
    b = -8.299
    input = 4
    z = (0.521 * 4) - 8.299
    z = -6.215
    guess = 0.002 DOWN
    answer = 1 UP
    Loss = -2.699
    ADL = -1.181

    w = 0.525
    b = -7.899

    i give up lol

    Anyway, i probably messed up some numbers somewhere but this is the general idea.
    You can give a system more dimensions by increaseing the amount of Neurons in the system.
                     [w1, w2, w3, w4,
    [z1,z2,z3,z4] = (w5, w6, w7, w8,   * [i1, i2, i3] ) + [b1, b2, b3, b4]
                     w9, w10, w11, w12]
    [a1, a2, a3, a4] = [z1. z2. z3. z4] ACTIVATION FUNCTION

    This is an example if a Neural Net that has 16-Dimensions to its plane
    This Neural New takes in 3 different Inputs.
    It also has 4 different Outpus.

    Another way to scale a Neural Net is by making it 'deeper'.
    This is done by adding more hidden layers between the input and the output.

    Layer 1:
    [za1,za2,za3]=[wa1, wa2, wa3,  * [i1, i2] + [ba1, ba2, ba3]
                   wa4, wa5, wa6]
    [aa1, aa2, aa3] = [za1, za2, za3] ACTIVATION FUNCTION
    Layer 2:[wb1,
    [zb1]=   wb2, * [aa1, aa2, aa3] + [bb1]
             wb3]
    [ab1] = [zb1] ACTIVATION FUNCTION

    This is an example of a Neural Network with 1 hidden layer.
    It has 2 input Neurons: [i1, i2]
    3 hidden Neurons: [za1, za2, za3]
    1 output Neuron: [zb1]
    The plane that this produces has 24-Dimensions


    You can mix and match the types of Activation functions, Loss Functions, and Optimization
        functions to produce differnt kinds of Neural Networks.
        Ex.)
            Actor/Critic(Stocks)
            PPO(^similar)
            LSTM(ChatGPT, Grok, LLMS, Stock price predictors)
            NPM(summarizes text into a number)
            lots more, google it, there are litteral tons.

    In reality the model can only do what you give it the ability to do.
    It can only 'see' what you give it the ability to 'see'
    When you run into a problem of a local minimum(the loss is stuck at a certain point and wont
        get lower, the network can do nothing but guess)
    This happens when the neural network doesn't have enough dimensions to approximate the function.
    In the example of the very bad 2d neural network, the only way for it to lower its loss in
        any meaningful way is if it can predict the randomizer function that google uses.
        That would take TONS of dimensions for a neural network to reliable stumble onto the
            right answer.


 */