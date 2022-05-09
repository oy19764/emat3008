import numpy as np


def test_inputs(input, input_name, test_name, ):
    """
    Testing function to check function inputs are valid.
        Parameters:
            input:          input being tested
            input_name:     name of input being tested
            test_name:      test to check input with
    """
    # define tests
    def test_int_or_float(input, input_name):
        if not isinstance(input, (int, float, np.int_, np.float_)):
            raise TypeError(f"{input_name}: {input} is not a valid type. \n" 
                            "Please input an integer or a float")
    

    def test_function(input, input_name):
        if not callable(input):
            raise TypeError(f"{input_name} is not a valid input. \n" 
                            "Please input a function")
    
    
    def test_array(input, input_name):
        if not isinstance(input, (np.ndarray, list)):
            raise TypeError(f"{input_name} is not a valid type. \n" 
                            "Please input a numpy array")


    def test_tuple(input, input_name):
        if not isinstance(input, tuple):
            raise TypeError(f"{input_name}: {input} is not a valid type. \n" 
                            "Please input a tuple")


    def test_int_or_tuple(input, input_name):
        if not isinstance(input, (tuple, int, float)):
            raise TypeError(f"{input_name}: {input} is not a valid type. \n" 
                            "Please input a tuple, integer or float")


    def test_bool(input, input_name):
        if not isinstance(input, bool):
            raise TypeError(f"{input_name}: {input} is not a valid type. \n" 
                            "Please input a boolean ('True' or 'False')")
   
    # call test to perform
    if test_name == 'test_int_or_float':
        test_int_or_float(input, input_name)

    if test_name == 'test_function':
        test_function(input, input_name)

    if test_name == 'test_array':
        test_array(input, input_name)

    if test_name == 'test_tuple':
        test_tuple(input, input_name)

    if test_name == 'test_bool':
        test_bool(input, input_name)

    if test_name == 'test_int_or_tuple':
        test_int_or_tuple(input, input_name)




def test_ode(f, x0, *args):
    """
    Test if valid numerical function is being passed
        Parameters:
                f:          Function to be checked
                x0:         initial values passed
                *args:      any additional arguments f requires
    """

    if callable(f):
        # test f returns valid output
        t_test = 1  # time value to test ode at
        test_output = f(t_test, x0, *args)
        # test valid output type
        if isinstance(test_output, (int, float, np.int_, np.float_, list, np.ndarray)):
            # test valid output size
            if np.array(test_output).shape == np.array(x0).shape:
               pass
            else:
                raise ValueError(f"Invalid output shape from ODE function {f},\n"
                                "x0 and {f} output should have the same shape")
        else:
            raise TypeError(f"Output from ODE function is a {type(test_output)},\n"
            "output should be an integer, float, list or array")
    else: 
        raise TypeError(f"{f} is not a valid input type. \n" 
                            "Please input a function")
