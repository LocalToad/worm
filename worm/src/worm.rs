
use std::fmt::{Display, Formatter, Result as FmtResult};
use rand::{thread_rng, Rng};
use std::f64::consts::E;
#[derive(Debug)]
#[derive(Clone)]
pub enum Error{
    TypeErr(String),
    NoneErr(String),
    SizeErr(String),
    ElseErr,
}
impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match self {
            Error::TypeErr(s) => write!(f, "WTF IS THIS! THIS IS NOT WHAT I ASKED FOR! {}", s),
            Error::NoneErr(s) => write!(f, "We ain't found shit! {}", s),
            Error::SizeErr(s) => write!(f, "Size does matter and {}", s),
            Error::ElseErr => write!(f, "You managed to fuck something up so bad even i couldn't think of it, honestly props, take a pic and get this fixed"),
        }
    }
}
impl Clone for Layer {
    fn clone(&self) -> Layer {
        let mut new = Layer::new(self.inputs, self.outputs, self.function);
        new.weights = self.weights.clone();
        new.bias = self.bias.clone();
        new
    }
}
#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}
impl Matrix {
    /// Creates a new Matrix with the specified dimensions, initialized with zeros.
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Returns the number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Gets the value at a specific row and column.
    /// Returns `None` if the indices are out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Result<f64, Error> {
        if row < self.rows && col < self.cols {
            let index = row * self.cols + col;
            if self.data[index].is_nan() {
                Err(Error::NoneErr(format!("at row: {}, col: {}", row, col)))
            } else {
                Ok(self.data[index])
            }
        } else if row > self.rows {
            Err(Error::SizeErr(format!("{} is too big, there are only {} rows", row, self.rows)))
        } else if col > self.cols {
            Err(Error::SizeErr(format!("{} is too big, there are only {} columns", col, self.cols)))
        } else {
            Err(Error::ElseErr)
        }
    }

    /// Sets the value at a specific row and column.
    /// Returns `true` if the value was set successfully, `false` otherwise (e.g., out of bounds).
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> bool {
        if row < self.rows && col < self.cols {
            let index = row * self.cols + col;
            self.data[index] = value;
            true
        } else {
            false
        }
    }

    pub fn add(&mut self, matrix: Matrix) -> Result<Matrix, Error> {
        if matrix.rows == self.rows && matrix.cols == self.cols {
            let mut new_matrix = Matrix::new(matrix.rows, matrix.cols);
            for row in 0..matrix.rows {
                for col in 0..matrix.cols {
                    new_matrix.data[row * self.cols + col] = self.get(row, col)? + matrix.get(row, col)?;
                }
            }
            Ok(new_matrix)
        } else if matrix.rows != self.rows {
            Err(Error::SizeErr(format!("{} rows cant combine with {} rows", self.rows, matrix.rows)))
        } else if matrix.cols != self.cols {
            Err(Error::SizeErr(format!("{} cols cant combine with {} columns", self.cols, matrix.cols)))
        } else {
            Err(Error::ElseErr)
        }
    }

    pub fn sub(&mut self, matrix: Matrix) -> Result<Matrix, Error> {
        if matrix.rows == self.rows && matrix.cols == self.cols {
            let mut new_matrix = Matrix::new(matrix.rows , matrix.cols);
            for row in 0.. self.rows {
                for col in 0.. self.cols {
                    let value = self.get(row, col)? - matrix.get(row, col)?;
                    println!("{}", value);
                    new_matrix.set(row,col,value);
                }
            }
            Ok(new_matrix)
        } else if matrix.rows != self.rows {
            Err(Error::SizeErr(format!("{} rows cant combine with {} rows", self.rows, matrix.rows)))
        } else if matrix.cols != self.cols {
            Err(Error::SizeErr(format!("{} cols cant combine with {} columns", self.cols, matrix.cols)))
        } else {
            Err(Error::ElseErr)
        }
    }

    pub fn mul(&mut self, matrix: Matrix) -> Result<Matrix, Error> {
        if self.cols == matrix.rows {
            let mut new_matrix = Matrix::new(self.rows, matrix.cols);
            for row in 0.. new_matrix.rows {
                for col in 0.. new_matrix.cols {
                    for inner_col in 0.. self.cols {
                        let mut value: f64 = 0.0;
                        for inner_row in 0.. matrix.rows {
                            value += self.get(row, inner_col)? * matrix.get(inner_row, col)?;
                        }
                        new_matrix.set(row, col, value);
                    }
                }
            }
            Ok(new_matrix)
        } else if self.cols > matrix.rows {
            Err(Error::SizeErr(format!("{} columns are too big for {} rows", self.cols, matrix.rows)))
        } else if matrix.rows > self.cols{
            Err(Error::SizeErr(format!("{} rows are too big for {} columns", matrix.rows, self.cols)))
        } else {
            Err(Error::ElseErr)
        }
    }

    pub fn float_add(&mut self, n: f64) -> Result<Matrix, Error> {
        let mut new_matrix = Matrix::new(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_matrix.set(row,col,(self.get(row, col)?) + n);
            }
        }
        Ok(new_matrix)
    }

    pub fn float_sub(&mut self, n: f64) -> Result<Matrix,Error> {
        let mut new_matrix = Matrix::new(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_matrix.set(row,col,self.get(row, col)? - n);
            }
        }
        Ok(new_matrix)
    }
    pub fn float_mul(&mut self, n: f64) -> Result<Matrix, Error> {
        let mut new_matrix = Matrix::new(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_matrix.set(row,col,self.get(row, col)? * n);
            }
        }
        Ok(new_matrix)
    }

    pub fn inv_vec_add(&mut self, vec: Vec<f64>) -> Result<Matrix, Error> {
        let mut new_matrix = Matrix::new(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_matrix.set(row,col,self.get(row, col)? + vec[row]);
            }
        }
        Ok(new_matrix)
    }

    pub fn inv_vec_mul(&mut self, vec: Vec<f64>) -> Result<Matrix, Error> {
        let mut new_matrix = Matrix::new(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_matrix.set(row,col,self.get(row, col)? * vec[col]);
            }
        }
        Ok(new_matrix)
    }

    pub fn clone(&self) -> Matrix {
        let mut new_matrix = Matrix::new(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_matrix.set(row,col,self.get(row, col).unwrap().clone());
            }
        }
        new_matrix
    }
}
pub struct Layer {
    inputs: usize,
    outputs: usize,
    function:i8, //0 for Relu, 1 for Leaky Relu, 2 for Sigmoid, 3 for Tanh, 4 for Softmax
    weights: Matrix,
    bias: Matrix //i know you think this will be better as a vec, it wont,
    // i have followed that thought train for hours and it always results in this needing to be a vec
}
impl Layer {
    fn new(inputs:usize, outputs:usize, function:i8) -> Layer {
        let weights = Matrix::new(inputs, outputs);//i know the internet says (output,input)
        //we are keeping this matrix in its inverted form bc fuck writing a method that inverts a matrix
        let bias = Matrix::new(1, outputs); //same for this fucker bc it make the shit work
        Layer {inputs, outputs, function, weights, bias}
    }
    fn randomize(&mut self) -> bool {
        for output in 0..self.outputs {
            for input in 0..self.inputs {
                let mut n = thread_rng();
                self.weights.set(output, input, n.r#gen());
            }
            let mut n = thread_rng();
            self.bias.set(output, 0, n.r#gen());
        }
        true
    }
    fn forward_prop(&self, input: Matrix) -> Result<Matrix, Error> {
        let mut output = Matrix::new(1, self.outputs);
        output.add(input.clone())?.mul(self.weights.clone())?.add(self.bias.clone())?;
        Ok(output)
    }
}
struct Model {
    inputs: usize,
    outputs: usize,
    layers: usize,
    input: Layer,
    hidden: Vec<Layer>,
    output: Layer,
    loss: i8,//0 for Mean Squared Error (MSE), 1 for Mean Absolute Error (MAE),
    //2 for Mean Absolute Percentage Error (MAPE), 3 for Binary Cross-Entropy Loss (BCEL),
    //4 for Categorical Cross-Entropy Loss (CCEL)
    optimizer: i8,//0 for Gradient Decent (GD), 1 for Stochastic Gradient Decent (SGD),
    //2 for Adaptive Gradient Algorithm (Adagrad), 3 for Root Mean Square Propogation (RMSprop)
    //4 for Adaptive Delta (AD), 5 for Adaptive Moment Estimation (Adam), 6 for Adamax,
    //7 for Nesterov Accelerated Gradient (NAG), 8 for Nesterov Adam (Nadam)
}
impl Model {
    fn think(self, input: Matrix) -> Result<Model, Error> {
        let mut output = Matrix::new(self.outputs, 1);
        output = self.output.forward_prop(self.black_box(self.input.forward_prop(input)?))?;
    }
    pub fn black_box(self, input: Matrix) -> Result<Matrix, Error> {
        let mut output = Matrix::new(input.rows, input.cols);
        for layer in 0..self.layers-3 {
            if layer = 0 {
                output = self.hidden[layer].forward_prop(input.clone())?;
            } else {
                output = self.hidden[layer].forward_prop(output)?;
            }
        }
        Ok(output)
    }
    pub fn train(&mut self, real: Matrix, guess: Matrix) -> bool {

    }
    //0 MSE Error = SUM{(guess - real)^2}/n
    //1 MAE Error = SUM{|guess - real|}/n
    //2 MAPE Error = (|guess - real| / real) * 100%
    //3 BCEL Loss = real * log(guess) + (1 - real) * log(1 - guess)
    //4 CCEL Loss = - SUM{real[i] * log(guess[i])} ONLY WORKS WITH SOFTMAX ACTIVATION

    //DERIVATIVES OF ABOVE FUNCTIONS
    //0 MSE (2/n) * (guess - real)
    //1 MAE 1 if guess > real, 0 if real > guess
    //2 MAPE 1 / (n * guess) if guess > real, -1 / (n * guess) if real > guess
    //3 BCEL -real / guess + (1 - real) / (1 - guess)
    //4 CCEL guess - real

    // a = alpha = learning rate(0 < float < 1 - closer to 1 = overstepping, closer to 0 = takes too long, commonly 0.001),
    // E = epsilon = divide by 0 prevention(positive float near 0, commonly 10^-8),
    // b = beta = decay(commonly 0.9 or 0.999)
    // g = Gradient
    // e = Squared Decaying Average
    // d = Parameter Update Magnitude
    // u = Infinity Norm
    // L = Lookahead

    //0 GD i(t) = i(t-1) - (a * g) Once per Epoch
    //1 SGD i(t) = i(t-1) - (a * g) Once per Step in Epoch
    //2 Adagrad i(t) = i(t-1) - (a / (sqrt{G[i]} + E)) * g(i(t-1))
        // G[i] += g(i)^2 every Epoch
    //3 RMSprop i(t) = i(t-1) - (a * g[i][t-1] / (sqrt{e_g[i]} + E)
        //e_g[i][t] = b * SGA[i][t-1] + 1 - b * g[i]^2
    //4 AD i[t] = i[t-1] + d[i]
        //e_g[i][t] = b * SGA[i][t-1] + 1 - b * g[i]^2
        //d[i][t] = d[i][t-1] - (sqrt{e_d[i]}[t-1] / sqrt{e_g}[t] * e[i]
        //e_d[i][t+1] = b * e_d[i][t] + (1 - b) * d[i][t]^2
    //5 Adam i[t] = i[t-1] - a * (m / (sqrt{v} + E))
        //b1 often 0.9 and b2 often 0.999
        //e_g1[i][t] = b1 * e_g1[i][t-1] + (1 - b1) * g[i]
        //e_g2[i][t] = b2 * e_g2[i][t-1] + (1 - b2) * g[i]^2
        //bias correction bc the starting momentum is assumed 0(which is not true)
            //corrected e_g1 = m = e_g1[i][t] / (1 - b1^t)
            //corrected e_g2 = v = e_g2[i][t] / (1 - b2^t)
    //6 Adamax i[t] = i[t-1] - (a * m[i][t]) / (u[i][t] + E)
        //e_g1[i][t] = b1 * e_g1[i][t-1] + (1 - b1) * g[i]
        //u[i][t] = max{(b2 * u[i][t-1]), abs{g}}
        //m[i][t] = e_g1[i][t] / (1 - b1^t)
    //7 NAG i[t] = L_i - (a * L_d[i])
        //L_i= i[t] - (b * d[i][t-1])
        //L_g[i] = L_i - (b * d[i][t])
        //L_d[i] = (b * d[i][t]) + L_g[i]
    //8 Nadam i[t] = L_i - a * ((b1 * m[i][t-1]) + ((1 - b1) * L_g[i] / (1 - b1^t)) / (sqrt{v[i][t-1]} + E))
        //L_i= i[t] - (b * d[i][t-1])
        //L_g[i] = L_i - (b * d[i][t])
        //e_g1[i][t] = b1 * e_g1[i][t-1] + (1 - b1) * g
        //e_g2[i][t] = b2 * e_g2[i][t-1] + (1 - b2) * g[i]^2
        //m[i][t] = e_g1[i][t] / (1 - b1^t)
        //u[i][t] = max{(b2 * u[i][t-1]), abs{g}}

    //please let all this hard work be useful my brain is frying
}
pub struct ModelBuilder {
    inputs: usize,
    outputs: usize,
    input: Option<Layer>,
    hidden: Vec<Layer>,
    output: Option<Layer>,
}
impl ModelBuilder {
    pub fn new(inputs:usize, outputs:usize) -> Self {
        Self {inputs, outputs, input: None, hidden: vec![], output: None}
    }
    pub fn input(mut self, outputs:usize, function:i8) -> Self {
        let inputs = self.inputs;
        self.input = Some(Layer::new(inputs, outputs, function));
        self
    }
    pub fn hidden(mut self, outputs:usize, function:i8) -> Self {
        let mut inputs: usize = 0;
        if self.hidden.len() == 0 {
            inputs = self.input.clone().unwrap().outputs
        } else {
            inputs = self.hidden[self.hidden.len()-1].outputs;
        }
        self.hidden.push(Layer::new(inputs, outputs, function));
        self
    }
    pub fn output(mut self, outputs: usize, function:i8) -> Self {
        let inputs = self.hidden[self.hidden.len()-1].outputs;
        self.output = Some(Layer::new(inputs, outputs, function));
        self
    }

    pub fn build(self, model_type: ModelType, loss: i8) -> Model {

        Model {
            inputs: self.inputs,
            outputs: self.outputs,
            layers: self.hidden.len()+2,
            input: self.input.unwrap(),
            hidden: self.hidden,
            output: self.output.unwrap(),
            loss,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_row() {
        let matrix = Matrix::new(2, 3);
        assert_eq!(matrix.rows, 2);
    }

    #[test]
    fn test_matrix_col() {
        let matrix = Matrix::new(2, 3);
        assert_eq!(matrix.cols, 3);
    }

    #[test]
    fn test_matrix_set_set() {
        let mut matrix = Matrix::new(2, 2);
        assert_eq!(matrix.set(0,0, 1.0), true);
    }

    #[test]
    fn test_matrix_get() -> Result<(), Error> {
        let mut matrix = Matrix::new(2, 2);
        matrix.set(0,0, 1.0);
        assert_eq!(matrix.get(0,0)?, 1.0);
        Ok(())
    }

    #[test]
    fn test_matrix_add() -> Result<(), Error> {
        let mut ans = Matrix::new(2, 2);
        ans.set(0,0, 5.0);
        ans.set(1,0,5.0);
        ans.set(0,1,5.0);
        ans.set(1,1,5.0);

        let mut matrix_0 = Matrix::new(2, 2);
        matrix_0.set(0,0, 1.0);
        matrix_0.set(0,1, 2.0);
        matrix_0.set(1,0, 3.0);
        matrix_0.set(1,1, 4.0);

        let mut matrix_1 = Matrix::new(2, 2);
        matrix_1.set(0,0, 4.0);
        matrix_1.set(0,1, 3.0);
        matrix_1.set(1,0, 2.0);
        matrix_1.set(1,1, 1.0);

        let result = matrix_0.add(matrix_1)?;
        assert_eq!(result, ans);
        Ok(())
    }

    #[test]
    fn test_matrix_sub() -> Result<(), Error> {
        let mut ans = Matrix::new(2, 2);
        ans.set(0,0, 1.0);
        ans.set(0,1, 1.0);
        ans.set(1,0, 1.0);
        ans.set(1,1, 1.0);

        let mut matrix_0 = Matrix::new(2, 2);
        matrix_0.set(0,0, 2.0);
        matrix_0.set(0,1, 3.0);
        matrix_0.set(1,0, 4.0);
        matrix_0.set(1,1, 5.0);

        let mut matrix_1 = Matrix::new(2, 2);
        matrix_1.set(0,0, 1.0);
        matrix_1.set(0,1, 2.0);
        matrix_1.set(1,0, 3.0);
        matrix_1.set(1,1, 4.0);

        let result = matrix_0.sub(matrix_1)?;
        assert_eq!(result, ans);
        Ok(())
    }

    #[test]
    fn test_matrix_mul() -> Result<(), Error> {
        let mut ans = Matrix::new(2, 2);
        ans.set(0,0, 8.0);
        ans.set(0,1, 12.0);
        ans.set(1,0, 8.0);
        ans.set(1,1, 12.0);

        let mut matrix_0 = Matrix::new(2, 2);
        matrix_0.set(0,0, 2.0);
        matrix_0.set(0,1, 2.0);
        matrix_0.set(1,0, 2.0);
        matrix_0.set(1,1, 2.0);

        let mut matrix_1 = Matrix::new(2, 2);
        matrix_1.set(0,0, 1.0);
        matrix_1.set(0,1, 2.0);
        matrix_1.set(1,0, 3.0);
        matrix_1.set(1,1, 4.0);

        let result = matrix_0.mul(matrix_1)?;
        assert_eq!(result, ans);
        Ok(())
    }

    #[test]
    fn test_matrix_float_add() -> Result<(), Error> {
        let mut ans = Matrix::new(2, 2);
        ans.set(0, 0, 2.0);
        ans.set(0, 1, 3.0);
        ans.set(1, 0, 4.0);
        ans.set(1, 1, 5.0);

        let mut matrix = Matrix::new(2, 2);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);

        let n = 1.0;
        assert_eq!(matrix.float_add(n)?, ans);
        Ok(())
    }

    #[test]
    fn test_matrix_float_sub() -> Result<(), Error> {
        let mut ans = Matrix::new(2, 2);
        ans.set(0,0, 1.0);
        ans.set(0,1, 2.0);
        ans.set(1,0, 3.0);
        ans.set(1,1, 4.0);

        let mut matrix = Matrix::new(2, 2);
        matrix.set(0,0, 2.0);
        matrix.set(0,1, 3.0);
        matrix.set(1,0, 4.0);
        matrix.set(1,1, 5.0);

        let n: f64 = 1.0;
        assert_eq!(matrix.float_sub(n)?, ans);
        Ok(())
    }

    #[test]
    fn test_matrix_float_mul() -> Result<(), Error> {
        let mut ans = Matrix::new(2, 2);
        ans.set(0, 0, 2.0);
        ans.set(0, 1, 4.0);
        ans.set(1, 0, 6.0);
        ans.set(1, 1, 8.0);

        let mut matrix = Matrix::new(2, 2);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);

        let n = 2.0;
        assert_eq!(matrix.float_mul(n)?, ans);
        Ok(())
    }
}
