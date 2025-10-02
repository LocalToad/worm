use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug)]
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
            let new_matrix = Matrix::new(matrix.rows, matrix.cols);
            for row in 0..matrix.rows {
                for col in 0..matrix.cols {
                    self.data[row * self.cols + col] = self.get(row, col)? + matrix.get(row, col)?;
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
            for row in 0.. matrix.rows {
                for col in 0..matrix.cols {
                    new_matrix.data[row * self.cols + col] = self.get(row, col)? - matrix.get(row, col)?;
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
                    let mut value: f64 = 0.0;
                    for inner_col in 0.. self.cols {
                        for inner_row in 0.. matrix.rows {
                            value += self.get(row, inner_col)? * matrix.get(inner_row, col)?;
                        }
                    }
                    new_matrix.set(row, col, value);
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
    pub fn float_mult(&mut self, n: f64) -> Result<Matrix, Error> {
        let mut new_matrix = Matrix::new(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_matrix.set(row,col,self.get(row, col)? * n);
            }
        }
        Ok(new_matrix)
    }
}

