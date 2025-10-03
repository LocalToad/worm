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

#[derive(Debug)]
#[derive(PartialEq)]
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
    pub fn float_mul(&mut self, n: f64) -> Result<Matrix, Error> {
        let mut new_matrix = Matrix::new(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_matrix.set(row,col,self.get(row, col)? * n);
            }
        }
        Ok(new_matrix)
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
        ans.set(1,0, 1.0);
        ans.set(0,1, 1.0);
        ans.set(1,1, 1.0);

        let mut matrix_0 = Matrix::new(2, 2);
        matrix_0.set(0,0, 2.0);
        matrix_0.set(0,1, 3.0);
        matrix_0.set(1,0, 4.0);
        matrix_0.set(1,1, 5.0);

        let mut matrix_1 = Matrix::new(2, 2);
        matrix_1.set(0,0, 1.0);
        matrix_1.set(1,0, 2.0);
        matrix_1.set(0,1, 3.0);
        matrix_1.set(1,0, 4.0);

        let result = matrix_0.sub(matrix_1)?;
        assert_eq!(result, ans);
        Ok(())
    }

    #[test]
    fn test_matrix_mul() -> Result<(), Error> {
        let mut ans = Matrix::new(2, 2);
        ans.set(0,0, 2.0);
        ans.set(1,0, 4.0);
        ans.set(0,1, 6.0);
        ans.set(1,1, 8.0);

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
        assert_eq!(ans.float_add(n)?, ans);
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
        assert_eq!(ans.float_sub(n)?, ans);
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
        assert_eq!(ans.float_mul(n)?, ans);
        Ok(())
    }
}
