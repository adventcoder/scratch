module Enumerable
  def mean
    acc = 0.0
    each_with_index { |x, i| acc += (x - acc) / (i + 1) }
    acc
  end
end

class Array
  def flat?
    none? { |x| Array === x }
  end

  def dot(other)
    if flat?
      if other.flat?
        vector_dot_vector(other)
      else
        vector_dot_matrix(other)
      end
    else
      if other.flat?
        map { |row| row.vector_dot_vector(other) }
      else
        map { |row| row.vector_dot_matrix(other) }
      end
    end
  end

  def vector_dot_matrix(other)
    other.map { |col| vector_dot_vector(col) }
  end

  def vector_dot_vector(other)
    zip(other).inject(0) { |sum, (x, y)| sum + x*y }
  end

  def invert!
    unless all? { |row| row.size == size }
      raise ArgumentError, 'not square matrix'
    end
    for n in 0 ... size
      if self[n][n] == 0
        i = (n + 1 ... size).find { |i2| self[i2][n] != 0 }
        raise ZeroDivisionError, 'singular matrix' if i.nil?
        self[n], self[i] = self[i], self[n]
      end
      unless self[n][n] == 1
        m = self[n][n]
        self[n][n] = 1
        for j in 0 ... size
          self[n][j] = self[n][j].quo(m)
        end
      end
      for i in 0 ... size
        next if i == n
        unless self[i][n] == 0
          m = self[i][n]
          self[i][n] = 0
          for j in 0 ... size
            self[i][j] -= self[n][j] * m
          end
        end
      end
    end
    self
  end
end

def fit(xcols, ycol)
  xbars = xcols.map { |xs| xs.mean }
  ybar = ycol.mean
  xvars = xcols.zip(xbars).map { |(xcol, xbar)| xcol.map { |x| x - xbar } }
  yvar = ycol.map { |y| y - ybar }
  # w = (y X^T) (X X^T)^-1
  weights = yvar.dot(xvars).dot(xvars.dot(xvars).invert!)
  offset = ybar - weights.dot(xbars)
  lambda { |xs| offset + weights.dot(xs) }
end

def fit_table(rows, j)
  xcols = rows.transpose
  ycol = xcols.delete_at(j)
  fit(xcols, ycol)
end

def read_table(file)
  rows = []
  while line = file.gets
    rows << line.split(/\s+/).map(&:to_f)
  end
  rows
end

ages = [16, 18, 17]
hours_studied = [100, 10, 20]
grades = [90, 70, 50]

grade_predictor = fit([ages, hours_studied], grades)
puts grade_predictor.call([17, 50])

