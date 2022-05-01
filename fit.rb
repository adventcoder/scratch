module Enumerable
  def mean
    acc = 0.0
    each_with_index { |x, i| acc += (x - acc) / (i + 1) }
    acc
  end
end

class Array
  def dot(other)
    unless other.size == size
      raise ArgumentError, "wrong number of elements: (#{other.size} for #{size})"
    end
    (0 ... size).inject(0) { |sum, i| sum + self[i] * other[i] }
  end

  def cross(other)
    map { |a| other.map { |b| yield a, b } }
  end

  def mmul(other)
    cross(other.transpose, &:dot)
  end

  def minv!
    raise ArgumentError, 'not square matrix' unless all? { |row| row.size == size }
    inv = Array.new(size) { Array.new(size, 0) }
    for n in 0 ... size
      inv[n][n] = 1
    end
    for n in 0 ... size
      if self[n][n] == 0
        i = (n + 1 ... size).find { |i2| self[i2][n] != 0 }
        raise ArgumentError, 'singular matrix' if i.nil?
        self[n], self[i] = self[i], self[n]
      end
      unless self[n][n] == 1
        m = self[n][n]
        self[n][n] = 1
        for j in n + 1 ... size
          self[n][j] = self[n][j].quo(m)
        end
        for j in 0 ... size
          inv[n][j] = inv[n][j].quo(m)
        end
      end
      for i in 0 ... size
        next if i == n
        unless self[i][n] == 0
          m = self[i][n]
          self[i][n] = 0
          for j in n + 1 ... size
            self[i][j] -= self[n][j] * m
          end
          for j in 0 ... size
            inv[i][j] -= inv[n][j] * m
          end
        end
      end
    end
    inv
  end
end

def fit(xcols, ycol)
  xbars = xcols.map { |xs| xs.mean }
  ybar = ycol.mean
  xvars = xcols.zip(xbars).map { |(xcol, xbar)| xcol.map { |x| x - xbar } }
  yvar = ycol.map { |y| y - ybar }
  weights = [yvar].mmul(xvars.transpose).mmul(xvars.mmul(xvars.transpose).minv!)[0]
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

